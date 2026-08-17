#!/usr/bin/zsh
# runGLMOnHeldNodes.sh -- wie runGLMStable.sbatch, aber wird NICHT per sbatch
# submittet (das wuerde eine NEUE Allokation anfordern). Stattdessen haengt
# sich jeder Schritt per `srun --jobid=$HELD_JOBID` an eine bereits laufende
# Platzhalter-Allokation (z.B. holdGLMNodes.sbatch, `sleep infinity`) an.
#
# Grund: Wenn vLLM abstuerzt, soll nur der vLLM-Step sterben -- die
# Platzhalter-Job-Allokation (und damit die reservierten Nodes) bleibt am
# Leben, solange ihr Hauptskript (sleep infinity) weiterlaeuft. Ein
# gewoehnlicher sbatch-Job wuerde beim Absturz die Nodes sofort freigeben,
# und andere Nutzer koennten sie sich schnappen, bevor der naechste Versuch
# neu queuen kann.
#
# Aufruf: ./runGLMOnHeldNodes.sh <HELD_JOBID>
# Log per Shell-Redirect erfassen, z.B.:
#   nohup zsh runGLMOnHeldNodes.sh 1926058 > log.heldrun.$(date +%Y%m%d_%H%M%S).txt 2>&1 &

set -e

HELD_JOBID=${1:?"Usage: $0 <HELD_JOBID>  (die Platzhalter-Job-ID, z.B. 1926058)"}

PROJECT_DIR=$(pwd)
PORT=6379                            # Ray-Port
VLLM_PORT=52223                      # OpenAI-API-Port
MAX_RESTARTS=3
STARTUP_TIMEOUT=7200                 # 2h: DeepGEMM-JIT-Warmup mit KALTEM Cache auf frischem
                                      # node-lokalem $TMPDIR braucht ~85-90 min (~2.65s/Kernel,
                                      # 1952 Kernel) -- gemessen in log.heldrun.20260714_175250,
                                      # nicht die ~20 min aus debug.sbatch (dort vermutlich warmer Cache)

if [[ ! -d "$PROJECT_DIR/.venv-python313" ]]; then
    echo "ERROR: $PROJECT_DIR/.venv-python313 fehlt." >&2
    exit 1
fi

# --- Haenge-Monitor automatisch starten (siehe monitorGLM.sh) ---
# Grund: Am 2026-07-15 hing der Server ~5 min lang spurlos, bevor vLLMs
# eigener interner Timeout ihn beendete -- ohne externe Ueberwachung waeren
# py-spy-Traces und GPU-Zustand zu diesem Zeitpunkt laengst verloren gewesen.
# Eigenes Logfile per /proc/<pid>/fd/1 ermitteln (Aufrufer leitet per `>` um).
# WICHTIG: nicht /proc/self/fd/1 direkt in einer $(...)-Command-Substitution
# verwenden -- "self" ist dann der Subshell/Pipe-Prozess der Substitution
# selbst (zeigt auf "pipe:[...]"), nicht das Skript. Erst $$ in eine Variable
# fixieren, dann erst in die Substitution einsetzen.
SCRIPT_PID=$$
SELF_LOGFILE=$(readlink -f /proc/$SCRIPT_PID/fd/1 2>/dev/null || true)
if [[ -n "$SELF_LOGFILE" && -f "$SELF_LOGFILE" ]]; then
    nohup zsh "$PROJECT_DIR/monitorGLM.sh" "$HELD_JOBID" "$SELF_LOGFILE" 60 15 \
        >> "$PROJECT_DIR/monitor.log" 2>&1 &
    disown
    echo "Haenge-Monitor gestartet (PID $!), beobachtet $SELF_LOGFILE, Snapshots nach diagnostics/"
else
    echo "WARNUNG: konnte eigenes Logfile nicht ermitteln -- Haenge-Monitor NICHT gestartet." >&2
fi

# --- Job-Kontext des Platzhalter-Jobs abfragen (wir laufen NICHT in seiner Batch-Shell) ---
job_info=$(scontrol show job "$HELD_JOBID" -o)
if [[ -z "$job_info" ]]; then
    echo "ERROR: Job $HELD_JOBID nicht gefunden (schon beendet?)." >&2
    exit 1
fi
job_state=$(echo "$job_info" | grep -oP 'JobState=\K\S+')
if [[ "$job_state" != "RUNNING" ]]; then
    echo "ERROR: Job $HELD_JOBID ist nicht RUNNING (State=$job_state)." >&2
    exit 1
fi
SLURM_JOB_NODELIST=$(echo "$job_info" | grep -oP '(?<![A-Za-z])NodeList=\K\S+')
SLURM_JOB_NUM_NODES=$(echo "$job_info" | grep -oP 'NumNodes=\K\S+')
SLURM_CPUS_PER_TASK=$(echo "$job_info" | grep -oP 'CPUs/Task=\K\S+')
echo "Haenge an Job $HELD_JOBID an: Nodes=$SLURM_JOB_NODELIST NumNodes=$SLURM_JOB_NUM_NODES CpusPerTask=$SLURM_CPUS_PER_TASK"

# --- TMPDIR des Platzhalter-Jobs abfragen: node-lokales NVMe, als Pfad-STRING
# ueber alle Nodes im Job identisch (siehe runGLMStable.sbatch) ---
TMPDIR=$(srun --wait=0 --jobid="$HELD_JOBID" --nodes=1 --ntasks=1 bash -c 'echo $TMPDIR' 2>/dev/null | tail -1)
if [[ -z "$TMPDIR" ]]; then
    echo "ERROR: Konnte TMPDIR von Job $HELD_JOBID nicht ermitteln." >&2
    exit 1
fi
echo "TMPDIR (node-lokal, alle Nodes): $TMPDIR"

LOCAL_VENV=$TMPDIR/venv
LOCAL_PY=$TMPDIR/python313

echo "Kopiere venv + Interpreter auf node-lokales NVMe (alle Nodes)..."
srun --wait=0 --jobid="$HELD_JOBID" --nodes="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 zsh -c "
    set -e
    mkdir -p '$LOCAL_VENV' '$LOCAL_PY'
    cp -a '$PROJECT_DIR/.venv/.' '$LOCAL_VENV/'
    cp -a '$PROJECT_DIR/.venv-python313/.' '$LOCAL_PY/'
    # .venv-python313 hat KEIN flaches bin/python3.13 -- uv legt es unter einem
    # per-Plattform-Unterordner (cpython-3.13.x-...) ab; dynamisch auffinden
    # (nicht den cpython-3.13-...-Symlink nehmen, der zeigt absolut zurueck auf NFS).
    LOCAL_PY_REAL=\$(find '$LOCAL_PY' -maxdepth 1 -type d -name 'cpython-*' | head -1)
    ln -sf \"\$LOCAL_PY_REAL/bin/python3.13\" '$LOCAL_VENV/bin/python'
    sed -i \"s|^home = .*|home = \$LOCAL_PY_REAL/bin|\" '$LOCAL_VENV/pyvenv.cfg'
    # uv-Entry-Point-Shebangs enden auf '.../bin/python' (OHNE '3'!), nicht nur
    # 'python3' -- die urspruengliche Regex hat deshalb nie etwas getroffen und
    # jedes Entry-Point-Skript (vllm, ray, ...) zeigte weiter auf NFS.
    grep -rlIP '^#!.*/\\.venv/bin/python3?\$' '$LOCAL_VENV/bin' 2>/dev/null | while read -r f; do
        sed -i \"1s|^#!.*|#!$LOCAL_VENV/bin/python3|\" \"\$f\"
    done
    echo \"[\$(hostname)] venv-Kopie + Repoint fertig (python: \$(readlink -f '$LOCAL_VENV/bin/python'))\"
"
echo "venv-Kopie auf allen Nodes abgeschlossen."

export VIRTUAL_ENV=$LOCAL_VENV
export PATH=$LOCAL_VENV/bin:$PATH
unset PYTHONHOME

# --- NCCL / InfiniBand ---
export NCCL_SOCKET_IFNAME=ib1
export GLOO_SOCKET_IFNAME=ib1
export NCCL_IB_DISABLE=1
export NCCL_NET_PLUGIN=none
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_FAMILY=AF_INET
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,ENV
export HF_HOME="/hpcwork/p0021834/workspace_patrick/hf_home"
export HF_TOKEN="$(cat $HOME/.cache/huggingface/token)"
export HF_HUB_OFFLINE=1

# --- CUDA-Toolkit fuer DeepGEMM ---
export CUDA_HOME=/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/ISV/CUDA/13.0.2
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export CC=gcc
export CXX=g++
export CUDAHOSTCXX=g++

# --- JIT-Caches auf NODE-LOKALES NVMe ---
JIT_CACHE=${TMPDIR:-/tmp}/vllm_jit
export FLASHINFER_WORKSPACE_BASE=$JIT_CACHE
export DG_JIT_CACHE_DIR=$JIT_CACHE/deep_gemm
export TRITON_CACHE_DIR=$JIT_CACHE/triton
export TORCHINDUCTOR_CACHE_DIR=$JIT_CACHE/inductor
export VLLM_CACHE_ROOT=$JIT_CACHE/vllm

MODEL=zai-org/GLM-5.2-FP8

# --- Knotenliste ---
nodes=(${(f)"$(scontrol show hostnames $SLURM_JOB_NODELIST)"})
head_node=${nodes[1]}
head_ip=$(srun --wait=0 --jobid="$HELD_JOBID" --nodes=1 --ntasks=1 -w $head_node hostname --ip-address | awk '{print $1}')
echo "Head: $head_node ($head_ip)"

start_ray() {
    srun --wait=0 --jobid="$HELD_JOBID" --nodes=1 --ntasks=1 -w $head_node \
        env PATH=$PATH VIRTUAL_ENV=$VIRTUAL_ENV \
        ray start --head --node-ip-address=$head_ip --port=$PORT \
        --num-gpus=4 --num-cpus=$SLURM_CPUS_PER_TASK --block &
    sleep 15

    for i in {2..$SLURM_JOB_NUM_NODES}; do
        node=${nodes[$i]}
        echo "Worker: $node"
        srun --wait=0 --jobid="$HELD_JOBID" --nodes=1 --ntasks=1 -w $node \
            env PATH=$PATH VIRTUAL_ENV=$VIRTUAL_ENV \
            ray start --address=$head_ip:$PORT \
            --num-gpus=4 --num-cpus=$SLURM_CPUS_PER_TASK --block &
        sleep 5
    done
    sleep 15
    # --overlap: der Head-Node hat mit --num-cpus=$SLURM_CPUS_PER_TASK bereits
    # die gesamte Node-CPU-Zuteilung fuer den blockierenden Head-Step belegt --
    # ohne --overlap haengt srun hier auf ewig in "Requested nodes are busy"
    # (gleiche Ursache wie das stop_ray-Deadlock in Job 1918835/RESEARCHLOG).
    srun --wait=0 --jobid="$HELD_JOBID" --overlap --nodes=1 --ntasks=1 -w $head_node \
        env PATH=$PATH VIRTUAL_ENV=$VIRTUAL_ENV ray status
}

stop_ray() {
    echo "Stoppe Ray auf allen Nodes..."
    srun --wait=0 --jobid="$HELD_JOBID" --overlap --nodes="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 env PATH=$PATH VIRTUAL_ENV=$VIRTUAL_ENV ray stop --force || true
    sleep 5
}

# --- Startup-Watchdog: killt nur den vLLM-Step, NICHT den Platzhalter-Job ---
watchdog() {
    local waited=0
    while (( waited < STARTUP_TIMEOUT )); do
        if curl -s -o /dev/null -w '%{http_code}' --max-time 5 \
                "http://${head_ip}:${VLLM_PORT}/v1/models" 2>/dev/null | grep -q 200; then
            echo "[watchdog] vLLM ist bereit nach ${waited}s."
            return 0
        fi
        sleep 15
        (( waited += 15 ))
    done
    echo "[watchdog] vLLM nach ${STARTUP_TIMEOUT}s nicht bereit -> breche nur den vLLM-Step ab (Reservierung Job $HELD_JOBID bleibt bestehen)." >&2
    kill "$vllm_srun_pid" 2>/dev/null || true
}

attempt=1
while (( attempt <= MAX_RESTARTS )); do
    echo "=== Versuch $attempt/$MAX_RESTARTS: Ray-Cluster + vLLM starten (auf gehaltenen Nodes von Job $HELD_JOBID) ==="
    start_ray

    watchdog &
    watchdog_pid=$!

    set +e
    # --overlap noetig: laeuft auf dem Head-Node, dessen CPU-Zuteilung bereits
    # vollstaendig vom blockierenden Ray-Head-Step belegt ist (s.o.).
    srun --wait=0 --jobid="$HELD_JOBID" --overlap --nodes=1 --ntasks=1 -w $head_node \
        env PATH=$PATH VIRTUAL_ENV=$VIRTUAL_ENV HF_HOME=$HF_HOME HF_TOKEN=$HF_TOKEN HF_HUB_OFFLINE=$HF_HUB_OFFLINE \
        vllm serve $MODEL \
        --served-model-name glm-5.2-fp8 \
        --tensor-parallel-size 4 \
        --pipeline-parallel-size 4 \
        --distributed-executor-backend ray \
        --disable-custom-all-reduce \
        --kv-cache-dtype fp8 \
        --max-model-len 128000 \
        --gpu-memory-utilization 0.83 \
        --max-num-seqs 64 \
        --mm-processor-cache-gb 0 \
        --trust-remote-code \
        --enable-auto-tool-choice \
        --structured-outputs-config '{"backend": "xgrammar"}' \
        --tool-call-parser glm47 \
        --reasoning-parser glm45 \
        --host 0.0.0.0 --port $VLLM_PORT &
    vllm_srun_pid=$!
    wait $vllm_srun_pid
    rc=$?
    set -e

    kill $watchdog_pid 2>/dev/null || true
    echo "vLLM serve beendet mit Exit-Code $rc (Versuch $attempt)"
    stop_ray

    if (( attempt >= MAX_RESTARTS )); then
        echo "Maximale Anzahl Neustarts erreicht. Reservierung (Job $HELD_JOBID) bleibt trotzdem bestehen -- Skript beendet sich nur selbst."
        exit $rc
    fi
    attempt=$(( attempt + 1 ))
    echo "Neustart in 20s..."
    sleep 20
done
