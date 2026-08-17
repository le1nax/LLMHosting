#!/usr/bin/zsh
#SBATCH --account=truhnlab
#SBATCH --partition=truhnlab
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128          # an Node-Spec anpassen
#SBATCH --mem=0                       # gesamtes Node-RAM anfordern: /dev/shm (vLLM ShmRingBuffer fuer Inter-Rank-Broadcast) zaehlt gegen das cgroup-Limit -> ohne --mem droht nach Stunden OOM/SIGBUS im Ray-Worker
#SBATCH --time=49:00:00
#SBATCH --job-name=glm52_vllm
#SBATCH --output=log.%J.txt
#
# runGLMStable.sbatch -- wie runGLMTool.sbatch, aber mit dem in RESEARCHLOG.md
# (Job 1881487) beschriebenen SIGBUS-Fix: venv + CPython-Interpreter laufen
# node-lokal von NVMe statt ueber NFS ($HPCWORK), plus Startup-Watchdog
# (Job 1881087: Fail-Fast statt endlosem Haengen) und begrenzte Neustarts.
#
# Voraussetzung: $PROJECT_DIR/.venv-python313 muss existieren (einmaliger
# Kopiervorgang der CPython-Distribution -- siehe README fuer den Befehl).

set -e
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
    echo "Einmaliger Kopiervorgang der CPython-Distribution noetig -- siehe README." >&2
    exit 1
fi

# --- Haenge-Monitor automatisch starten (siehe monitorGLM.sh) ---
# Grund: Am 2026-07-15 hing der Server ~5 min lang spurlos, bevor vLLMs
# eigener interner Timeout ihn beendete -- ohne externe Ueberwachung waeren
# py-spy-Traces und GPU-Zustand zu diesem Zeitpunkt laengst verloren gewesen.
# Nutzt --overlap (wie stop_ray) um sich an denselben Job anzuhaengen.
nohup zsh "$PROJECT_DIR/monitorGLM.sh" "$SLURM_JOB_ID" "$PROJECT_DIR/log.$SLURM_JOB_ID.txt" 60 15 \
    >> "$PROJECT_DIR/monitor.log" 2>&1 &
disown
echo "Haenge-Monitor gestartet (PID $!), beobachtet log.$SLURM_JOB_ID.txt, Snapshots nach diagnostics/"

# --- venv + Interpreter auf JEDEM Node nach node-lokales NVMe kopieren ---
# Grund (RESEARCHLOG, Job 1881487): venv+Interpreter lagen auf NFS ($HPCWORK
# bzw. privates $HOME) -> demand-paged Code-Pages koennen bei transientem
# NFS-Hickup nicht nachgeladen werden -> Kernel liefert SIGBUS -> Ray-Worker
# stirbt -> ganzer vLLM-Server reisst ab. $TMPDIR ist node-lokal (NVMe), aber
# als Pfad-STRING ueber alle Nodes im Job identisch -> unten kann derselbe
# Pfad fuer alle Nodes verwendet werden.
LOCAL_VENV=$TMPDIR/venv
LOCAL_PY=$TMPDIR/python313

echo "Kopiere venv + Interpreter auf node-lokales NVMe (alle Nodes)..."
srun --wait=0 --ntasks-per-node=1 zsh -c "
    set -e
    mkdir -p '$LOCAL_VENV' '$LOCAL_PY'
    cp -a '$PROJECT_DIR/.venv/.' '$LOCAL_VENV/'
    cp -a '$PROJECT_DIR/.venv-python313/.' '$LOCAL_PY/'
    # bin/python neu verlinken: .venv-python313 hat KEIN flaches bin/python3.13 --
    # uv legt es unter einem per-Plattform-Unterordner (cpython-3.13.x-...) ab;
    # dynamisch auffinden (nicht den cpython-3.13-...-Symlink nehmen, der zeigt
    # absolut zurueck auf NFS und wuerde die ganze Kopie hier aushebeln).
    LOCAL_PY_REAL=\$(find '$LOCAL_PY' -maxdepth 1 -type d -name 'cpython-*' | head -1)
    ln -sf \"\$LOCAL_PY_REAL/bin/python3.13\" '$LOCAL_VENV/bin/python'
    sed -i \"s|^home = .*|home = \$LOCAL_PY_REAL/bin|\" '$LOCAL_VENV/pyvenv.cfg'
    # Entry-Point-Skripte (vllm, ray, ...) haben ihr Shebang fest auf einen
    # alten, ebenfalls privaten Pfad einkompiliert -> auf lokale Kopie umbiegen.
    # WICHTIG: uv-Shebangs enden auf '.../bin/python' (OHNE '3'!) -- eine Regex,
    # die nur auf 'python3$' matcht, trifft NIE etwas, und jedes Entry-Point-
    # Skript zeigt dann weiterhin auf NFS (genau der Bug, den dieser Kopiervorgang
    # eigentlich beheben soll).
    grep -rlIP '^#!.*/\\.venv/bin/python3?\$' '$LOCAL_VENV/bin' 2>/dev/null | while read -r f; do
        sed -i \"1s|^#!.*|#!$LOCAL_VENV/bin/python3|\" \"\$f\"
    done
    echo \"[\$(hostname)] venv-Kopie + Repoint fertig (python: \$(readlink -f '$LOCAL_VENV/bin/python'))\"
"
echo "venv-Kopie auf allen Nodes abgeschlossen."

export VIRTUAL_ENV=$LOCAL_VENV
export PATH=$LOCAL_VENV/bin:$PATH
unset PYTHONHOME

# --- NCCL / InfiniBand: Interface auf einem Compute-Node mit `ip addr` prüfen!
export NCCL_SOCKET_IFNAME=ib1        # ib1 ist das einzige IB-IF mit IPv4-Adresse (134.61.53.x)
export GLOO_SOCKET_IFNAME=ib1
export NCCL_IB_DISABLE=1             # Omni-Path (hfi1): NCCL-IB-Verbs-Pfad segfaultet -> TCP-Sockets nutzen
export NCCL_NET_PLUGIN=none          # externes NCCL-Net-Plugin nicht laden
export NCCL_SOCKET_FAMILY=AF_INET    # IPv6 ist defekt (errno 97) -> NCCL auf IPv4 zwingen, sonst Deadlock
export GLOO_SOCKET_FAMILY=AF_INET
export NCCL_DEBUG=INFO               # Diagnose: Transportwahl sichtbar machen
export NCCL_DEBUG_SUBSYS=INIT,NET,ENV
export HF_HOME="/hpcwork/p0021834/workspace_patrick/hf_home"
# HF_HOME/token gehoert cu829455 persoenlich (Modus 660, keine Gruppen-Leserechte) ->
# eigenen Token verwenden statt den gemeinsamen (huggingface_hub bevorzugt HF_TOKEN
# gegenueber der Token-Datei). Die Modell-Gewichte selbst liegen gruppenlesbar unter
# HF_HOME/hub (Gruppe p0021834) und sind unabhaengig davon zugaenglich.
export HF_TOKEN="$(cat $HOME/.cache/huggingface/token)"
# Compute-Nodes haben kein Internet und das Modell liegt bereits vollstaendig im
# Cache -> offline bleiben, damit gar kein Hub-API-/Auth-Call noetig ist.
export HF_HUB_OFFLINE=1

# --- CUDA-Toolkit fuer DeepGEMM: GLM-5.2 (Sparse Attention) braucht nvcc zur JIT-Kompilierung ---
export CUDA_HOME=/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/ISV/CUDA/13.0.2
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# --- Host-Compiler fuer nvcc/JIT (FlashInfer/DeepGEMM): Intel icx wird von CUDA 13 abgelehnt -> gcc nutzen ---
export CC=gcc
export CXX=g++
export CUDAHOSTCXX=g++

# --- JIT-Caches auf NODE-LOKALES NVMe (xfs) statt NFS-Home ---
# Grund: 16 Ranks ueber 4 Knoten sperren beim DeepGEMM/FlashInfer-JIT denselben
# Cache auf dem NFS-Home -> "OSError: [Errno 116] Stale file handle" -> Ranks sterben
# -> Cross-Node-NCCL-Deadlock. Node-lokal vermeidet das Cross-Node-Locking komplett.
JIT_CACHE=${TMPDIR:-/tmp}/vllm_jit
export FLASHINFER_WORKSPACE_BASE=$JIT_CACHE
export DG_JIT_CACHE_DIR=$JIT_CACHE/deep_gemm
export TRITON_CACHE_DIR=$JIT_CACHE/triton
export TORCHINDUCTOR_CACHE_DIR=$JIT_CACHE/inductor
export VLLM_CACHE_ROOT=$JIT_CACHE/vllm

MODEL=zai-org/GLM-5.2-FP8
#export HF_HUB_OFFLINE=1              # Compute-Nodes haben kein Internet

# --- Knotenliste (zsh: 1-indiziert, ${(f)...} splittet auf Zeilen) ---
nodes=(${(f)"$(scontrol show hostnames $SLURM_JOB_NODELIST)"})
head_node=${nodes[1]}
head_ip=$(srun --wait=0 --nodes=1 --ntasks=1 -w $head_node hostname --ip-address | awk '{print $1}')
echo "Head: $head_node ($head_ip)"

start_ray() {
    # --- Ray-Head starten ---
    srun --wait=0 --nodes=1 --ntasks=1 -w $head_node \
        env PATH=$PATH VIRTUAL_ENV=$VIRTUAL_ENV \
        ray start --head --node-ip-address=$head_ip --port=$PORT \
        --num-gpus=4 --num-cpus=$SLURM_CPUS_PER_TASK --block &
    sleep 15

    # --- Ray-Worker auf den restlichen Knoten ---
    for i in {2..$SLURM_JOB_NUM_NODES}; do
        node=${nodes[$i]}
        echo "Worker: $node"
        srun --wait=0 --nodes=1 --ntasks=1 -w $node \
            env PATH=$PATH VIRTUAL_ENV=$VIRTUAL_ENV \
            ray start --address=$head_ip:$PORT \
            --num-gpus=4 --num-cpus=$SLURM_CPUS_PER_TASK --block &
        sleep 5
    done
    sleep 15
    ray status   # sollte 16 GPUs zeigen
}

stop_ray() {
    echo "Stoppe Ray auf allen Nodes..."
    # --overlap: jeder Node hat mit --num-cpus=$SLURM_CPUS_PER_TASK bereits die
    # gesamte Node-CPU-Zuteilung fuer den blockierenden ray-start-Prozess belegt --
    # ohne --overlap haengt srun hier auf ewig in "Requested nodes are busy"
    # (Root Cause des Deadlocks in Job 1918835, siehe RESEARCHLOG).
    srun --wait=0 --overlap --ntasks-per-node=1 env PATH=$PATH VIRTUAL_ENV=$VIRTUAL_ENV ray stop --force || true
    sleep 5
}

# --- Startup-Watchdog: fail-fast statt endlosem Haengen (RESEARCHLOG Job 1881087) ---
# EngineCore kann bei einem toten Worker unbegrenzt auf "shared memory
# broadcast block" warten, ohne dass der Job je abbricht -> GPU-Stunden
# verbrennen als Zombie. Watchdog bricht den Job ab, wenn /v1/models nach
# STARTUP_TIMEOUT Sekunden nicht erreichbar ist.
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
    echo "[watchdog] vLLM nach ${STARTUP_TIMEOUT}s nicht bereit -> breche Job $SLURM_JOB_ID ab." >&2
    scancel $SLURM_JOB_ID
}

# --- vLLM mit begrenzten Neustarts ---
# Restrisiko (RESEARCHLOG): kleine cvmfs-Libs + HF-Weights-mmap waehrend der
# Ladephase koennen weiterhin transient fehlschlagen. Statt den ganzen Job zu
# verlieren, Ray+vLLM bis zu MAX_RESTARTS mal neu starten.
attempt=1
while (( attempt <= MAX_RESTARTS )); do
    echo "=== Versuch $attempt/$MAX_RESTARTS: Ray-Cluster + vLLM starten ==="
    start_ray

    watchdog &
    watchdog_pid=$!

    set +e
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
        --host 0.0.0.0 --port $VLLM_PORT
    rc=$?
    set -e

    kill $watchdog_pid 2>/dev/null || true
    echo "vLLM serve beendet mit Exit-Code $rc (Versuch $attempt)"
    stop_ray

    if (( attempt >= MAX_RESTARTS )); then
        echo "Maximale Anzahl Neustarts erreicht, breche ab."
        exit $rc
    fi
    attempt=$(( attempt + 1 ))
    echo "Neustart in 20s..."
    sleep 20
done
