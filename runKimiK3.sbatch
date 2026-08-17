#!/usr/bin/zsh
#SBATCH --account=truhnlab
#SBATCH --partition=truhnlab
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --time=49:00:00
#SBATCH --job-name=kimi_k3_vllm
#SBATCH --output=log.kimik3.%J.txt
#SBATCH --beeond
#
# runKimiK3.sbatch -- Kimi-K3 (2.8T params, MXFP4, 896 experts/16 active) auf
# 8 Nodes x 4 H100-80GB (32 GPUs) via vLLM in einem Apptainer-Container.
#
# STATUS: erster Entwurf, NICHT auf echten Nodes getestet -- anders als
# runGLMStable.sbatch (das nach mehreren Runden Debugging jetzt stabil laeuft),
# gibt es hierzu noch keine Praxiserfahrung auf diesem Cluster. Siehe
# Kommentare unten fuer die Annahmen, die einen echten Testlauf brauchen.
#
# Warum Container statt venv: Kimi-K3 braucht vLLM 0.27.0+ (MXFP4/KDA-Attention-
# Support), das noch nicht auf PyPI ist ("Day-0 support", Modell 2 Tage alt).
# vLLMs eigenes Docker-Image vllm/vllm-openai:kimi-k3 enthaelt die richtige
# Version bereits fertig gebaut -> per Apptainer/Singularity ausfuehren statt
# ein bleeding-edge-vLLM in ein venv zu bauen.
#
# Warum die Gewichte NICHT node-lokal (pro Node) kopiert werden (anders als
# bei GLM): Kimi-K3 ist ~1.56 TB gross, node-lokales NVMe pro Node ist nur
# 1.5 TB -> passt nicht mal EINE Kopie drauf.
#
# Stattdessen: --beeond (SBATCH-Flag oben) spannt ueber alle 8 Nodes hinweg
# ein temporaeres, gemeinsames BeeGFS-Dateisystem auf ($BEEOND), das die
# node-lokalen NVMe-Platten ALLER 8 Nodes zu einem einzigen, schnellen,
# parallelen FS aggregiert (8 x 1.5 TB = 12 TB Kapazitaet -- 1.56 TB Gewichte
# passen dort locker rein). Die Gewichte werden EINMAL (mit 10 parallelen
# Workern) von NFS nach $BEEOND kopiert, danach lesen alle 8 Nodes von dort --
# schneller als NFS und ohne Last auf dem geteilten Cluster-Dateisystem.
# Wird beim Jobende automatisch von SLURM aufgeraeumt (siehe --help-Text).
#
# Nur das Container-Image (7.9 GB) wird zusaetzlich node-lokal (in $TMPDIR)
# kopiert (SIGBUS-Schutz, RESEARCHLOG Job 1881487 -- dieselbe Begruendung wie
# fuer GLMs venv-Kopie); das liegt getrennt von $BEEOND, da es pro Node in
# jedem Fall lokal sein muss (Apptainer exec braucht keinen Netzwerkpfad).
#
# Parallelismus-Wahl (TP=4 x DP=8, EP automatisch = TP*DP = 32):
# vLLMs eigene Doku bestaetigt: bei --enable-expert-parallel gilt
# ep_size = tp_size * dp_size, unabhaengig von der TP/DP-Aufteilung.
# TP=4 (intra-Node, NVLink) x DP=8 (eine Node-lokale DP-Rank pro Node, Ray-
# koordiniert) ergibt EP=32 -- volle 32-fache Experten-Sharding UND nur 8
# (TP-4-fach gesplittete) Kopien der Attention/Dense-Gewichte statt 32 volle
# Kopien bei reinem DP=32/TP=1. Vergleich mit offiziellem "DP16+EP16"-Profil
# (vLLM-Blog, GPU-Modell nicht genannt): gleiche Grundidee, auf unsere
# 8 Nodes x 4 GPUs skaliert.
#
# UNGETESTETE ANNAHMEN (vor dem ersten echten Lauf pruefen):
# - --data-parallel-backend ray erwartet vermutlich trotzdem einen vorher
#   gestarteten Ray-Cluster (wie bei GLMs --distributed-executor-backend ray)
#   -- Skript startet ihn vorsorglich genauso wie runGLMStable.sbatch.
# - Ob Apptainer /rwthfs automatisch bindet oder --bind explizit noetig ist:
#   Skript bindet explizit, schadet nicht falls schon automatisch gebunden.
# - MoE-Backend/all2all-Flags (deep_gemm_mega_moe vs flashinfer_trtllm,
#   deepep_v2 vs flashinfer_nvlink_one_sided) sind laut vLLM-Recipe-Seite
#   nur Empfehlungen, keine vollstaendigen Beispielkommandos -- ggf. anpassen.
# - Kein Haenge-Monitor wie monitorGLM.sh: py-spy muesste dafuer INNERHALB
#   des Containers laufen (andere PID-Namespace) -- noch nicht gebaut.
# - --beeond ist neu fuer uns (bisher nie in diesem Repo benutzt): ob $BEEOND
#   tatsaechlich auf allen 8 Nodes identisch gemountet ist, ob 10 parallele
#   cp-Worker fuer 1.56 TB tatsaechlich schneller sind als sequentiell, und
#   ob Apptainer $BEEOND ohne Extra-Bind sieht (--bind unten trotzdem explizit
#   gesetzt) -- alles beim ersten echten Lauf zu verifizieren.
#
# GEFUNDEN in Job 2509534 (2026-07-30): --wait=0 auf JEDEM srun-Call noetig.
# Dieser Cluster hat ein WaitTime-Default von ~5s: sobald EIN Task in einem
# --ntasks-per-node=1-Schritt fertig ist (egal ob Erfolg oder Fehler), killt
# srun nach 5s alle noch laufenden Tasks auf den anderen Nodes -- auch wenn
# die einfach nur etwas laenger brauchen (z.B. normale NFS-Zeitschwankung
# beim Kopieren). Erklärt vermutlich rueckwirkend auch das "ganzer Schritt
# stirbt wegen einem einzelnen Node"-Verhalten, das bei GLMs venv-Kopie
# schon frueher beobachtet wurde (dort als Ressourcen-Problem gedeutet,
# war aber wahrscheinlich derselbe WaitTime-Effekt).

set -e
PROJECT_DIR=$(pwd)
DATA_DIR=/rwthfs/rz/cluster/hpcwork/p0021834/workspace_geiger
SIF=$DATA_DIR/kimi-k3.sif
HF_HOME_SHARED=$DATA_DIR/hf_home
MODEL=moonshotai/Kimi-K3
PORT=6379                            # Ray-Port
VLLM_PORT=52223                      # OpenAI-API-Port
MAX_RESTARTS=3
STARTUP_TIMEOUT=10800                # 3h: konservativ angesetzt, da wir keine
                                      # gemessenen Ladezeiten fuer dieses Modell
                                      # haben (1.56 TB Gewichte + MXFP4-Warmup)

# --- Logging-Helfer: jede Zeile mit Zeitstempel, damit sich im Fehlerfall
# genau rekonstruieren laesst, WANN welcher Schritt begann/endete. ---
klog() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

klog "=== runKimiK3.sbatch startet: Job=$SLURM_JOB_ID Nodes=$SLURM_JOB_NODELIST ==="
klog "SIF=$SIF HF_HOME=$HF_HOME_SHARED MODEL=$MODEL TMPDIR=$TMPDIR"
klog "TP=4 DP=8 (EP=32 implizit) MAX_RESTARTS=$MAX_RESTARTS STARTUP_TIMEOUT=${STARTUP_TIMEOUT}s"

if [[ ! -f "$SIF" ]]; then
    klog "ERROR: $SIF fehlt."
    exit 1
fi
if [[ ! -d "$HF_HOME_SHARED/hub/models--moonshotai--Kimi-K3" ]]; then
    klog "ERROR: Kimi-K3-Gewichte fehlen unter $HF_HOME_SHARED."
    exit 1
fi
klog "Vorab-Checks OK: SIF und Gewichte-Verzeichnis vorhanden."

# --- Container-Image auf JEDEM Node nach node-lokales NVMe kopieren ---
# (SIGBUS-Schutz wie bei GLMs venv-Kopie, RESEARCHLOG Job 1881487; hier nur
# das 7.9 GB Image, NICHT die 1.56 TB Gewichte -- passen nicht node-lokal.)
LOCAL_SIF=$TMPDIR/kimi-k3.sif
EXPECTED_SIF_SIZE=$(stat -c %s "$SIF")
klog "Kopiere Container-Image (${EXPECTED_SIF_SIZE} bytes) auf node-lokales NVMe (alle Nodes)..."
srun --wait=0 --nodes=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 zsh -c "
    set -e
    echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] [\$(hostname)] GPU-Sichtbarkeitscheck:\"
    nvidia-smi -L 2>&1 || echo \"[\$(hostname)] WARNUNG: nvidia-smi fehlgeschlagen\"
    cp '$SIF' '$LOCAL_SIF'
    actual_size=\$(stat -c %s '$LOCAL_SIF')
    if [[ \"\$actual_size\" != '$EXPECTED_SIF_SIZE' ]]; then
        echo \"[\$(hostname)] ERROR: Image-Kopie unvollstaendig (erwartet $EXPECTED_SIF_SIZE, erhalten \$actual_size)\" >&2
        exit 1
    fi
    echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] [\$(hostname)] Image-Kopie fertig und groessengeprueft (\$actual_size bytes)\"
    apptainer exec --nv '$LOCAL_SIF' python3 -c 'import vllm; print(\"[\" + __import__(\"socket\").gethostname() + \"] vLLM-Version im Container:\", vllm.__version__)' 2>&1 \
        || echo \"[\$(hostname)] WARNUNG: Container-Smoke-Test (vllm import) fehlgeschlagen\"
"
copy_rc=$?
if (( copy_rc != 0 )); then
    klog "ERROR: Image-Kopie/Smoke-Test schlug auf mindestens einem Node fehl (rc=$copy_rc). Breche ab."
    exit 1
fi
klog "Image-Kopie + Container-Smoke-Test auf allen Nodes abgeschlossen."

# --- Gewichte EINMAL parallel von NFS nach $BEEOND kopieren (gemeinsames FS
# ueber alle 8 Nodes, siehe Kommentar oben) ---
if [[ -z "${BEEOND:-}" ]]; then
    klog "ERROR: \$BEEOND ist nicht gesetzt -- --beeond-Flag hat evtl. nicht gegriffen. Breche ab."
    exit 1
fi
klog "\$BEEOND = $BEEOND"
df -h "$BEEOND" 2>&1 | while IFS= read -r l; do klog "  [df BEEOND] $l"; done

SNAPSHOT_HASH=$(basename $(find "$HF_HOME_SHARED/hub/models--moonshotai--Kimi-K3/snapshots" -maxdepth 1 -mindepth 1 -type d | head -1))
SRC_SNAP="$HF_HOME_SHARED/hub/models--moonshotai--Kimi-K3/snapshots/$SNAPSHOT_HASH"
HF_HOME_BEEOND="$BEEOND/hf_home"
DST_SNAP="$HF_HOME_BEEOND/hub/models--moonshotai--Kimi-K3/snapshots/$SNAPSHOT_HASH"
klog "Kopiere Gewichte von $SRC_SNAP nach $DST_SNAP (10 parallele Worker, -L um Blob-Symlinks aufzuloesen)..."

mkdir -p "$DST_SNAP"
copy_t0=$(date +%s)
find "$SRC_SNAP" -maxdepth 1 -mindepth 1 \( -type f -o -type l \) -print0 \
    | xargs -0 -P 10 -I{} cp -L {} "$DST_SNAP/"
weights_copy_rc=$?
copy_t1=$(date +%s)
klog "Gewichte-Kopie abgeschlossen in $(( copy_t1 - copy_t0 ))s, exit code $weights_copy_rc"

if (( weights_copy_rc != 0 )); then
    klog "ERROR: Gewichte-Kopie nach \$BEEOND fehlgeschlagen (rc=$weights_copy_rc). Breche ab."
    exit 1
fi

# Groessen-Gegencheck: Summe aller Dateien in Ziel vs. Quelle (echte Bytes,
# nicht ueber Symlinks -- -L bei du folgt Links wie beim Kopieren mit -L).
SRC_BYTES=$(du -cbL "$SRC_SNAP" 2>/dev/null | tail -1 | cut -f1)
DST_BYTES=$(du -cb "$DST_SNAP" 2>/dev/null | tail -1 | cut -f1)
klog "Groessenvergleich: Quelle=${SRC_BYTES}B Ziel=${DST_BYTES}B"
if [[ "$SRC_BYTES" != "$DST_BYTES" ]]; then
    klog "ERROR: Groessen stimmen nicht ueberein -- Kopie vermutlich unvollstaendig. Breche ab."
    exit 1
fi
klog "Gewichte-Kopie groessengeprueft OK."

# --- Node-lokale JIT-/Runtime-Caches (gleiche Begruendung wie GLM) ---
JIT_CACHE=${TMPDIR:-/tmp}/vllm_jit
export FLASHINFER_WORKSPACE_BASE=$JIT_CACHE
export TRITON_CACHE_DIR=$JIT_CACHE/triton
export TORCHINDUCTOR_CACHE_DIR=$JIT_CACHE/inductor
export VLLM_CACHE_ROOT=$JIT_CACHE/vllm

# --- NCCL / InfiniBand: dieselben Workarounds wie runGLMStable.sbatch ---
export NCCL_SOCKET_IFNAME=ib1
export GLOO_SOCKET_IFNAME=ib1
export NCCL_IB_DISABLE=1
export NCCL_NET_PLUGIN=none
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_FAMILY=AF_INET
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,ENV

# --- HuggingFace: HF_HOME zeigt jetzt auf $BEEOND (Gewichte gerade dorthin
# kopiert), nicht mehr auf NFS -- eigener Token (kein Zugriffsproblem hier,
# eigenes hf_home) ---
export HF_HOME=$HF_HOME_BEEOND
export HF_TOKEN="$(cat $HOME/.cache/huggingface/token)"
export HF_HUB_OFFLINE=1              # Compute-Nodes haben kein Internet

# --- Kimi-K3-spezifisch (siehe recipes.vllm.ai/moonshotai/Kimi-K3) ---
export VLLM_ALL2ALL_BACKEND=deepep_v2   # RDMA-Pfad; bei Problemen auf
                                          # flashinfer_nvlink_one_sided wechseln

APPTAINER_EXEC="apptainer exec --nv \
    --bind /rwthfs/rz/cluster/hpcwork/p0021834:/rwthfs/rz/cluster/hpcwork/p0021834 \
    --bind $TMPDIR:$TMPDIR \
    --bind $BEEOND:$BEEOND \
    --env HF_HOME=$HF_HOME --env HF_TOKEN=$HF_TOKEN --env HF_HUB_OFFLINE=1 \
    --env NCCL_SOCKET_IFNAME --env GLOO_SOCKET_IFNAME --env NCCL_IB_DISABLE \
    --env NCCL_NET_PLUGIN --env NCCL_SOCKET_FAMILY --env GLOO_SOCKET_FAMILY \
    --env NCCL_DEBUG --env NCCL_DEBUG_SUBSYS --env VLLM_ALL2ALL_BACKEND \
    --env FLASHINFER_WORKSPACE_BASE --env TRITON_CACHE_DIR \
    --env TORCHINDUCTOR_CACHE_DIR --env VLLM_CACHE_ROOT \
    $LOCAL_SIF"

# --- Knotenliste ---
nodes=(${(f)"$(scontrol show hostnames $SLURM_JOB_NODELIST)"})
head_node=${nodes[1]}
klog "Knotenliste: ${nodes[*]}"
head_ip=$(srun --wait=0 --nodes=1 --ntasks=1 -w $head_node hostname --ip-address | awk '{print $1}')
klog "Head: $head_node ($head_ip)"

start_ray() {
    klog "start_ray: starte Ray-Head auf $head_node ($head_ip)"
    srun --wait=0 --nodes=1 --ntasks=1 -w $head_node \
        $=APPTAINER_EXEC ray start --head --node-ip-address=$head_ip --port=$PORT \
        --num-gpus=4 --num-cpus=$SLURM_CPUS_PER_TASK --block &
    sleep 15

    for i in {2..$SLURM_JOB_NUM_NODES}; do
        node=${nodes[$i]}
        klog "start_ray: starte Ray-Worker auf $node"
        srun --wait=0 --nodes=1 --ntasks=1 -w $node \
            $=APPTAINER_EXEC ray start --address=$head_ip:$PORT \
            --num-gpus=4 --num-cpus=$SLURM_CPUS_PER_TASK --block &
        sleep 5
    done
    sleep 15
    klog "start_ray: ray status (sollte 32 GPUs zeigen):"
    srun --wait=0 --overlap --nodes=1 --ntasks=1 -w $head_node $=APPTAINER_EXEC ray status 2>&1 | while IFS= read -r l; do klog "  [ray status] $l"; done
}

stop_ray() {
    klog "stop_ray: stoppe Ray auf allen Nodes..."
    srun --wait=0 --overlap --ntasks-per-node=1 $=APPTAINER_EXEC ray stop --force
    ray_stop_rc=$?
    klog "stop_ray: exit code $ray_stop_rc"
    sleep 5
}

# --- Startup-Watchdog (gleiches Prinzip wie runGLMStable.sbatch) ---
watchdog() {
    local waited=0
    klog "watchdog: gestartet, pruefe alle 30s http://${head_ip}:${VLLM_PORT}/v1/models"
    while (( waited < STARTUP_TIMEOUT )); do
        code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://${head_ip}:${VLLM_PORT}/v1/models" 2>/dev/null)
        if (( waited % 300 == 0 )); then
            klog "watchdog: noch nicht bereit nach ${waited}s (letzter http_code=${code:-<keine Antwort>})"
        fi
        if [[ "$code" == "200" ]]; then
            klog "watchdog: vLLM ist bereit nach ${waited}s."
            return 0
        fi
        sleep 30
        (( waited += 30 ))
    done
    klog "watchdog: vLLM nach ${STARTUP_TIMEOUT}s nicht bereit -> breche Job $SLURM_JOB_ID ab."
    scancel $SLURM_JOB_ID
}

attempt=1
while (( attempt <= MAX_RESTARTS )); do
    klog "=== Versuch $attempt/$MAX_RESTARTS: Ray-Cluster + vLLM (Kimi-K3) starten ==="
    start_ray

    watchdog &
    watchdog_pid=$!

    VLLM_CMD=(srun --wait=0 --nodes=1 --ntasks=1 -w $head_node \
        $=APPTAINER_EXEC vllm serve $MODEL \
        --served-model-name kimi-k3 \
        --tensor-parallel-size 4 \
        --data-parallel-size 8 \
        --data-parallel-size-local 1 \
        --data-parallel-backend ray \
        --enable-expert-parallel \
        --trust-remote-code \
        --max-model-len 262144 \
        --gpu-memory-utilization 0.85 \
        --max-num-seqs 32 \
        --enable-auto-tool-choice \
        --host 0.0.0.0 --port $VLLM_PORT)
    klog "vllm serve Kommando: ${VLLM_CMD[*]}"

    set +e
    "${VLLM_CMD[@]}"
    rc=$?
    set -e

    kill $watchdog_pid 2>/dev/null || true
    klog "vLLM serve beendet mit Exit-Code $rc (Versuch $attempt/$MAX_RESTARTS)"
    stop_ray

    if (( attempt >= MAX_RESTARTS )); then
        klog "Maximale Anzahl Neustarts erreicht, breche ab."
        exit $rc
    fi
    attempt=$(( attempt + 1 ))
    klog "Neustart in 20s..."
    sleep 20
done
