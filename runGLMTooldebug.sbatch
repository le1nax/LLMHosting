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

source .venv/bin/activate

set -e
#MODEL=$HPCWORK/models/GLM-5.2-FP8
PORT=6379                            # Ray-Port
VLLM_PORT=52223                      # OpenAI-API-Port

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

# 

# --- Knotenliste (zsh: 1-indiziert, ${(f)...} splittet auf Zeilen) ---
nodes=(${(f)"$(scontrol show hostnames $SLURM_JOB_NODELIST)"})
head_node=${nodes[1]}
head_ip=$(srun --nodes=1 --ntasks=1 -w $head_node hostname --ip-address | awk '{print $1}')
echo "Head: $head_node ($head_ip)"

# --- DEBUG: RAM-/shm-Monitoring pro Node nach stdout -----------------------------
# Schreibt alle 30s JE NODE: RAM-Auslastung, /dev/shm-Belegung und die Top-5
# RSS-Prozesse. Damit wird sichtbar, WAS ueber die Laufzeit den Host-RAM fuellt
# (Verdacht: langsamer Leak -> nach Stunden SIGBUS/OOM). Auswertung spaeter z.B.:
#   grep MEMMON log.<JID>.txt
# --overlap: laeuft parallel zu den Ray-Steps auf denselben Nodes; --gres=none: keine GPU.
srun --overlap --gres=none --cpus-per-task=1 \
     --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 bash -c '
    host=$(hostname)
    while true; do
        ts=$(date "+%H:%M:%S")
        tot=$(grep MemTotal /proc/meminfo | tr -dc 0-9)
        avail=$(grep MemAvailable /proc/meminfo | tr -dc 0-9)
        usedpct=$(( (tot - avail) * 100 / tot ))
        echo "[MEMMON $host $ts] RAM used=${usedpct}% avail=$(( avail/1024/1024 ))G/$(( tot/1024/1024 ))G | shm: $(df -h /dev/shm 2>/dev/null | tail -1 | tr -s " ")"
        ps -eo rss=,comm= --sort=-rss 2>/dev/null | head -5 | while read rss comm; do
            echo "[MEMMON $host $ts]   $(( rss/1024 )) MB  $comm"
        done
        sleep 30
    done
' &

# --- Ray-Head starten ---
srun --nodes=1 --ntasks=1 -w $head_node \
    ray start --head --node-ip-address=$head_ip --port=$PORT \
    --num-gpus=4 --num-cpus=$SLURM_CPUS_PER_TASK --block &
sleep 15

# --- Ray-Worker auf den restlichen Knoten ---
for i in {2..$SLURM_JOB_NUM_NODES}; do
    node=${nodes[$i]}
    echo "Worker: $node"
    srun --nodes=1 --ntasks=1 -w $node \
        ray start --address=$head_ip:$PORT \
        --num-gpus=4 --num-cpus=$SLURM_CPUS_PER_TASK --block &
    sleep 5
done
sleep 15
ray status   # sollte 16 GPUs zeigen

# --- vLLM auf dem Head-Node (das Batch-Skript läuft hier) ---
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

# source .venv/bin/activate
# export HF_HOME="/hpcwork/p0021834/workspace_patrick/hf_home"

# python3 -m vllm.entrypoints.openai.api_serve \
#     --model zai-org/GLM-5.2 \

