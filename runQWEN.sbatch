#!/usr/bin/zsh
#SBATCH --account=truhnlab
#SBATCH --partition=truhnlab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128          # an Node-Spec anpassen
#SBATCH --mem=0                       # gesamtes Node-RAM: Checkpoint (~220 GB Lustre) kann so geprefetcht werden
#SBATCH --time=49:00:00
#SBATCH --job-name=qwen35_vllm
#SBATCH --output=log_qwen.%J.txt

source .venv/bin/activate

set -e
VLLM_PORT=52223                      # OpenAI-API-Port

# Single-Node, 4 GPUs, reines Tensor-Parallel (TP=4) ueber den mp-Executor:
# kein Ray, keine Pipeline-Parallelitaet, keine Inter-Node-IB/NCCL-Workarounds noetig.
# NCCL nutzt intra-node NVLink/P2P automatisch.

export HF_HOME="/hpcwork/p0021834/workspace_patrick/hf_home"

# --- CUDA-Toolkit fuer JIT-Kernels (FlashInfer/Triton): nvcc zur JIT-Kompilierung ---
export CUDA_HOME=/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/ISV/CUDA/13.0.2
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# --- Host-Compiler fuer nvcc/JIT (FlashInfer): Intel icx wird von CUDA 13 abgelehnt -> gcc nutzen ---
export CC=gcc
export CXX=g++
export CUDAHOSTCXX=g++

# --- JIT-Caches auf NODE-LOKALES NVMe (xfs) statt NFS-Home (schneller, kein NFS-Locking) ---
JIT_CACHE=${TMPDIR:-/tmp}/vllm_jit
export FLASHINFER_WORKSPACE_BASE=$JIT_CACHE
export DG_JIT_CACHE_DIR=$JIT_CACHE/deep_gemm
export TRITON_CACHE_DIR=$JIT_CACHE/triton
export TORCHINDUCTOR_CACHE_DIR=$JIT_CACHE/inductor
export VLLM_CACHE_ROOT=$JIT_CACHE/vllm

MODEL=Qwen/Qwen3.5-397B-A17B-GPTQ-Int4
#export HF_HUB_OFFLINE=1              # Compute-Nodes haben kein Internet

# --- vLLM (laeuft direkt auf der einen allokierten Node) ---
# Speicher: Gewichte ~218 GB (Int4) auf 4x80 GB = 320 GB -> bei util 0.88 ~55-60 GB KV.
# KV ist dank 2 KV-Heads (GQA) sehr guenstig (~60 KB/Token fp8) -> volle 262k passen problemlos.
vllm serve $MODEL \
    --served-model-name qwen3.5-397b \
    --quantization moe_wna16 \
    --tensor-parallel-size 4 \
    --kv-cache-dtype fp8 \
    --max-model-len 262144 \
    --gpu-memory-utilization 0.88 \
    --max-num-seqs 64 \
    --mm-processor-cache-gb 0 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --structured-outputs-config '{"backend": "xgrammar"}' \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --host 0.0.0.0 --port $VLLM_PORT
