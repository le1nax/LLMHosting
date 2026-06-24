#!/usr/bin/zsh
#SBATCH --account=truhnlab
#SBATCH --partition=truhnlab
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128
#SBATCH --time=0:30:00
#SBATCH --job-name=debug_vllm
#SBATCH --output=log.debug.%J.txt

# Debug-Job: kleines, bereits gecachtes Modell auf 4 Knoten (TP=4 x PP=4) verteilt,
# um dieselbe Multi-Node-Infrastruktur (Ray, Cross-Node-NCCL, JIT-Cache-Locks)
# wie beim grossen GLM-5.2 zu testen -- aber ohne 20 min Ladezeit.

source .venv/bin/activate

set -e
PORT=6379
VLLM_PORT=8000

# --- Netzwerk / NCCL (identisch zu runGLM.sbatch) ---
export NCCL_SOCKET_IFNAME=ib1
export GLOO_SOCKET_IFNAME=ib1
export NCCL_IB_DISABLE=1
export NCCL_NET_PLUGIN=none
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_FAMILY=AF_INET
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,ENV
export HF_HOME="/hpcwork/p0021834/workspace_patrick/hf_home"

# --- CUDA / Host-Compiler (identisch) ---
export CUDA_HOME=/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/ISV/CUDA/13.0.2
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CC=gcc
export CXX=g++
export CUDAHOSTCXX=g++

MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

# --- Knotenliste ---
nodes=(${(f)"$(scontrol show hostnames $SLURM_JOB_NODELIST)"})
head_node=${nodes[1]}
head_ip=$(srun --nodes=1 --ntasks=1 -w $head_node hostname --ip-address | awk '{print $1}')
echo "Head: $head_node ($head_ip)"

# --- Ray-Head ---
srun --nodes=1 --ntasks=1 -w $head_node \
    ray start --head --node-ip-address=$head_ip --port=$PORT \
    --num-gpus=4 --num-cpus=$SLURM_CPUS_PER_TASK --block &
sleep 15

# --- Ray-Worker ---
for i in {2..$SLURM_JOB_NUM_NODES}; do
    node=${nodes[$i]}
    echo "Worker: $node"
    srun --nodes=1 --ntasks=1 -w $node \
        ray start --address=$head_ip:$PORT \
        --num-gpus=4 --num-cpus=$SLURM_CPUS_PER_TASK --block &
    sleep 5
done
sleep 15
ray status

# --- vLLM: gleiche Multi-Node-Topologie wie GLM ---
vllm serve $MODEL \
    --served-model-name debug \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 4 \
    --distributed-executor-backend ray \
    --disable-custom-all-reduce \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --trust-remote-code \
    --host 0.0.0.0 --port $VLLM_PORT
