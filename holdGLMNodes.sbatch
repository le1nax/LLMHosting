#!/usr/bin/zsh
#SBATCH --account=truhnlab
#SBATCH --partition=truhnlab
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --time=30-00:00:00
#SBATCH --job-name=glm52_vllm
#SBATCH --nodelist=i25g0005,i25g0006,i25g0008,i25g0009
#SBATCH --output=log.hold.%J.txt
#
# holdGLMNodes.sbatch -- placeholder job to keep i25g[0003-0006] allocated to
# this account while the HF_HOME token-permission issue (see log.1918835.txt)
# is sorted out, so the real runGLMStable.sbatch deployment doesn't have to
# re-queue from scratch. Does no GPU work; just holds the allocation.
# Cancel this job (scancel <jobid>) once ready to submit the real deployment.

echo "Holding nodes $(scontrol show hostnames $SLURM_JOB_NODELIST) for job glm52_vllm placeholder."
sleep infinity
