#!/bin/bash
#SBATCH --job-name=sft-alce
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --partition=batch_gpu
#SBATCH --mem=64GB
#SBATCH --output=sft-alce-%j.out
#SBATCH --error=sft-alce-%j.err

cd /path/to/citation-sft
source ./.venv/bin/activate

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

srun accelerate launch \
    --multi_gpu \
    --num_processes=4 \
    --num_machines=2 \
    --machine_rank=$SLURM_NODEID \
    --main_process_ip=$MASTER_ADDR \
    --main_process_port=$MASTER_PORT \
    -m citation_sft.train