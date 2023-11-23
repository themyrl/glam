#!/bin/bash
#SBATCH --job-name=INAT_BASE     # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=35:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --qos=qos_gpu-t4
#SBATCH --output=logs/%A_%a.out # output file name
#SBATCH --error=logs/%A_%a.err  # error file name
#SBATCH --array=0

set -x

cd $WORK/transseg2d


module purge
module load pytorch-gpu/py3/1.8.0

srun python testslurm.py
