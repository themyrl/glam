#!/bin/bash
#SBATCH --job-name=swin_tiny_ade_pt_test     # job name
#SBATCH --ntasks=8                  # number of MP tasks
#SBATCH --ntasks-per-node=4          # number of MPI tasks per node
#SBATCH --gres=gpu:4                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=02:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_gpu-dev
#SBATCH --output=logs/swin_tiny_ade_pt_test%j.out # output file name
#SBATCH --error=logs/swin_tiny_ade_pt_test%j.err  # error file name

set -x


cd $WORK/transseg2d
module purge
module load cuda/10.1.2



# CONFIG="configs/orininal_swin/upernet_swin_tiny_pt_patch4_window7_512x512_160k_ade20k.py"
CONFIG="work_dirs/upernet_swin_tiny_pt_patch4_window7_512x512_160k_ade20k/upernet_swin_tiny_pt_patch4_window7_512x512_160k_ade20k.py"
CHECK="work_dirs/upernet_swin_tiny_pt_patch4_window7_512x512_160k_ade20k/iter_160000.pth" 
TMPDIR="/gpfsscratch/rech/arf/unm89rb/tmpdir"

srun /gpfslocalsup/pub/idrtools/bind_gpu.sh python -u tools/test.py $CONFIG $CHECK --eval mIoU --tmpdir $TMPDIR --options model.pretrained="pretrained_models/swin_tiny_patch4_window7_224.pth" --launcher="slurm" ${@:3}
