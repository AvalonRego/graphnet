#!/bin/bash -l
# Standard output and error:
#SBATCH -o /u/arego/graphnet/examples/04_training/job/T3Out.%j
#SBATCH -e /u/arego/graphnet/examples/04_training/job/T3Err.%j
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J test_model
#
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#
# --- default case: use a single GPU on a shared node ---
# #SBATCH --gres=gpu:a100:1
# #SBATCH --cpus-per-task=18
# #SBATCH --mem=125000
#
# --- uncomment to use 2 GPUs on a shared node ---
# #SBATCH --gres=gpu:a100:2
# #SBATCH --cpus-per-task=36
# #SBATCH --mem=250000
#
# --- uncomment to use 4 GPUs on a full node ---
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=72
#SBATCH --mem=500000
#
#SBATCH --mail-type=all
#SBATCH --mail-user=arego@mpcdf.mpg.de
#SBATCH --time=0:10:00

module purge
module load anaconda/3/2023.03 cuda/12.1 gcc/12

source /raven/u/arego/graphnet-venv/bin/activate

export WANDB_MODE=offline

mpath=/u/arego/ptmp_link/Class/train_model_without_configs/dynedge_type_example_wandb/model_config.yml
dpath=/u/arego/ptmp_link/Class/train_model_without_configs/dynedge_type_example_wandb/dataset_config.yml
srun python3 /u/arego/graphnet/examples/04_training/Load.py \
 --gpus 0 1 2 3 --num-workers 18 --target type --wandb \
 --model-config "$mpath"\
 --dataset-config "$dpath"

echo "job finished"
