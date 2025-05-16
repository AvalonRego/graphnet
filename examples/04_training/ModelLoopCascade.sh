#!/bin/bash -l
# Standard output and error:
#SBATCH -o /u/arego/graphnet/examples/04_training/job/Training.%j.out
#SBATCH -e /u/arego/graphnet/examples/04_training/job/Training.%j.err
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J trainingg
#
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#
# # --- default case: use a single GPU on a shared node ---
# # SBATCH --gres=gpu:a100:1
# # SBATCH --cpus-per-task=18
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
#SBATCH --time=24:00:00

module purge
module load anaconda/3/2023.03 cuda/12.1 gcc/12

source /raven/u/arego/graphnet-venv/bin/activate
export WANDB_MODE=offline

# Define paths
DATA_PATH="/u/arego/ptmp_link/RC4K1_100_parquet"
SAVE_PATH="/u/arego/ptmp_link/Class/TrainingLoop/Cascade"
mkdir -p "$SAVE_PATH"

srun python3 /u/arego/graphnet/examples/04_training/ModelLoop.py \
--gpus 0 1 2 3 --num-workers 18 --target type --wandb \
--batch-size 32 --save_path "$SAVE_PATH" --path "$DATA_PATH"

echo "Job $SLURM_JOB_ID finished"

