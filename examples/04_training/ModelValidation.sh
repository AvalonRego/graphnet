#!/bin/bash -l

#SBATCH -o /u/arego/graphnet/examples/04_training/job/Val.%A_%a.out
#SBATCH -e /u/arego/graphnet/examples/04_training/job/Val.%A_%a.err
#SBATCH -D ./
#SBATCH -J Validation
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125000
#SBATCH --mail-type=all
#SBATCH --mail-user=arego@mpcdf.mpg.de
#SBATCH --time=24:00:00
#SBATCH --array=0-5

module purge
module load anaconda/3/2023.03 cuda/12.1 gcc/12

source /raven/u/arego/graphnet-venv/bin/activate

# Define model path
MODEL_PATH="/u/arego/ptmp_link/Class/RNN/MIX/20250318_173954"

# Define an array of data paths
DATA_PATHS=(
    "/u/arego/ptmp_link/TestT"
    "/u/arego/ptmp_link/TestC"
    "/u/arego/ptmp_link/TestM"
    "/u/arego/ptmp_link/R1T4K_100_parquet"
    "/u/arego/ptmp_link/RC4K1_100_parquet"
    "/u/arego/ptmp_link/MIX"
)

echo "Running $DATA_PATH"

# Get data path based on SLURM array task ID
DATA_PATH=${DATA_PATHS[$SLURM_ARRAY_TASK_ID]}

srun python3 /u/arego/graphnet/examples/04_training/RNNValidation.py \
    --gpus 0 --num-workers 18 --target type \
    --batch-size 32 --path "$DATA_PATH" --model-path "$MODEL_PATH"

echo "Job \$SLURM_JOB_ID for data path $DATA_PATH finished"
