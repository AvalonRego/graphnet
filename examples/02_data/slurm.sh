#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./job.out.%j
#SBATCH -e ./job.err.%j
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J test_gpu
#
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#
# --- default case: use a single GPU on a shared node ---
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125000
#
# --- uncomment to use 2 GPUs on a shared node ---
# #SBATCH --gres=gpu:a100:2
# #SBATCH --cpus-per-task=36
# #SBATCH --mem=250000
#
# --- uncomment to use 4 GPUs on a full node ---
# #SBATCH --gres=gpu:a100:4
# #SBATCH --cpus-per-task=72
# #SBATCH --mem=500000
#
#SBATCH --mail-type=all
#SBATCH --mail-user=arego@mpcdf.mpg.de
#SBATCH --time=12:00:00

module purge
module load anaconda/3/2023.03 cuda/12.1 gcc/12


source /raven/u/arego/graphnet-venv/bin/activate

# Run the Python script
srun python3  /u/arego/graphnet/examples/02_data/ReadMyDataSet.py parquet

echo "job finished"