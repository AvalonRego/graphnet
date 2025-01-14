#!/bin/bash -l
#
# Python MPI4PY example job script for MPCDF Raven.
# May use more than one node.
#
#SBATCH -o ./jobs/TT1.%j.out
#SBATCH -e ./jobs/TT1.%j.err
#SBATCH -D ./
#SBATCH -J TT
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=8:00:00
#SBATCH --mem=120000
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=all
#SBATCH --mail-user=arego@mpcdf.mpg.de

module purge
module load anaconda/3/2023.03 cuda/12.1 gcc/12


source /raven/u/arego/graphnet-venv/bin/activate

# Run the Python script
srun python3 /u/arego/graphnet/examples/05_liquido/pone_h5.py sqlite

echo "job finished"