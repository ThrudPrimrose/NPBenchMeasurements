#!/bin/bash
#SBATCH --job-name=aL
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

mkdir -p logs results

# -------------------------------
# Load GCC 14.2 from Spack
# -------------------------------
spack load gcc@14.2.0
spack load openssl
spack load python@3.12.9%gcc@14.2
alias python=python3.12
echo "Compiler:"
gcc --version

# -------------------------------
# OpenMP configuration
# -------------------------------
export OMP_NUM_THREADS=128
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_DYNAMIC=false



echo "====================================="
echo "Job ID:      ${SLURM_JOB_ID}"
echo "OMP threads: ${OMP_NUM_THREADS}"
echo "Node(s):     ${SLURM_NODELIST}"
echo "====================================="

# -------------------------------
# Run benchmark
# -------------------------------

#cp -R ../../../npbench npbench && chmod 777 npbench
cd npbench
python run_framework.py --f dace_cpu -p L
python run_framework.py --f jax -p L
python run_framework.py --f numba -p L
python run_framework.py --f numpy -p L
python run_framework.py --f pythran -p L