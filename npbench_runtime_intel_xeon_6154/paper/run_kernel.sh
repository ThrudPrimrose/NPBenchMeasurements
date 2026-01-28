#!/bin/bash
#SBATCH --job-name=iPaper
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --mem=0
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

mkdir -p logs results

# -------------------------------
# Load GCC 14.2 from Spack
# -------------------------------
spack load gcc@14.2
spack load python@3.12.9%gcc@14.2
spack load sqlite

echo "Compiler:"
gcc --version

# -------------------------------
# OpenMP configuration
# -------------------------------
export OMP_NUM_THREADS=36
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
cd npbench
#python3.12 run_framework.py --f dace_cpu -p paper -e True
python3.12 run_framework.py --f jax -p paper -e True
python3.12 run_framework.py --f numba -p paper -e True
python3.12 run_framework.py --f numpy -p paper -e True
python3.12 run_framework.py --f pythran -p paper -e True