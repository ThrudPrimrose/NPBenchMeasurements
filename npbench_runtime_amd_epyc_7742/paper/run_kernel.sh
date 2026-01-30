#!/bin/bash
#SBATCH --job-name=aPaper
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
spack load openblas@0.3.29%gcc@14.2
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

export OPENBLAS_DIR=$(spack location -i openblas@0.3.29%gcc@14.2)
export C_INCLUDE_PATH=${OPENBLAS_DIR}/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=${OPENBLAS_DIR}/include:$CPLUS_INCLUDE_PATH
export CPATH=${OPENBLAS_DIR}/include:$CPATH
export LD_LIBRARY_PATH=${OPENBLAS_DIR}/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=${OPENBLAS_DIR}/lib:$LIBRARY_PATH
export LDFLAGS="-L${OPENBLAS_DIR}/lib $LDFLAGS"
export CXXFLAGS="-I${OPENBLAS_DIR}/include $CXXFLAGS"
export CPPFLAGS="-I${OPENBLAS_DIR}/include $CPPFLAGS"
export CFLAGS="-I${OPENBLAS_DIR}/include $CFLAGS"


echo "====================================="
echo "Job ID:      ${SLURM_JOB_ID}"
echo "OMP threads: ${OMP_NUM_THREADS}"
echo "Node(s):     ${SLURM_NODELIST}"
echo "====================================="

# -------------------------------
# Run benchmark
# -------------------------------
cd npbench
echo DACE_CPU
python3.12 run_framework.py --f dace_cpu -p paper -e True -t 1800
echo NUMPY
python3.12 run_framework.py --f numpy -p paper -e True -t 1800
echo JAX
python3.12 run_framework.py --f jax -p paper -e True -t 1800
echo NUMBA
python3.12 run_framework.py --f numba -p paper -e True -t 1800
