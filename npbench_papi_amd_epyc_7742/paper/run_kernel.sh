#!/bin/bash
#SBATCH --job-name=apapi
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=03:00:00
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

# -------------------------------
# Kernel argument
# -------------------------------
if [ "$#" -ne 1 ]; then
    echo "Usage: sbatch run_kernel.sbatch <kernel>"
    exit 1
fi

KERNEL=$1

echo "====================================="
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Kernel:      ${KERNEL}"
echo "OMP threads: ${OMP_NUM_THREADS}"
echo "Node(s):     ${SLURM_NODELIST}"
echo "====================================="

# -------------------------------
# Run benchmark
# -------------------------------
cp ../collect_roofline_metrics.py .
BENCHMARK_BIN=collect_roofline_metrics.py
OUTDIR=results/${KERNEL}
mkdir -p ${OUTDIR}
chmod +x collect_roofline_metrics.py
python ${BENCHMARK_BIN} -p paper --benchmarks "${KERNEL}" --repeat=10 --build_event_sets=false
