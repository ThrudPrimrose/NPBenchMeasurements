#!/bin/bash

BENCHMARK_SET=(
  adi arc_distance atax azimint_hist azimint_naive bicg
  cavity_flow channel_flow cholesky2 cholesky compute contour_integral
  conv2d_bias correlation covariance crc16 deriche
  durbin fdtd_2d floyd_warshall gemm gemver gesummv go_fast gramschmidt
  hdiff heat_3d jacobi_1d jacobi_2d k2mm k3mm lenet ludcmp lu
  mandelbrot1 mandelbrot2 mlp mvt nbody nussinov resnet
  scattering_self_energies seidel_2d softmax spmv
  symm syr2k syrk trisolv trmm vadv
)

mkdir -p logs

for KERNEL in "${BENCHMARK_SET[@]}"; do
    echo "Submitting kernel: ${KERNEL}"
    sbatch run_kernel.sh ${KERNEL}
done
