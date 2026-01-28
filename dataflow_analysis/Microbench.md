For linpack/stream you need my npbench for on branch 'collect-papi-metrics'. 

Then its these 2 files:
Linpack:  https://github.com/alexanderfluck/npbench/blob/collect-papi-metrics/npbench/hardware_info/practical/flops_with_linpack.py

STREAM: https://github.com/alexanderfluck/npbench/blob/collect-papi-metrics/npbench/hardware_info/practical/memory_with_stream.py 

Both should compile and directly run the benchmark.

Linpack needs the mpirun command and a blas library (openblas or mkl) I think. And it might run for a while so don't set time limit too low
Stream should be quick and not need anything extra apart from openmp.

If anything doesn't work or crashes let me know