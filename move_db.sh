cd npbench_runtime_amd_epyc_7742/L/npbench
cp npbench.db ..
python plot_results.py -p L

cd ../../..
cd npbench_runtime_amd_epyc_7742/paper/npbench
cp npbench.db ..
python plot_results.py -p paper

cd ../../..
cd npbench_runtime_intel_xeon_6154/L/npbench
cp npbench.db ..
python plot_results.py -p L

cd ../../..
cd npbench_runtime_intel_xeon_6154/paper/npbench
cp npbench.db ..
python plot_results.py -p paper

cd ../../..

git add npbench_runtime_amd_epyc_7742/L/npbench.db
git add npbench_runtime_amd_epyc_7742/paper/npbench.db
git add npbench_runtime_intel_xeon_6154/L/npbench.db
git add npbench_runtime_intel_xeon_6154/paper/npbench.db