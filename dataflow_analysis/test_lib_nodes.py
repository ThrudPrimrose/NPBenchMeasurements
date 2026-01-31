import argparse
import traceback
import subprocess
import re
import copy
import os
import importlib
import multiprocessing as mp
import pandas as pd
from datetime import datetime, timezone
from collections import defaultdict
from math import sqrt
from statistics import median
from dace.sdfg.performance_evaluation.helpers import get_uuid
from dace.sdfg.performance_evaluation.work_depth import get_tasklet_work
import dace.transformation.auto.auto_optimize as opt
from dace.transformation.dataflow.wcr_conversion import AugAssignToWCR, WCRToAugAssign

import dace
from dace.config import Config
from dace.codegen.instrumentation import papi
from dace.sdfg import infer_types
# branch alexanderfluck/combined

from npbench.infrastructure import (Benchmark, utilities as util, DaceFramework)

######################################## Helper Functions #######################################################

def get_bench_sdfg(bench:Benchmark, dace_framework:DaceFramework):
        module_pypath = "npbench.benchmarks.{r}.{m}".format(r=bench.info["relative_path"].replace('/', '.'),
                                                            m=bench.info["module_name"])
        if "postfix" in dace_framework.info.keys():
            postfix = dace_framework.info["postfix"]
        else:
            postfix = dace_framework.fname
        module_str = "{m}_{p}".format(m=module_pypath, p=postfix)
        func_str = bench.info["func_name"]

        ldict = dict()
        # Import DaCe implementation
        try:
            module = importlib.import_module(module_str)
            ct_impl = getattr(module, func_str)

        except Exception as e:
            print("Failed to load the DaCe implementation.")
            raise (e)

        ##### Experimental: Load strict SDFG
        sdfg_loaded = False
        if dace_framework.load_strict:
            path = os.path.join(os.getcwd(), 'dace_sdfgs', f"{module_str}-{func_str}.sdfg")
            try:
                strict_sdfg = dace.SDFG.from_file(path)
                sdfg_loaded = True
            except Exception:
                pass

        if not sdfg_loaded:
            #########################################################
            # Prepare SDFGs
            base_sdfg, _ = util.benchmark("__npb_result = ct_impl.to_sdfg(simplify=False)",
                                                   out_text="DaCe parsing time",
                                                   context=locals(),
                                                   output='__npb_result',
                                                   verbose=False)
            strict_sdfg = copy.deepcopy(base_sdfg)
            strict_sdfg._name = "strict"
            ldict['strict_sdfg'] = strict_sdfg
            simplified_sdfg, _ = util.benchmark("strict_sdfg.simplify()",
                                            out_text="DaCe Strict Transformations time",
                                            context=locals(),
                                            verbose=False)
            # sdfg_list = [strict_sdfg]
            # time_list = [parse_time[0] + strict_time[0]]
        else:
            ldict['strict_sdfg'] = strict_sdfg

        ##### Experimental: Saving strict SDFG
        if dace_framework.save_strict and not sdfg_loaded:
            path = os.path.join(os.getcwd(), 'dace_sdfgs')
            try:
                os.mkdir(path)
            except FileExistsError:
                pass
            path = os.path.join(os.getcwd(), 'dace_sdfgs', f"{module_str}-{func_str}.sdfg")
            strict_sdfg.save(path)

        return base_sdfg, simplified_sdfg

######################################## Helper Functions end ###################################################

if __name__ == "__main__":
    data_rows = []
    parser = argparse.ArgumentParser()
    #parser.add_argument("-p",
    #                    "--preset",
    #                    choices=['S', 'M', 'L', 'paper'],
    #                    nargs="?",
    #                    default='S')
    parser.add_argument("-v",
                        "--validate",
                        type=util.str2bool,
                        nargs="?",
                        default=True)
    parser.add_argument("-e",
                        "--build_event_sets",
                        type=util.str2bool,
                        nargs="?",
                        default=True)
    parser.add_argument("-r", "--repeat", type=int, nargs="?", default=10)
    parser.add_argument("-b", "--benchmarks", type=str, nargs="+", default=None)

    args = vars(parser.parse_args())

    benchmark_set = ['adi','arc_distance','atax','azimint_hist','azimint_naive','bicg',
                  'cavity_flow','channel_flow','cholesky2','cholesky','compute','contour_integral',
                  'conv2d_bias','correlation',#'covariance2',
                  'covariance','crc16','deriche',#'doitgen',
                  'durbin','fdtd_2d','floyd_warshall','gemm',
                  'gemver','gesummv','go_fast','gramschmidt',
                  'hdiff','heat_3d','jacobi_1d','jacobi_2d','k2mm','k3mm','lenet','ludcmp','lu',#'mandelbrot1',
                  #'mandelbrot2',
                  'mlp','mvt','nbody','nussinov','resnet','scattering_self_energies','seidel_2d',
                  'softmax','spmv',#'stockham_fft',
                  'symm','syr2k','syrk','trisolv','trmm','vadv']
    
    benchmarks = list()
    if args["benchmarks"]:
        requested_bms = []
        for benchmark_name in args["benchmarks"]:
            if benchmark_name in benchmark_set:
                benchmarks.append(benchmark_name)
            else:
                print(f"Could not find a benchmark with the name {benchmark_name}")
    else:
        benchmarks = list(benchmark_set)
    

    dace_cpu_framework = DaceFramework("dace_cpu")
    repetitions = args["repeat"]

    for preset in ["L", "paper"]:
        start = (int(datetime.now(timezone.utc).timestamp() * 1000))
        ba_fail=[]

        timeout_fail = []
        error_fail = []
        substitute = True
        
        sdfgs_without_lib_nodes = []
        sdfgs_with_lib_nodes = []

        for benchmark_name in benchmarks:
            print("="*50, benchmark_name, "(", preset, ")", "="*50)
            benchmark = Benchmark(benchmark_name)
            sdfg, simplified_sdfg = get_bench_sdfg(benchmark, dace_cpu_framework)
            base_sdfg = copy.deepcopy(sdfg)
            base_sdfg.save("base_sdfg.sdfg")

            print("Base SDFG Lib nodes")

            mma_node_count = 0
            for n, g in base_sdfg.all_nodes_recursive():
                if isinstance(n, dace.nodes.LibraryNode):
                    print(n, n.label)
                    if "MatMul" in n.label or "Dot" in n.label or "solve" in n.label:
                        mma_node_count += 1

            print(f"Base SDFG has {mma_node_count} library nodes")

            if mma_node_count == 0:
                sdfgs_without_lib_nodes.append(benchmark_name)
            else:
                sdfgs_with_lib_nodes.append(benchmark_name)

            opt.auto_optimize(sdfg, dace.dtypes.DeviceType.CPU)
            substitutions = benchmark.info["parameters"][preset]
            sdfg.save("curr_sdfg.sdfg")
            infer_types.set_default_schedule_and_storage_types(sdfg)
            for n, g in sdfg.all_nodes_recursive():
                if isinstance(n, dace.nodes.LibraryNode):
                    print(n)

            if "azimint" in benchmark_name:
                raise Exception("A")


        print("SDFGs with lib nodes:", sdfgs_with_lib_nodes)
        print("SDFGs without lib nodes:", sdfgs_without_lib_nodes)

    end = (int(datetime.now(timezone.utc).timestamp() * 1000))
    print("Duration:",  (end - start)/(1000*60), "min")
