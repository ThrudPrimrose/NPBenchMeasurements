#!/usr/bin/env python3
import argparse
import pathlib
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import LogLocator, NullFormatter

CACHE_LINE = 64  # bytes
WRITE_ALLOCATE = 1
WORD_SIZE = 4  # word=2, dword=4, qword=8, assume most loads to be 4?

# Channel flow not analyzable due to while loop
# Azimint timedout when collecting papi metrics
# Softmax is also a pure library call
sdfgs_with_lib_nodes = ['azimint_hist', 'azimint_naive',  'channel_flow', 'softmax', 'atax', 'bicg', 'cholesky', 'contour_integral', 'correlation', 'covariance', 'durbin', 'gemm', 'gemver', 'gesummv', 'gramschmidt', 'k2mm', 'k3mm', 'lenet', 'ludcmp', 'lu', 'mlp', 'mvt', 'nbody', 'scattering_self_energies', 'spmv', 'symm', 'trisolv', 'trmm']
sdfgs_with_no_lib_nodes = ['adi', 'arc_distance', 'cavity_flow', 'cholesky2', 'compute', 'conv2d_bias', 'crc16', 'deriche', 'fdtd_2d', 'floyd_warshall', 'go_fast', 'hdiff', 'heat_3d', 'jacobi_1d', 'jacobi_2d', 'nussinov', 'resnet', 'seidel_2d',  'syr2k', 'syrk', 'vadv']

def read_sqlite_db(db_path: pathlib.Path):
    print(f"\n=== Database: {db_path} ===")
    conn = sqlite3.connect(db_path)
    dfs = dict()
    try:
        tables = pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table';", conn
        )["name"].tolist()
        if not tables:
            print("  (No tables found)")
            return
        for table in tables:
            df = pd.read_sql(f"SELECT * FROM {table}", conn)
            dfs[table] = df
    finally:
        conn.close()
    return dfs


def compute_traffic_for_level(row, time_sec, level="L3"):
    """
    Compute traffic estimates for a given cache level (L2 or L3).
    
    Args:
        row: DataFrame row containing PAPI counters
        time_sec: Execution time in seconds
        level: "L2" or "L3"
    
    Returns:
        Dictionary with traffic estimates
    """
    # Get counters based on level
    if level == "L3":
        tcm = row.get("PAPI_L3_TCM", 0.0)
        dcr = row.get("PAPI_L3_DCR", 0.0)
        dcw = row.get("PAPI_L3_DCW", 0.0)
        tca = row.get("PAPI_L3_TCA", 0.0)
    else:  # L2
        tcm = row.get("PAPI_L2_TCM", 0.0)
        dcr = row.get("PAPI_L2_DCR", 0.0)
        dcw = row.get("PAPI_L2_DCW", 0.0)
        tca = row.get("PAPI_L2_TCA", 0.0)
    
    lst = row.get("PAPI_LST_INS", 0.0)
    
    # Data movement estimates
    bytes_min = tcm * CACHE_LINE
    bytes_rw = (dcr + dcw) * CACHE_LINE
    bytes_upper = tca * CACHE_LINE
    bytes_wb = tcm * CACHE_LINE
    bytes_lst = lst * WORD_SIZE

    return {
        f"{level} Misses": bytes_min,
        f"{level} DRead + DWrite": bytes_rw,
        f"{level} DMisses": bytes_wb,
        f"{level} Access": bytes_upper,
        "LST": bytes_lst,
    }


def compute_all_traffic(row, time_sec):
    """Compute traffic for both L2 and L3 levels."""
    flops = row.get("PAPI_FP_OPS", 0.0) + row.get("PAPI_DP_OPS", 0.0)
    
    # Get L2 and L3 traffic
    l2_traffic = compute_traffic_for_level(row, time_sec, level="L2")
    l3_traffic = compute_traffic_for_level(row, time_sec, level="L3")
    
    # Merge dictionaries
    result = {**l2_traffic, **l3_traffic}
    result["PAPI FLOPs"] = flops
    
    return pd.Series(result)


def plot_traffic_by_level_violin(df_raw: pd.DataFrame, df_agg: pd.DataFrame, level="L3", output_file="traffic_comparison_violin.pdf", filter_benchmarks=None):
    """
    Plot traffic comparison using violin plots for a specific cache level.
    
    Args:
        df_raw: Raw DataFrame with individual measurements
        df_agg: Aggregated DataFrame with computed traffic values
        level: "L2" or "L3"
        output_file: Output filename
        filter_benchmarks: List of benchmarks to include, or None for all
    """
    # Get time per benchmark
    time_per_bench = df_raw.groupby("benchmark")["time"].mean()
    
    # Compute traffic for each individual measurement
    all_data = []
    
    for benchmark in df_raw['benchmark'].unique():
        # Skip if not in filter list
        if filter_benchmarks is not None and benchmark not in filter_benchmarks:
            continue
            
        bench_data = df_raw[df_raw['benchmark'] == benchmark]
        
        # Pivot each repetition
        for rep in bench_data['repetition'].unique():
            rep_data = bench_data[bench_data['repetition'] == rep]
            pivot_rep = rep_data.pivot_table(
                index="benchmark",
                columns="event",
                values="total_count",
                aggfunc="first"
            ).fillna(0.0)
            
            if len(pivot_rep) > 0:
                traffic = compute_traffic_for_level(
                    pivot_rep.iloc[0], 
                    time_per_bench[benchmark], 
                    level=level
                )
                
                for metric_name, value in traffic.items():
                    all_data.append({
                        'benchmark': benchmark,
                        'metric': metric_name,
                        'value': value,
                        'repetition': rep
                    })
    
    # Add symbolic bytes if available
    if "Symbolic Bytes" in df_agg.columns:
        for _, row in df_agg.iterrows():
            # Skip if not in filter list
            if filter_benchmarks is not None and row['benchmark'] not in filter_benchmarks:
                continue
                
            if pd.notna(row["Symbolic Bytes"]):
                # Add 10 copies of symbolic bytes (one per repetition) for fair comparison
                for rep in range(1, 11):
                    all_data.append({
                        'benchmark': row['benchmark'],
                        'metric': 'Symbolic Bytes',
                        'value': row["Symbolic Bytes"],
                        'repetition': rep
                    })
    
    df_violin = pd.DataFrame(all_data)
    
    if len(df_violin) == 0:
        print(f"Warning: No data for {output_file}")
        return
    
    # Save violin data
    csv_name = output_file.replace('.pdf', '.csv').replace('_violin', '_violin_data')
    df_violin.to_csv(csv_name, index=False)
    print(f"✓ Saved: {csv_name}")
    
    # Plot setup
    sns.set(style="whitegrid")
    benchmarks = sorted(df_violin["benchmark"].unique())
    n_cols = 4
    n_rows = (len(benchmarks) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5*n_cols, 3.5*n_rows), squeeze=False)
    
    for i, benchmark in enumerate(benchmarks):
        ax = axes[i // n_cols, i % n_cols]
        data = df_violin[df_violin["benchmark"] == benchmark]
        
        # Violin plot
        sns.violinplot(
            data=data,
            x="metric",
            y="value",
            hue="metric",
            ax=ax,
            inner="box",
            cut=0,
            legend=False
        )
        
        ax.set_title(f"{benchmark}", fontsize=14, fontweight='bold')
        ax.set_xlabel("", fontsize=11)
        ax.set_ylabel("Bytes", fontsize=11)
        ax.tick_params(axis='x', rotation=90, labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.set_ylim(bottom=0)
        
        # Add grid
        ax.grid(True, axis='y', alpha=0.3)
    
    # Hide empty subplots
    for j in range(i+1, n_rows*n_cols):
        fig.delaxes(axes[j // n_cols, j % n_cols])
    
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    plt.savefig(output_file.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_file}")


def plot_traffic_by_level_median(df_raw: pd.DataFrame, df_agg: pd.DataFrame, level="L3", output_file="traffic_comparison_median.pdf", filter_benchmarks=None):
    """
    Plot traffic comparison using median values with dot plots for a specific cache level.
    
    Args:
        df_raw: Raw DataFrame with individual measurements
        df_agg: Aggregated DataFrame with computed traffic values
        level: "L2" or "L3"
        output_file: Output filename
        filter_benchmarks: List of benchmarks to include, or None for all
    """
    # Get time per benchmark
    time_per_bench = df_raw.groupby("benchmark")["time"].median()
    
    # Compute traffic for each individual measurement, then take median
    all_data = []
    
    for benchmark in df_raw['benchmark'].unique():
        # Skip if not in filter list
        if filter_benchmarks is not None and benchmark not in filter_benchmarks:
            continue
            
        bench_data = df_raw[df_raw['benchmark'] == benchmark]
        
        # Collect all repetitions
        bench_traffic = []
        for rep in bench_data['repetition'].unique():
            rep_data = bench_data[bench_data['repetition'] == rep]
            pivot_rep = rep_data.pivot_table(
                index="benchmark",
                columns="event",
                values="total_count",
                aggfunc="first"
            ).fillna(0.0)
            
            if len(pivot_rep) > 0:
                traffic = compute_traffic_for_level(
                    pivot_rep.iloc[0], 
                    time_per_bench[benchmark], 
                    level=level
                )
                bench_traffic.append(traffic)
        
        # Compute median for each metric
        if bench_traffic:
            metrics = bench_traffic[0].keys()
            for metric_name in metrics:
                values = [t[metric_name] for t in bench_traffic]
                median_val = np.median(values)
                all_data.append({
                    'benchmark': benchmark,
                    'metric': metric_name,
                    'value': median_val
                })
    
    # Add symbolic bytes if available
    if "Symbolic Bytes" in df_agg.columns:
        for _, row in df_agg.iterrows():
            # Skip if not in filter list
            if filter_benchmarks is not None and row['benchmark'] not in filter_benchmarks:
                continue
                
            if pd.notna(row["Symbolic Bytes"]):
                all_data.append({
                    'benchmark': row['benchmark'],
                    'metric': 'Symbolic Bytes',
                    'value': row["Symbolic Bytes"]
                })
    
    df_median = pd.DataFrame(all_data)
    
    if len(df_median) == 0:
        print(f"Warning: No data for {output_file}")
        return
    
    # Save median data
    csv_name = output_file.replace('.pdf', '.csv').replace('_median', '_median_data')
    df_median.to_csv(csv_name, index=False)
    print(f"✓ Saved: {csv_name}")
    
    # Plot setup
    sns.set(style="whitegrid")
    benchmarks = sorted(df_median["benchmark"].unique())
    n_cols = 4
    n_rows = (len(benchmarks) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5*n_cols, 3.5*n_rows), squeeze=False)
    
    for i, benchmark in enumerate(benchmarks):
        ax = axes[i // n_cols, i % n_cols]
        data = df_median[df_median["benchmark"] == benchmark]
        
        # Dot plot (scatter plot)
        sns.scatterplot(
            data=data,
            x="metric",
            y="value",
            hue="metric",
            s=50,
            ax=ax,
            legend=False,
            edgecolor='black',
            linewidth=1.5
        )
        
        ax.set_title(f"{benchmark}", fontsize=14, fontweight='bold')
        ax.set_xlabel("", fontsize=11)
        ax.set_ylabel("Bytes (median)", fontsize=11)
        ax.tick_params(axis='x', rotation=90, labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.set_ylim(bottom=0)
        
        # Add grid
        ax.grid(True, axis='y', alpha=0.3)
    
    # Hide empty subplots
    for j in range(i+1, n_rows*n_cols):
        fig.delaxes(axes[j // n_cols, j % n_cols])
    
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    plt.savefig(output_file.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_file}")


def plot_traffic_by_level(df: pd.DataFrame, level="L3", output_file="traffic_comparison.pdf"):
    """
    Plot traffic comparison for a specific cache level (original scatter version).
    
    Args:
        df: DataFrame containing results
        level: "L2" or "L3"
        output_file: Output filename
    """
    # Traffic columns for this level
    traffic_columns = [
        f'{level} Misses',
        f'{level} DRead + DWrite',
        f'{level} DMisses',
        f'{level} Access',
        'LST',
    ]
    
    # Add symbolic bytes for L3 only
    if "Symbolic Bytes" in df.columns:
        traffic_columns.append("Symbolic Bytes")
    
    # Melt the dataframe for plotting
    df_melt = df.melt(
        id_vars=["benchmark"],
        value_vars=traffic_columns,
        var_name="Variant",
        value_name="Bytes_Moved"
    )
    
    # Plot setup
    sns.set(style="whitegrid")
    benchmarks = df["benchmark"].unique()
    n_cols = 3
    n_rows = (len(benchmarks) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), squeeze=False)
    
    for i, benchmark in enumerate(benchmarks):
        ax = axes[i // n_cols, i % n_cols]
        data = df_melt[df_melt["benchmark"] == benchmark]
        
        # Scatter plot: x = variant, y = bytes moved
        sns.scatterplot(
            data=data,
            x="Variant",
            y="Bytes_Moved",
            hue="Variant",
            style="Variant",
            s=100,
            ax=ax,
            legend=False
        )
        ax.set_title(f"Benchmark: {benchmark}")
        ax.set_xlabel("Variant / Formula")
        ax.set_ylabel("Bytes Moved")
        ax.tick_params(axis='x', rotation=90)
        ax.set_ylim(bottom=0)
    
    # Hide empty subplots
    for j in range(i+1, n_rows*n_cols):
        fig.delaxes(axes[j // n_cols, j % n_cols])
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"✓ Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Read SQLite database files into pandas and create violin plots"
    )
    parser.add_argument(
        "--path",
        type=pathlib.Path,
        default=pathlib.Path("../npbench_papi_intel_xeon_6154/L/npbench_papi_metrics.db"),
        help="Path to a SQLite file or a directory containing SQLite files",
    )
    parser.add_argument(
        "--pattern",
        default="*.db",
        help="Filename pattern to match (default: *.db)",
    )
    parser.add_argument(
        "--preset",
        default="L",
        help="Preset to filter benchmarks (default: L)",
    )
    args = parser.parse_args()

    if args.path.is_file():
        dfs = read_sqlite_db(args.path)
    elif args.path.is_dir():
        raise Exception("Directory input not supported in this version.")

    df = dfs["event_counts"]
    
    # Filter out channel_flow benchmark
    print(f"\nTotal rows before filtering: {len(df)}")
    df = df[df['benchmark'] != 'channel_flow']
    print(f"Total rows after filtering channel_flow: {len(df)}")
    print(f"Remaining benchmarks: {sorted(df['benchmark'].unique())}\n")

    # Add repetition number per benchmark-event combination
    df['repetition'] = df.groupby(['benchmark', 'event']).cumcount() + 1
    
    # Save raw data with repetitions for violin plots
    df_violin = df[['benchmark', 'preset', 'event', 'total_count', 'time', 'repetition']].copy()
    df_violin.to_csv("violin_plot_raw_data.csv", index=False)
    print("✓ Saved: violin_plot_raw_data.csv")
    
    # Pivot table so we can access metrics by event per benchmark (using mean)
    pivot = df.pivot_table(
        index="benchmark",
        columns="event",
        values="total_count",
        aggfunc="mean"
    ).fillna(0.0)

    # Time per benchmark (average)
    time_per_bench = df.groupby("benchmark")["time"].mean()

    # Apply traffic computation per benchmark
    results = pivot.apply(
        lambda row: compute_all_traffic(row, time_per_bench[row.name]),
        axis=1
    )

    # Merge with pivot for inspection
    results = results.reset_index()

    # Load symbolic analysis
    try:
        symbolic_analysis = pd.read_csv("volumes_per_preset.csv")
        preset = args.preset

        # Filter symbolic_analysis for the current preset
        symbolic_filtered = symbolic_analysis[symbolic_analysis["preset"] == preset]

        # Merge with results on benchmark/kernel
        results = results.merge(
            symbolic_filtered,
            left_on="benchmark",
            right_on="kernel",
            how="left"
        )

        # Drop Na (at least one analysis failed)
        results = results.dropna(subset=["preset"])

        # Sum up read and write for symbolic bytes
        results["Symbolic Bytes"] = (
            results["symbolic_volume_read_bytes"] + results["symbolic_volume_write_bytes"]
        )
        
        print(f"\n✓ Loaded symbolic analysis with {len(results)} benchmarks")
    except FileNotFoundError:
        print("\nWarning: volumes_per_preset.csv not found, skipping symbolic analysis")

    print(results)

    # Save the dataframe to CSV
    results.to_csv("results.csv", index=False)
    print("✓ Saved: results.csv")

    # Save to SQLite database
    conn = sqlite3.connect("results.db")
    results.to_sql("results", conn, if_exists="replace", index=False)
    conn.close()
    print("✓ Saved: results.db")

    print("\n" + "="*60)
    print("GENERATING VIOLIN PLOTS (ALL BENCHMARKS)")
    print("="*60)
    
    # Generate violin plots using raw data - all benchmarks
    plot_traffic_by_level_violin(df, results, level="L2", 
                                  output_file="traffic_comparison_L2_violin_all.pdf")
    plot_traffic_by_level_violin(df, results, level="L3", 
                                  output_file="traffic_comparison_L3_violin_all.pdf")

    print("\n" + "="*60)
    print("GENERATING MEDIAN DOT PLOTS (ALL BENCHMARKS)")
    print("="*60)
    
    # Generate median dot plots - all benchmarks
    plot_traffic_by_level_median(df, results, level="L2", 
                                  output_file="traffic_comparison_L2_median_all.pdf")
    plot_traffic_by_level_median(df, results, level="L3", 
                                  output_file="traffic_comparison_L3_median_all.pdf")

    print("\n" + "="*60)
    print("GENERATING VIOLIN PLOTS (NO LIB NODES ONLY)")
    print("="*60)
    
    # Filter for no lib nodes (excluding channel_flow which is already removed)
    no_lib_filter = [b for b in sdfgs_with_no_lib_nodes if b != 'channel_flow']
    
    # Generate violin plots - no lib nodes only
    plot_traffic_by_level_violin(df, results, level="L2", 
                                  output_file="traffic_comparison_L2_violin_nolib.pdf",
                                  filter_benchmarks=no_lib_filter)
    plot_traffic_by_level_violin(df, results, level="L3", 
                                  output_file="traffic_comparison_L3_violin_nolib.pdf",
                                  filter_benchmarks=no_lib_filter)

    print("\n" + "="*60)
    print("GENERATING MEDIAN DOT PLOTS (NO LIB NODES ONLY)")
    print("="*60)
    
    # Generate median dot plots - no lib nodes only
    plot_traffic_by_level_median(df, results, level="L2", 
                                  output_file="traffic_comparison_L2_median_nolib.pdf",
                                  filter_benchmarks=no_lib_filter)
    plot_traffic_by_level_median(df, results, level="L3", 
                                  output_file="traffic_comparison_L3_median_nolib.pdf",
                                  filter_benchmarks=no_lib_filter)

    print("\n" + "="*60)
    print("GENERATING ORIGINAL SCATTER PLOTS (for comparison)")
    print("="*60)
    
    # Generate original scatter plots for comparison
    plot_traffic_by_level(results, level="L2", output_file="traffic_comparison_L2_scatter.pdf")
    plot_traffic_by_level(results, level="L3", output_file="traffic_comparison_L3_scatter.pdf")

    print("\n" + "="*60)
    print("ALL PLOTS GENERATED SUCCESSFULLY!")
    print("="*60)
    print("\nGenerated files:")
    print("  Violin plots (all): *_violin_all.pdf/png")
    print("  Median plots (all): *_median_all.pdf/png")
    print("  Violin plots (no lib): *_violin_nolib.pdf/png")
    print("  Median plots (no lib): *_median_nolib.pdf/png")
    print("  Scatter plots: *_scatter.pdf/png")
    print("  Data CSVs: *_data.csv")
    print("="*60)


if __name__ == "__main__":
    main()