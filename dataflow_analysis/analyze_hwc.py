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


def plot_traffic_by_level(df: pd.DataFrame, level="L3", output_file="traffic_comparison.pdf"):
    """
    Plot traffic comparison for a specific cache level.
    
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
        "Symbolic Bytes"
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


def plot_flops(df: pd.DataFrame):
    """Plot PAPI FLOPs vs Symbolic FLOPs with benchmarks on y-axis"""
    # Filter out benchmarks where both values are 0
    df_filtered = df[(df['PAPI FLOPs'] > 0) | (df['work'] > 0)].copy()
    
    # Sort benchmarks alphabetically
    df_filtered = df_filtered.sort_values('benchmark', ascending=False)
    
    # Create figure with extra space for legend
    fig, ax = plt.subplots(figsize=(12, max(10, len(df_filtered) * 0.5)))
    
    benchmarks = df_filtered['benchmark'].values
    papi_flops = df_filtered['PAPI FLOPs'].values
    symbolic_flops = df_filtered['work'].values
    y_positions = np.arange(len(benchmarks))
    
    # Add diagonal lines connecting the two points for each benchmark
    for i, (papi, symbolic) in enumerate(zip(papi_flops, symbolic_flops)):
        ax.plot([papi, symbolic], [i, i], 'k-', alpha=0.2, linewidth=1.5)
    
    # Plot symbolic FLOPs first (so PAPI appears on top)
    ax.scatter(symbolic_flops, y_positions, label='Symbolic FLOPs (work)',
               s=150, alpha=0.7, marker='s', color='#ff7f0e',
               edgecolors='black', linewidth=1)
    
    # Plot PAPI FLOPs on top with smaller marker
    ax.scatter(papi_flops, y_positions, label='PAPI FLOPs',
               s=120, alpha=0.8, marker='o', color='#1f77b4',
               edgecolors='black', linewidth=1)
    
    # Set y-axis
    ax.set_yticks(y_positions)
    ax.set_yticklabels(benchmarks, fontsize=18)
    ax.set_ylabel('Benchmark', fontsize=20)
    
    # X axis settings
    ax.set_xlabel('FLOPs', fontsize=20)
    ax.set_xscale('log')
    ax.tick_params(axis='x', labelsize=18)
    
    # Major ticks at 10^n
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=20))
    # Minor ticks at 2..9 * 10^n
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=(2,3,4,5,6,7,8,9), numticks=100))
    ax.xaxis.set_minor_formatter(NullFormatter())
    
    # Grid lines
    ax.grid(True, which='major', axis='x', linestyle='-', linewidth=1, alpha=0.3, color='gray')
    ax.grid(True, which='minor', axis='x', linestyle=':', linewidth=0.5, alpha=0.15, color='gray')
    
    # Limits
    ax.set_xlim(left=max(1, min(papi_flops.min(), symbolic_flops.min()) * 0.5))
    
    # Add legend outside plot area
    ax.legend(fontsize=16, loc='upper center', bbox_to_anchor=(0.5, -0.05),
              ncol=2, framealpha=0.95, edgecolor='black', fancybox=True, shadow=True)
    
    # Add title
    ax.set_title('FLOPs Obtained Through Hardware Counters vs\n'
                 'Symbolic Analysis', fontsize=22, pad=20)
    
    plt.tight_layout()
    plt.savefig("flops_comparison.pdf", bbox_inches='tight', dpi=900)
    plt.savefig("flops_comparison.png", bbox_inches='tight', dpi=900)
    plt.close()
    print("✓ Saved: flops_comparison.pdf and flops_comparison.png")


def main():
    parser = argparse.ArgumentParser(
        description="Read SQLite database files into pandas and print them"
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

    df = dfs["event_averages"]

    # Pivot table so we can access metrics by event_name per benchmark
    pivot = df.pivot_table(
        index="benchmark",
        columns="event_name",
        values="average",
        aggfunc="first"
    ).fillna(0.0)

    # Time per benchmark
    time_per_bench = df.groupby("benchmark")["time"].first()

    # Apply traffic computation per benchmark
    results = pivot.apply(
        lambda row: compute_all_traffic(row, time_per_bench[row.name]),
        axis=1
    )

    # Merge with pivot for inspection
    results = results.reset_index()

    # Load symbolic analysis
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

    print(results)

    # Save the dataframe to CSV
    results.to_csv("results.csv", index=False)

    # Save to SQLite database
    conn = sqlite3.connect("results.db")
    results.to_sql("results", conn, if_exists="replace", index=False)
    conn.close()

    # Generate plots
    plot_traffic_by_level(results, level="L2", output_file="traffic_comparison_L2.pdf")
    plot_traffic_by_level(results, level="L3", output_file="traffic_comparison_L3.pdf")
    plot_flops(results)


if __name__ == "__main__":
    main()