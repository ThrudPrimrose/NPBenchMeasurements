#!/usr/bin/env python3

import argparse
import pathlib
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


CACHE_LINE = 64  # bytes
WRITE_ALLOCATE = 1
WORD_SIZE = 4 #word=2, dword=4, qword=8, assume most loads to be 4? 

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

    # =========================
    # PAPI METRICS – SEMANTICS
    # =========================

    # PAPI_DP_OPS
    #   Double-precision floating-point operations retired.
    #   Used as FLOPs in roofline models.
    #
    #   FLOPs = PAPI_DP_OPS


    # -------------------------
    # Load / Store Instructions
    # -------------------------

    # PAPI_LD_INS
    #   Number of retired load instructions.
    #   Counts instructions, NOT bytes.
    #   Not suitable alone for data movement estimation.

    # PAPI_SR_INS
    #   Number of retired store instructions.
    #   Store instructions may trigger write-allocate traffic.

    # PAPI_LST_INS
    #   Total load + store instructions.


    # -------------------------
    # L1 Cache Misses
    # -------------------------

    # PAPI_L1_DCM
    #   L1 data cache misses.
    #   Each miss implies a cache-line transfer from L2.
    #
    #   Bytes_L2 ≈ PAPI_L1_DCM * CACHE_LINE_SIZE

    # PAPI_L1_ICM
    #   L1 instruction cache misses (ignore for data movement).

    # PAPI_L1_TCM
    #   Total L1 cache misses (data + instruction).

    # PAPI_L1_LDM
    #   L1 load misses.

    # PAPI_L1_STM
    #   L1 store misses (affected by write-allocate policy).


    # -------------------------
    # L2 Cache Misses
    # -------------------------

    # PAPI_L2_DCM
    #   L2 data cache misses.
    #   Each miss implies a cache-line transfer from L3.

    # PAPI_L2_ICM
    #   L2 instruction cache misses.

    # PAPI_L2_TCM
    #   Total L2 cache misses.
    #
    #   Bytes_L3 ≈ PAPI_L2_TCM * CACHE_LINE_SIZE

    # PAPI_L2_LDM
    #   L2 load misses.

    # PAPI_L2_STM
    #   L2 store misses.


    # -------------------------
    # L3 Cache Misses (DRAM Traffic)
    # -------------------------

    # PAPI_L3_TCM
    #   Total L3 cache misses.
    #   Each miss fetches one cache line from main memory (DRAM).
    #
    #   Bytes_DRAM ≈ PAPI_L3_TCM * CACHE_LINE_SIZE
    #
    #   OperationalIntensity = PAPI_DP_OPS / Bytes_DRAM

    # PAPI_L3_LDM
    #   L3 load misses.


    # -------------------------
    # Cache Accesses (Hits + Misses)
    # -------------------------

    # PAPI_L2_DCA
    #   L2 data cache accesses (hits + misses).
    #   L2 hit rate:
    #     L2_hit_rate = 1 - (PAPI_L2_DCM / PAPI_L2_DCA)

    # PAPI_L3_DCA
    #   L3 data cache accesses.

    # PAPI_L2_TCA
    #   Total L2 cache accesses.

    # PAPI_L3_TCA
    #   Total L3 cache accesses.


    # -------------------------
    # Cache Reads / Writes
    # -------------------------

    # PAPI_L2_DCR
    #   L2 data cache reads.

    # PAPI_L2_DCW
    #   L2 data cache writes.

    # PAPI_L3_DCR
    #   L3 data cache reads.

    # PAPI_L3_DCW
    #   L3 data cache writes.
    #
    #   Read/Write-aware L3 traffic model:
    #     Bytes_L3 ≈ (PAPI_L3_DCR + WRITE_ALLOCATE * PAPI_L3_DCW) * CACHE_LINE_SIZE
    #
    #   Where WRITE_ALLOCATE = 1 on most x86 CPUs.


    # -------------------------
    # Cache Coherency (Multicore)
    # -------------------------

    # PAPI_CA_SNP
    #   Cache snoop requests.

    # PAPI_CA_SHR
    #   Shared cache-line requests.

    # PAPI_CA_CLN
    #   Clean cache-line requests.

    # PAPI_CA_ITV
    #   Cache invalidations.
    #
    #   Usually ignored for single-thread or socket-local roofline models.


    # =========================
    # COMMON DERIVED METRICS
    # =========================

    # CACHE_LINE_SIZE = 64  # bytes (x86 / AMD EPYC / Intel)

    # Bytes moved at each level:
    #   Bytes_L2   = PAPI_L1_DCM * CACHE_LINE_SIZE
    #   Bytes_L3   = PAPI_L2_TCM * CACHE_LINE_SIZE
    #   Bytes_DRAM = PAPI_L3_TCM * CACHE_LINE_SIZE

    # Roofline-safe upper bound (avoids double counting):
    #   TotalBytes = max(PAPI_L1_DCM, PAPI_L2_TCM, PAPI_L3_TCM) * CACHE_LINE_SIZE

    # Bandwidth:
    #   BW = TotalBytes / execution_time
    # Assuming df is your 'event_averages' table
    # Pivot table so we can access metrics by event_name per benchmark
    pivot = df.pivot_table(
        index="benchmark",
        columns="event_name",
        values="average",
        aggfunc="first"  # in case duplicates exist
    ).fillna(0.0)

    # Time per benchmark (assuming same time per group)
    time_per_bench = df.groupby("benchmark")["time"].first()

    # Function to compute all traffic models per row
    def compute_traffic(row, time_sec):
        l1_dcm = row.get("PAPI_L1_DCM", 0.0)
        l2_tcm = row.get("PAPI_L2_TCM", 0.0)
        l3_tcm = row.get("PAPI_L3_TCM", 0.0)
        l3_ldm = row.get("PAPI_L3_LDM", 0.0)
        l3_stm = row.get("PAPI_L3_STM", 0.0)
        l3_dcr = row.get("PAPI_L3_DCR", 0.0)
        l3_dcw = row.get("PAPI_L3_DCW", 0.0)
        l3_tca = row.get("PAPI_L3_TCA", 0.0)
        dp_ops = row.get("PAPI_DP_OPS", 0.0)
        lst = row.get("PAPI_LST_INS", 0.0)
        flops = row.get("PAPI_FP_OPS", 0.0) + row.get("PAPI_DP_OPS", 0.0)

        # Data movement estimates
        # Formulas to try:
        bytes_dram_min   = l3_tcm * CACHE_LINE
        bytes_dram_rw    = (l3_dcr + l3_dcw) * CACHE_LINE
        bytes_dram_upper = l3_tca * CACHE_LINE
        bytes_hier       = max(l1_dcm, l2_tcm, l3_tcm) * CACHE_LINE
        bytes_dram_wb    = (l3_tcm) * CACHE_LINE
        bytes_dmiss      = (l3_ldm * 2) * CACHE_LINE

        bytes_lst = lst * WORD_SIZE

        def safe_div(a, b):
            return a / b if b > 0 else float("nan")

        # Operational intensity
        oi_min = safe_div(dp_ops, bytes_dram_min)
        oi_rw  = safe_div(dp_ops, bytes_dram_rw)
        oi_wb  = safe_div(dp_ops, bytes_dram_wb)

        # Bandwidth (bytes/sec)
        bw_min = safe_div(bytes_dram_min, time_sec)
        bw_rw  = safe_div(bytes_dram_rw, time_sec)
        bw_wb  = safe_div(bytes_dram_wb, time_sec)

        return pd.Series({
            "L3 Misses": bytes_dram_min,
            "L3 DRead + DWrite": bytes_dram_rw,
            "L3 DMisses": bytes_dram_wb,
            "L3 LST Miss": bytes_dmiss,
            "L3 Access": bytes_dram_upper,
            "Max(L1, L2, L3) Misses": bytes_hier,
            "LST": bytes_lst,
            "PAPI FLOPs": flops,
        })

    # Apply per benchmark
    results = pivot.apply(lambda row: compute_traffic(row, time_per_bench[row.name]), axis=1)

    # Merge with pivot for inspection if needed
    results = results.reset_index()

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

    # Sum up read and write
    results["Symbolic Bytes"] = (
        results["symbolic_volume_read_bytes"] + 
        results["symbolic_volume_write_bytes"]
    )
    print(results)

    # Save the dataframe to CSV
    results.to_csv("results.csv", index=False)
    # Connect (or create) a SQLite database file
    conn = sqlite3.connect("results.db")
    # Save the DataFrame as a table named 'results'
    results.to_sql("results", conn, if_exists="replace", index=False)
    # Close the connection
    conn.close()

    plot_memory(df)
    plot_flops(results)

def plot_memory(df: pd.DataFrame):

    # Read CSV or use existing df
    df = pd.read_csv("results.csv")
    traffic_columns = df.index.tolist()

    # Traffic columns (exclude OI)
    traffic_columns = [
        'L3 Misses',
        'L3 DRead + DWrite',
        'L3 DMisses',
        'L3 LST Miss',
        'L3 Access',
        'Max(L1, L2, L3) Misses',
        'LST',
        "Symbolic Bytes",
    ]

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
            legend=False  # avoid repeating legend
        )

        ax.set_title(f"Benchmark: {benchmark}")
        ax.set_xlabel("Variant / Formula")
        ax.set_ylabel("Bytes Moved")
        ax.tick_params(axis='x', rotation=90)

    # Hide empty subplots
    for j in range(i+1, n_rows*n_cols):
        fig.delaxes(axes[j // n_cols, j % n_cols])

    plt.tight_layout()
    plt.ylim(bottom=0)

    plt.savefig("traffic_comparison.pdf")

def plot_flops(df: pd.DataFrame):
    """
    Plot PAPI FLOPs vs Symbolic FLOPs with benchmarks on y-axis
    """
    # Filter out benchmarks where both values are 0
    print(df)
    df_filtered = df[(df['PAPI FLOPs'] > 0) | (df['work'] > 0)].copy()
    
    # Sort benchmarks alphabetically
    df_filtered = df_filtered.sort_values('benchmark', ascending=False)
    
    # Create figure with extra space for legend
    fig, ax = plt.subplots(figsize=(12, max(10, len(df_filtered) * 0.5)))
    
    benchmarks = df_filtered['benchmark'].values
    papi_flops = df_filtered['PAPI FLOPs'].values
    symbolic_flops = df_filtered['work'].values
    
    y_positions = np.arange(len(benchmarks))
    
    # Add diagonal lines connecting the two points for each benchmark (plot first so they're in background)
    for i, (papi, symbolic) in enumerate(zip(papi_flops, symbolic_flops)):
        ax.plot([papi, symbolic], [i, i], 'k-', alpha=0.2, linewidth=1.5)
    
    # Plot symbolic FLOPs first (so PAPI appears on top)
    ax.scatter(symbolic_flops, y_positions, label='Symbolic FLOPs (work)', 
              s=150, alpha=0.7, marker='s', color='#ff7f0e', edgecolors='black', linewidth=1)
    
    # Plot PAPI FLOPs on top with smaller marker
    ax.scatter(papi_flops, y_positions, label='PAPI FLOPs', 
              s=120, alpha=0.8, marker='o', color='#1f77b4', edgecolors='black', linewidth=1)
    
    # Set y-axis
    ax.set_yticks(y_positions)
    ax.set_yticklabels(benchmarks, fontsize=16)
    ax.set_ylabel('Benchmark Name', fontsize=20)
    
    # Set x-axis
    ax.set_xlabel('FLOPs', fontsize=20)
    ax.set_xscale('log')
    # Major grid lines (at 10^n)
    ax.grid(True, which='major', axis='x', linestyle='-', linewidth=1, alpha=0.3, color='gray')
    # Minor grid lines (at 2*10^n, 3*10^n, etc.)
    ax.grid(True, which='minor', axis='x', linestyle=':', linewidth=0.5, alpha=0.15, color='gray')
    ax.set_xlim(left=max(1, min(papi_flops.min(), symbolic_flops.min()) * 0.5))
    ax.tick_params(axis='x', labelsize=16)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='x', linewidth=1)
    
    # Add legend outside plot area
    ax.legend(fontsize=16, loc='upper center', bbox_to_anchor=(0.5, -0.05), 
             ncol=2, framealpha=0.95, edgecolor='black', fancybox=True, shadow=True)
    
    # Add title
    ax.set_title('FLOPs Obtained Through Hardware Counters vs\n' \
                 'Symbolic Analysis', 
                fontsize=22, pad=20)
    
    plt.tight_layout()
    plt.savefig("flops_comparison.pdf", bbox_inches='tight', dpi=900)
    plt.savefig("flops_comparison.png", bbox_inches='tight', dpi=900)
    plt.close()
    print("✓ Saved: flops_comparison.pdf and flops_comparison.png")

if __name__ == "__main__":
    main()
