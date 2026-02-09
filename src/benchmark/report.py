"""Report generation and visualization."""

import json
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_results(path: str) -> Dict:
    """Load benchmark results from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def results_to_dataframe(results: Dict) -> pd.DataFrame:
    """Convert benchmark results to pandas DataFrame."""
    rows = []
    
    for result in results["results"]:
        row = {
            "test_name": result["test_name"],
            "scenario": result["scenario"],
            "num_categories": result["num_categories"],
            "insert_p50_ms": result["insert"]["latency"]["p50_ms"],
            "insert_p95_ms": result["insert"]["latency"]["p95_ms"],
            "insert_throughput": result["insert"]["throughput"]["vectors_per_second"],
            "search_p50_ms": result["search"]["latency"]["p50_ms"],
            "search_p95_ms": result["search"]["latency"]["p95_ms"],
            "search_qps": result["search"]["qps"],
            "ram_mb": result["resources"]["ram_mb"],
            "setup_time_seconds": result["setup_time_seconds"],
            "total_time_seconds": result["total_time_seconds"],
        }
        
        if result["search"].get("recall"):
            row["recall"] = result["search"]["recall"].get("recall@10", 0)
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def generate_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary table comparing scenarios."""
    # Group by scenario, num_categories, test_name
    summary = df.groupby(["scenario", "num_categories", "test_name"]).agg({
        "insert_p50_ms": "mean",
        "insert_p95_ms": "mean",
        "insert_throughput": "mean",
        "search_p50_ms": "mean",
        "search_p95_ms": "mean",
        "search_qps": "mean",
        "ram_mb": "mean",
    }).round(2)
    
    return summary.reset_index()


def print_comparison_table(df: pd.DataFrame, test_name: str = None):
    """Print formatted comparison table."""
    if test_name:
        df = df[df["test_name"] == test_name]
    
    summary = generate_summary_table(df)
    
    print("\n" + "="*100)
    print("BENCHMARK COMPARISON SUMMARY")
    print("="*100)
    
    # Group by test name
    for name in summary["test_name"].unique():
        test_df = summary[summary["test_name"] == name]
        
        print(f"\n--- {name} ---")
        print(f"{'Scenario':<12} {'Categories':<12} {'Search P50':<12} {'Search P95':<12} {'QPS':<12} {'Insert P50':<12} {'RAM (MB)':<12}")
        print("-" * 84)
        
        for _, row in test_df.iterrows():
            scenario = row["scenario"]
            cats = int(row["num_categories"]) if row["num_categories"] > 0 else "N/A"
            print(f"{scenario:<12} {str(cats):<12} {row['search_p50_ms']:<12.2f} {row['search_p95_ms']:<12.2f} {row['search_qps']:<12.2f} {row['insert_p50_ms']:<12.2f} {row['ram_mb']:<12.2f}")


def plot_latency_comparison(
    df: pd.DataFrame,
    output_dir: str = "./results",
    test_name: Optional[str] = None,
):
    """Generate latency comparison plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    if test_name:
        df = df[df["test_name"] == test_name]
    
    # Get unique test names
    test_names = df["test_name"].unique()
    
    for name in test_names:
        test_df = df[df["test_name"] == name]
        
        # Separate by scenario
        scenario_a = test_df[test_df["scenario"] == "A"]
        scenario_b = test_df[test_df["scenario"] == "B"]
        
        if len(scenario_a) == 0 or len(scenario_b) == 0:
            continue
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot search latency vs categories
        ax1 = axes[0]
        
        a_cats = scenario_a.groupby("num_categories")["search_p50_ms"].mean()
        b_cats = scenario_b.groupby("num_categories")["search_p50_ms"].mean()
        
        x = np.arange(len(a_cats))
        width = 0.35
        
        ax1.bar(x - width/2, a_cats.values, width, label="Scenario A (single collection)", color="steelblue")
        ax1.bar(x + width/2, b_cats.values, width, label="Scenario B (multi collection)", color="coral")
        
        ax1.set_xlabel("Number of Categories")
        ax1.set_ylabel("Search Latency P50 (ms)")
        ax1.set_title(f"{name}: Search Latency vs Categories")
        ax1.set_xticks(x)
        ax1.set_xticklabels([str(int(c)) for c in a_cats.index])
        ax1.legend()
        ax1.grid(axis="y", alpha=0.3)
        
        # Plot QPS comparison
        ax2 = axes[1]
        
        a_qps = scenario_a.groupby("num_categories")["search_qps"].mean()
        b_qps = scenario_b.groupby("num_categories")["search_qps"].mean()
        
        ax2.bar(x - width/2, a_qps.values, width, label="Scenario A (single collection)", color="steelblue")
        ax2.bar(x + width/2, b_qps.values, width, label="Scenario B (multi collection)", color="coral")
        
        ax2.set_xlabel("Number of Categories")
        ax2.set_ylabel("Queries Per Second (QPS)")
        ax2.set_title(f"{name}: QPS vs Categories")
        ax2.set_xticks(x)
        ax2.set_xticklabels([str(int(c)) for c in a_qps.index])
        ax2.legend()
        ax2.grid(axis="y", alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"latency_{name}.png"), dpi=150)
        plt.close()
        
        print(f"Saved plot: latency_{name}.png")


def plot_all_patterns_comparison(df: pd.DataFrame, output_dir: str = "./results"):
    """Generate comparison plot across all patterns."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter to a specific category count for comparison
    category_counts = df["num_categories"].unique()
    category_counts = [c for c in category_counts if c > 0]
    
    if not category_counts:
        return
    
    # Use the middle category count
    target_cats = sorted(category_counts)[len(category_counts) // 2]
    df_filtered = df[df["num_categories"] == target_cats]
    
    test_names = df_filtered["test_name"].unique()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(test_names))
    width = 0.35
    
    a_latencies = []
    b_latencies = []
    
    for name in test_names:
        test_df = df_filtered[df_filtered["test_name"] == name]
        a_latencies.append(test_df[test_df["scenario"] == "A"]["search_p50_ms"].mean())
        b_latencies.append(test_df[test_df["scenario"] == "B"]["search_p50_ms"].mean())
    
    ax.bar(x - width/2, a_latencies, width, label="Scenario A (single collection)", color="steelblue")
    ax.bar(x + width/2, b_latencies, width, label="Scenario B (multi collection)", color="coral")
    
    ax.set_xlabel("Query Pattern")
    ax.set_ylabel("Search Latency P50 (ms)")
    ax.set_title(f"Search Latency Comparison ({int(target_cats)} categories)")
    ax.set_xticks(x)
    ax.set_xticklabels(test_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "patterns_comparison.png"), dpi=150)
    plt.close()
    
    print("Saved plot: patterns_comparison.png")


def plot_scaling_analysis(df: pd.DataFrame, output_dir: str = "./results"):
    """Generate scaling analysis plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get one pattern for clean scaling view
    test_name = "uniform_random"
    df_pattern = df[df["test_name"] == test_name]
    
    if len(df_pattern) == 0:
        if len(df) == 0:
            return
        test_name = df["test_name"].iloc[0]
        df_pattern = df[df["test_name"] == test_name]
    
    scenario_a = df_pattern[df_pattern["scenario"] == "A"]
    scenario_b = df_pattern[df_pattern["scenario"] == "B"]
    
    if len(scenario_a) == 0 or len(scenario_b) == 0:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Search latency scaling
    ax1 = axes[0, 0]
    a_data = scenario_a.groupby("num_categories")["search_p50_ms"].mean()
    b_data = scenario_b.groupby("num_categories")["search_p50_ms"].mean()
    
    ax1.plot(a_data.index, a_data.values, "o-", label="Scenario A", color="steelblue", linewidth=2, markersize=8)
    ax1.plot(b_data.index, b_data.values, "s-", label="Scenario B", color="coral", linewidth=2, markersize=8)
    ax1.set_xlabel("Number of Categories")
    ax1.set_ylabel("Search Latency P50 (ms)")
    ax1.set_title("Search Latency Scaling")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_xscale("log")
    
    # Search P95 scaling
    ax2 = axes[0, 1]
    a_data = scenario_a.groupby("num_categories")["search_p95_ms"].mean()
    b_data = scenario_b.groupby("num_categories")["search_p95_ms"].mean()
    
    ax2.plot(a_data.index, a_data.values, "o-", label="Scenario A", color="steelblue", linewidth=2, markersize=8)
    ax2.plot(b_data.index, b_data.values, "s-", label="Scenario B", color="coral", linewidth=2, markersize=8)
    ax2.set_xlabel("Number of Categories")
    ax2.set_ylabel("Search Latency P95 (ms)")
    ax2.set_title("Search Latency P95 Scaling")
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_xscale("log")
    
    # QPS scaling
    ax3 = axes[1, 0]
    a_data = scenario_a.groupby("num_categories")["search_qps"].mean()
    b_data = scenario_b.groupby("num_categories")["search_qps"].mean()
    
    ax3.plot(a_data.index, a_data.values, "o-", label="Scenario A", color="steelblue", linewidth=2, markersize=8)
    ax3.plot(b_data.index, b_data.values, "s-", label="Scenario B", color="coral", linewidth=2, markersize=8)
    ax3.set_xlabel("Number of Categories")
    ax3.set_ylabel("Queries Per Second")
    ax3.set_title("QPS Scaling")
    ax3.legend()
    ax3.grid(alpha=0.3)
    ax3.set_xscale("log")
    
    # Insert throughput scaling
    ax4 = axes[1, 1]
    a_data = scenario_a.groupby("num_categories")["insert_throughput"].mean()
    b_data = scenario_b.groupby("num_categories")["insert_throughput"].mean()
    
    ax4.plot(a_data.index, a_data.values, "o-", label="Scenario A", color="steelblue", linewidth=2, markersize=8)
    ax4.plot(b_data.index, b_data.values, "s-", label="Scenario B", color="coral", linewidth=2, markersize=8)
    ax4.set_xlabel("Number of Categories")
    ax4.set_ylabel("Vectors Per Second")
    ax4.set_title("Insert Throughput Scaling")
    ax4.legend()
    ax4.grid(alpha=0.3)
    ax4.set_xscale("log")
    
    plt.suptitle(f"Scaling Analysis ({test_name})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scaling_analysis.png"), dpi=150)
    plt.close()
    print("Saved plot: scaling_analysis.png")


def plot_ram_comparison(df: pd.DataFrame, output_dir: str = "./results"):
    """Generate RAM usage comparison plot."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter out baseline and insert tests for cleaner comparison
    df_filtered = df[(df["num_categories"] > 0) & (df["test_name"] != "insert")]
    
    if len(df_filtered) == 0:
        return
    
    # Get a representative pattern
    test_names = df_filtered["test_name"].unique()
    if len(test_names) == 0:
        return
    
    test_name = "uniform_random" if "uniform_random" in test_names else test_names[0]
    df_pattern = df_filtered[df_filtered["test_name"] == test_name]
    
    scenario_a = df_pattern[df_pattern["scenario"] == "A"]
    scenario_b = df_pattern[df_pattern["scenario"] == "B"]
    
    if len(scenario_a) == 0 or len(scenario_b) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    a_data = scenario_a.groupby("num_categories")["ram_mb"].mean()
    b_data = scenario_b.groupby("num_categories")["ram_mb"].mean()
    
    x = np.arange(len(a_data))
    width = 0.35
    
    ax.bar(x - width/2, a_data.values, width, label="Scenario A (single collection)", color="steelblue")
    ax.bar(x + width/2, b_data.values, width, label="Scenario B (multi collection)", color="coral")
    
    ax.set_xlabel("Number of Categories")
    ax.set_ylabel("RAM Usage (MB)")
    ax.set_title(f"RAM Usage Comparison ({test_name})")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(c)) for c in a_data.index])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ram_comparison.png"), dpi=150)
    plt.close()
    
    print("Saved plot: ram_comparison.png")


def plot_insert_throughput_comparison(df: pd.DataFrame, output_dir: str = "./results"):
    """Generate insert throughput comparison plot."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter to insert tests only
    df_insert = df[df["test_name"] == "insert"]
    
    if len(df_insert) == 0:
        return
    
    scenario_a = df_insert[df_insert["scenario"] == "A"]
    scenario_b = df_insert[df_insert["scenario"] == "B"]
    
    if len(scenario_a) == 0 or len(scenario_b) == 0:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Throughput comparison
    a_throughput = scenario_a.groupby("num_categories")["insert_throughput"].mean()
    b_throughput = scenario_b.groupby("num_categories")["insert_throughput"].mean()
    
    x = np.arange(len(a_throughput))
    width = 0.35
    
    ax1.bar(x - width/2, a_throughput.values, width, label="Scenario A", color="steelblue")
    ax1.bar(x + width/2, b_throughput.values, width, label="Scenario B", color="coral")
    ax1.set_xlabel("Number of Categories")
    ax1.set_ylabel("Vectors Per Second")
    ax1.set_title("Insert Throughput Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(int(c)) for c in a_throughput.index])
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
    
    # Insert latency P50 comparison
    a_latency = scenario_a.groupby("num_categories")["insert_p50_ms"].mean()
    b_latency = scenario_b.groupby("num_categories")["insert_p50_ms"].mean()
    
    ax2.bar(x - width/2, a_latency.values, width, label="Scenario A", color="steelblue")
    ax2.bar(x + width/2, b_latency.values, width, label="Scenario B", color="coral")
    ax2.set_xlabel("Number of Categories")
    ax2.set_ylabel("Insert Latency P50 (ms)")
    ax2.set_title("Insert Latency P50 Comparison")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(int(c)) for c in a_latency.index])
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "insert_comparison.png"), dpi=150)
    plt.close()
    
    print("Saved plot: insert_comparison.png")


def plot_latency_percentiles(df: pd.DataFrame, output_dir: str = "./results"):
    """Generate latency percentiles comparison (P50, P95) for all scenarios."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter out insert tests
    df_search = df[df["test_name"] != "insert"]
    
    if len(df_search) == 0:
        return
    
    # Get middle category count
    category_counts = df_search["num_categories"].unique()
    category_counts = [c for c in category_counts if c > 0]
    
    if not category_counts:
        return
    
    target_cats = sorted(category_counts)[len(category_counts) // 2]
    df_filtered = df_search[df_search["num_categories"] == target_cats]
    
    test_names = df_filtered["test_name"].unique()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.arange(len(test_names))
    width = 0.35
    
    # P50 comparison
    a_p50 = []
    b_p50 = []
    for name in test_names:
        test_df = df_filtered[df_filtered["test_name"] == name]
        a_p50.append(test_df[test_df["scenario"] == "A"]["search_p50_ms"].mean())
        b_p50.append(test_df[test_df["scenario"] == "B"]["search_p50_ms"].mean())
    
    ax1.bar(x - width/2, a_p50, width, label="Scenario A", color="steelblue")
    ax1.bar(x + width/2, b_p50, width, label="Scenario B", color="coral")
    ax1.set_xlabel("Query Pattern")
    ax1.set_ylabel("Latency P50 (ms)")
    ax1.set_title(f"Search Latency P50 ({int(target_cats)} categories)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(test_names, rotation=45, ha="right")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
    
    # P95 comparison
    a_p95 = []
    b_p95 = []
    for name in test_names:
        test_df = df_filtered[df_filtered["test_name"] == name]
        a_p95.append(test_df[test_df["scenario"] == "A"]["search_p95_ms"].mean())
        b_p95.append(test_df[test_df["scenario"] == "B"]["search_p95_ms"].mean())
    
    ax2.bar(x - width/2, a_p95, width, label="Scenario A", color="steelblue")
    ax2.bar(x + width/2, b_p95, width, label="Scenario B", color="coral")
    ax2.set_xlabel("Query Pattern")
    ax2.set_ylabel("Latency P95 (ms)")
    ax2.set_title(f"Search Latency P95 ({int(target_cats)} categories)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(test_names, rotation=45, ha="right")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "latency_percentiles.png"), dpi=150)
    plt.close()
    
    print("Saved plot: latency_percentiles.png")


def plot_performance_ratio(df: pd.DataFrame, output_dir: str = "./results"):
    """Generate performance ratio plot (Scenario B / Scenario A).
    
    Values > 1.0 mean Scenario B is slower (worse).
    Values < 1.0 mean Scenario B is faster (better).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter out insert tests
    df_search = df[df["test_name"] != "insert"]
    
    if len(df_search) == 0:
        return
    
    # Get unique test names and category counts
    test_names = df_search["test_name"].unique()
    category_counts = sorted([c for c in df_search["num_categories"].unique() if c > 0])
    
    if len(test_names) == 0 or len(category_counts) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate ratio for each test pattern
    x = np.arange(len(test_names))
    width = 0.15
    
    for i, cat_count in enumerate(category_counts):
        ratios = []
        for test_name in test_names:
            test_df = df_search[(df_search["test_name"] == test_name) & (df_search["num_categories"] == cat_count)]
            a_latency = test_df[test_df["scenario"] == "A"]["search_p50_ms"].mean()
            b_latency = test_df[test_df["scenario"] == "B"]["search_p50_ms"].mean()
            
            if a_latency > 0:
                ratio = b_latency / a_latency
            else:
                ratio = 1.0
            ratios.append(ratio)
        
        offset = (i - len(category_counts)/2 + 0.5) * width
        ax.bar(x + offset, ratios, width, label=f"{int(cat_count)} categories")
    
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, label='Equal performance')
    ax.set_xlabel("Query Pattern")
    ax.set_ylabel("Performance Ratio (Scenario B / Scenario A)")
    ax.set_title("Performance Ratio: Scenario B vs A\n(Higher = Scenario A is better, Lower = Scenario B is better)")
    ax.set_xticks(x)
    ax.set_xticklabels(test_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "performance_ratio.png"), dpi=150)
    plt.close()
    
    print("Saved plot: performance_ratio.png")


def generate_markdown_report(results: Dict, output_path: str):
    """Generate a markdown report from benchmark results."""
    df = results_to_dataframe(results)
    summary = generate_summary_table(df)
    
    config = results["config"]
    
    lines = [
        "# Qdrant Multi-tenant Benchmark Results",
        "",
        f"**Timestamp:** {results['timestamp']}",
        "",
        "## Configuration",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Dataset | {config['dataset']['name']} |",
        f"| Vectors | {config['dataset']['num_vectors']:,} |",
        f"| Dimensions | {config['dataset']['dimensions']} |",
        f"| Distance | {config['dataset']['distance']} |",
        f"| HNSW M | {config['hnsw']['m']} |",
        f"| HNSW ef_construct | {config['hnsw']['ef_construct']} |",
        f"| HNSW ef_search | {config['hnsw']['ef_search']} |",
        f"| Top-K | {config['search']['top_k']} |",
        f"| Queries | {config['benchmark']['num_queries']} |",
        f"| Categories Tested | {config['category_counts']} |",
        "",
        "## Summary Results",
        "",
    ]
    
    # Add summary table
    for name in summary["test_name"].unique():
        test_df = summary[summary["test_name"] == name]
        
        lines.extend([
            f"### {name}",
            "",
            "| Scenario | Categories | Search P50 (ms) | Search P95 (ms) | QPS | Insert Throughput |",
            "|----------|------------|-----------------|-----------------|-----|-------------------|",
        ])
        
        for _, row in test_df.iterrows():
            scenario = row["scenario"]
            cats = int(row["num_categories"]) if row["num_categories"] > 0 else "N/A"
            lines.append(
                f"| {scenario} | {cats} | {row['search_p50_ms']:.2f} | "
                f"{row['search_p95_ms']:.2f} | {row['search_qps']:.2f} | "
                f"{row['insert_throughput']:.2f} |"
            )
        
        lines.extend(["", ""])
    
    lines.extend([
        "## Interpretation",
        "",
        "- **Scenario A**: Single collection with payload filtering by `category_id`",
        "- **Scenario B**: Multiple collections (one per category)",
        "- **Baseline**: Single collection without any filtering",
        "",
        "### Key Observations",
        "",
        "1. **Search Latency**: Compare P50/P95 values between scenarios",
        "2. **QPS**: Higher is better - indicates system throughput",
        "3. **Insert Throughput**: Vectors inserted per second",
        "4. **Scaling**: How performance changes with number of categories",
        "",
    ])
    
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"Saved report: {output_path}")


def generate_full_report(results_path: str, output_dir: str = "./results"):
    """Generate complete report with tables and plots."""
    results = load_results(results_path)
    df = results_to_dataframe(results)
    
    # Print comparison table to console
    print_comparison_table(df)
    
    # Generate plots
    plot_latency_comparison(df, output_dir)
    plot_all_patterns_comparison(df, output_dir)
    plot_scaling_analysis(df, output_dir)
    
    # Generate new plots
    plot_ram_comparison(df, output_dir)
    plot_insert_throughput_comparison(df, output_dir)
    plot_latency_percentiles(df, output_dir)
    plot_performance_ratio(df, output_dir)
    
    # Generate markdown report
    generate_markdown_report(results, os.path.join(output_dir, "RESULTS.md"))
