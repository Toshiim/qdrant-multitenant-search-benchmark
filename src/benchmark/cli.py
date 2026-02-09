"""Command-line interface for the benchmark."""

import argparse
import json
import os
import sys
import time

from .config import load_config
from .dataset import load_dataset
from .query_patterns import get_all_patterns, get_pattern_by_name
from .report import generate_full_report
from .runner import BenchmarkRunner


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Qdrant Multi-tenant Search Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full benchmark with default config
  python -m benchmark.cli run

  # Run with specific config file
  python -m benchmark.cli run --config my_config.yaml

  # Run only specific patterns
  python -m benchmark.cli run --patterns hot_category_loop,uniform_random

  # Run with specific category counts
  python -m benchmark.cli run --categories 10,50,100

  # Skip collection loading (reuse existing collections - faster)
  python -m benchmark.cli run --skip-load

  # Run with cold cache (reset index before each test)
  python -m benchmark.cli run --skip-load --reset-cache

  # Load collections only (measure load metrics separately)
  python -m benchmark.cli load

  # Generate report from existing results
  python -m benchmark.cli report --input results/benchmark_20240101_120000.json

Query Patterns:
  - hot_category_loop      : All queries to single category
  - category_batch_sweep   : Sequential batches per category
  - interleaved_categories : Round-robin category switching
  - uniform_random         : Random category selection
  - zipfian_distribution   : Skewed hot/cold distribution
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run benchmark")
    run_parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    run_parser.add_argument(
        "--patterns", "-p",
        type=str,
        default=None,
        help="Comma-separated list of patterns to run",
    )
    run_parser.add_argument(
        "--categories", "-n",
        type=str,
        default=None,
        help="Comma-separated list of category counts to test",
    )
    run_parser.add_argument(
        "--output", "-o",
        type=str,
        default="./results",
        help="Output directory for results",
    )
    run_parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip baseline test",
    )
    run_parser.add_argument(
        "--scenario",
        type=str,
        choices=["A", "B", "both"],
        default="both",
        help="Run only specific scenario",
    )
    run_parser.add_argument(
        "--skip-load",
        action="store_true",
        help="Skip collection loading if collections exist (faster benchmark)",
    )
    run_parser.add_argument(
        "--reset-cache",
        action="store_true",
        help="Reset HNSW index cache before each search test (cold start simulation)",
    )

    # Load command (new)
    load_parser = subparsers.add_parser("load", help="Load collections only (measure load metrics)")
    load_parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    load_parser.add_argument(
        "--categories", "-n",
        type=str,
        default=None,
        help="Comma-separated list of category counts to test",
    )
    load_parser.add_argument(
        "--output", "-o",
        type=str,
        default="./results",
        help="Output directory for results",
    )
    load_parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip baseline collection",
    )
    load_parser.add_argument(
        "--scenario",
        type=str,
        choices=["A", "B", "both"],
        default="both",
        help="Load only specific scenario",
    )

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate report from results")
    report_parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to benchmark results JSON file",
    )
    report_parser.add_argument(
        "--output", "-o",
        type=str,
        default="./results",
        help="Output directory for report",
    )

    # Info command
    info_parser = subparsers.add_parser("info", help="Show available options")

    return parser.parse_args()


def cmd_run(args):
    """Execute benchmark run command."""
    print("="*60)
    print("Qdrant Multi-tenant Search Benchmark")
    print("="*60)

    # Load configuration
    config = load_config(args.config)
    print(f"\nLoaded config from: {args.config}")
    print(f"  HNSW: m={config.hnsw.m}, ef_construct={config.hnsw.ef_construct}, ef_search={config.hnsw.ef_search}")
    print(f"  Search: top_k={config.search.top_k}, distance={config.search.distance_metric}")
    print(f"  Benchmark: {config.benchmark.num_queries} queries, {config.benchmark.repeat} repeats")

    # Load dataset
    print(f"\nLoading dataset: {config.dataset.name}")
    dataset = load_dataset(config)
    print(f"  Vectors: {len(dataset.vectors):,}")
    print(f"  Dimensions: {dataset.dimensions}")
    print(f"  Distance: {dataset.distance}")
    print(f"  Query vectors: {len(dataset.queries)}")

    # Determine patterns to run
    if args.patterns:
        pattern_names = [p.strip() for p in args.patterns.split(",")]
        patterns = [get_pattern_by_name(name, config) for name in pattern_names]
    else:
        patterns = get_all_patterns(config)

    print(f"\nPatterns to run:")
    for p in patterns:
        print(f"  - {p.name}: {p.description}")

    # Determine category counts
    if args.categories:
        category_counts = [int(c.strip()) for c in args.categories.split(",")]
    else:
        category_counts = config.categories.get("counts", [10, 100])

    print(f"\nCategory counts: {category_counts}")

    # Create runner and execute
    runner = BenchmarkRunner(config, dataset)

    include_baseline = not args.no_baseline
    skip_load = args.skip_load
    reset_cache = args.reset_cache

    print(f"\nInclude baseline: {include_baseline}")
    print(f"Skip load: {skip_load}")
    print(f"Reset cache (cold start): {reset_cache}")

    result = runner.run_all(
        patterns=patterns,
        category_counts=category_counts,
        include_baseline=include_baseline,
        skip_load=skip_load,
        reset_cache=reset_cache,
    )

    # Save results
    os.makedirs(args.output, exist_ok=True)
    results_path = os.path.join(args.output, f"benchmark_{result.timestamp}.json")
    result.save(results_path)
    print(f"\n\nResults saved to: {results_path}")

    # Generate report
    generate_full_report(results_path, args.output)

    print("\n" + "="*60)
    print("Benchmark complete!")
    print("="*60)

    return 0


def cmd_load(args):
    """Execute collection load command (measure load metrics separately)."""
    print("="*60)
    print("Qdrant Multi-tenant Benchmark - Collection Loading")
    print("="*60)

    # Load configuration
    config = load_config(args.config)
    print(f"\nLoaded config from: {args.config}")
    print(f"  HNSW: m={config.hnsw.m}, ef_construct={config.hnsw.ef_construct}")
    print(f"  Batch size: {config.benchmark.batch_size}")

    # Load dataset
    print(f"\nLoading dataset: {config.dataset.name}")
    dataset = load_dataset(config)
    print(f"  Vectors: {len(dataset.vectors):,}")
    print(f"  Dimensions: {dataset.dimensions}")
    print(f"  Distance: {dataset.distance}")

    # Determine category counts
    if args.categories:
        category_counts = [int(c.strip()) for c in args.categories.split(",")]
    else:
        category_counts = config.categories.get("counts", [10, 100])

    print(f"\nCategory counts: {category_counts}")

    # Create runner
    runner = BenchmarkRunner(config, dataset)

    include_baseline = not args.no_baseline
    print(f"\nInclude baseline: {include_baseline}")
    print(f"Scenario: {args.scenario}")

    # Load collections
    load_results = runner.load_collections(
        category_counts=category_counts,
        include_baseline=include_baseline,
        scenarios=args.scenario,
    )

    # Save load results
    os.makedirs(args.output, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(args.output, f"load_metrics_{timestamp}.json")
    
    results_dict = {
        "timestamp": timestamp,
        "config": {
            "qdrant": {
                "host": config.qdrant.host,
                "port": config.qdrant.port,
            },
            "hnsw": {
                "m": config.hnsw.m,
                "ef_construct": config.hnsw.ef_construct,
            },
            "dataset": {
                "name": dataset.name,
                "num_vectors": len(dataset.vectors),
                "dimensions": dataset.dimensions,
            },
            "batch_size": config.benchmark.batch_size,
        },
        "load_results": [r.to_dict() for r in load_results],
    }
    
    with open(results_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n\nLoad results saved to: {results_path}")

    # Print summary
    print("\n" + "="*60)
    print("Load Metrics Summary")
    print("="*60)
    print(f"{'Scenario':<12} {'Categories':<12} {'Setup (s)':<12} {'Insert (s)':<12} {'Throughput (vec/s)':<20}")
    print("-"*70)
    for r in load_results:
        print(f"{r.scenario:<12} {r.num_categories:<12} {r.setup_time_seconds:<12.2f} "
              f"{r.insert_metrics.latency.total_seconds:<12.2f} "
              f"{r.insert_metrics.throughput.vectors_per_second:<20.2f}")

    print("\n" + "="*60)
    print("Collection loading complete!")
    print("="*60)
    print("\nYou can now run search benchmarks with --skip-load flag:")
    print(f"  python -m benchmark.cli run --skip-load --reset-cache")

    return 0


def cmd_report(args):
    """Execute report generation command."""
    print("Generating report from:", args.input)
    generate_full_report(args.input, args.output)
    return 0


def cmd_info(args):
    """Show available options."""
    print("\nAvailable Query Patterns:")
    print("-" * 40)
    print("  hot_category_loop      - All queries to single category")
    print("  category_batch_sweep   - Sequential batches per category")
    print("  interleaved_categories - Round-robin category switching")
    print("  uniform_random         - Random category selection")
    print("  zipfian_distribution   - Skewed hot/cold distribution")

    print("\nAvailable Datasets:")
    print("-" * 40)
    print("  synthetic              - Generated random vectors")
    print("  dbpedia-openai-1M-angular  - 1M vectors, 1536 dims")
    print("  deep-image-96-angular      - 10M vectors, 96 dims")
    print("  gist-960-euclidean         - 1M vectors, 960 dims")
    print("  glove-100-angular          - 1.2M vectors, 100 dims")

    print("\nOptimization Options:")
    print("-" * 40)
    print("  --skip-load    - Skip collection loading if collections exist")
    print("                   Use after running 'load' command for faster benchmarks")
    print("  --reset-cache  - Reset HNSW index cache before each search test")
    print("                   Simulates cold start without reloading data")

    print("\nRecommended Workflow:")
    print("-" * 40)
    print("  1. Load collections once:  python -m benchmark.cli load")
    print("  2. Run benchmarks:         python -m benchmark.cli run --skip-load --reset-cache")
    print("  3. Run again (warm cache): python -m benchmark.cli run --skip-load")

    return 0


def main():
    """Main entry point."""
    args = parse_args()

    if args.command == "run":
        return cmd_run(args)
    elif args.command == "load":
        return cmd_load(args)
    elif args.command == "report":
        return cmd_report(args)
    elif args.command == "info":
        return cmd_info(args)
    else:
        print("Use --help for usage information")
        return 1


if __name__ == "__main__":
    sys.exit(main())
