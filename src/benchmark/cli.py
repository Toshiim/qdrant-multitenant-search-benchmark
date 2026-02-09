"""Command-line interface for the benchmark."""

import argparse
import os
import sys

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
    print(f"\nInclude baseline: {include_baseline}")

    result = runner.run_all(
        patterns=patterns,
        category_counts=category_counts,
        include_baseline=include_baseline,
    )

    # Save results
    # Create timestamped folder: Result/{test_name}_{timestamp}/
    result_folder = os.path.join("Result", f"benchmark_{result.timestamp}")
    os.makedirs(result_folder, exist_ok=True)
    
    results_path = os.path.join(result_folder, "benchmark_results.json")
    result.save(results_path)
    print(f"\n\nResults saved to: {results_path}")

    # Generate report in the same folder
    generate_full_report(results_path, result_folder)

    print("\n" + "="*60)
    print("Benchmark complete!")
    print("="*60)

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

    return 0


def main():
    """Main entry point."""
    args = parse_args()

    if args.command == "run":
        return cmd_run(args)
    elif args.command == "report":
        return cmd_report(args)
    elif args.command == "info":
        return cmd_info(args)
    else:
        print("Use --help for usage information")
        return 1


if __name__ == "__main__":
    sys.exit(main())
