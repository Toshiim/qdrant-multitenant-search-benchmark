# Qdrant Multi-tenant Search Benchmark

A comprehensive benchmark tool for comparing multi-tenant approaches in Qdrant vector database.

## Overview

This benchmark compares two fundamental approaches for implementing multi-tenancy in Qdrant:

- **Scenario A**: Single collection with payload filtering (`category_id` field)
- **Scenario B**: Multiple collections (one per category/tenant)

The goal is to provide quantitative and qualitative analysis of the trade-offs between these approaches.

## Features

- **5 Query Patterns**: Different access patterns to simulate real-world workloads
- **Baseline Testing**: Compare against non-filtered search
- **Comprehensive Metrics**: Latency (p50/p95), throughput, QPS, recall, resource usage
- **Multiple Datasets**: Support for standard ANN-benchmark datasets
- **Automated Reporting**: Tables and plots for easy comparison

## Installation

### Prerequisites

- Python 3.9+
- Docker and Docker Compose
- Qdrant (via Docker)

### Setup

```bash
# Clone the repository
git clone https://github.com/your-org/qdrant-multitenant-search-benchmark.git
cd qdrant-multitenant-search-benchmark

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .

# Start Qdrant
docker-compose up -d
```

## Usage

### Quick Start

```bash
# Run full benchmark with default settings
python -m benchmark.cli run

# Run with custom configuration
python -m benchmark.cli run --config my_config.yaml

# Run specific patterns only
python -m benchmark.cli run --patterns hot_category_loop,uniform_random

# Test specific category counts
python -m benchmark.cli run --categories 10,50,100,500
```

### Optimized Workflow (Recommended)

For faster benchmark iterations, you can separate the collection loading phase from search testing:

```bash
# Step 1: Load collections once (measure load metrics)
python -m benchmark.cli load

# Step 2: Run search benchmarks with cold cache (reuses existing collections)
python -m benchmark.cli run --skip-load --reset-cache

# Step 3: Run again with warm cache for comparison
python -m benchmark.cli run --skip-load
```

**Benefits of this workflow:**
- Collection loading is measured separately as useful benchmark data
- Search tests run much faster without reloading data each time
- `--reset-cache` option resets HNSW index to simulate cold start without reloading
- Easily compare warm vs cold cache performance

### Generate Report from Existing Results

```bash
python -m benchmark.cli report --input results/benchmark_20240101_120000.json
```

### Show Available Options

```bash
python -m benchmark.cli info
```

## Query Patterns

The benchmark includes 5 different query patterns to simulate various access scenarios:

### 1. Hot Category Loop
All queries target a single category. Simulates best-case scenario with maximum data locality and warm caches.

### 2. Category Batch Sweep
Queries are executed in sequential batches per category. Evaluates the cost of category switching and partial cache warming.

### 3. Interleaved Categories
Round-robin category switching between queries. Worst-case scenario with minimal temporal locality.

### 4. Uniform Random Categories
Categories selected uniformly at random. Average-case scenario without hot spots.

### 5. Zipfian Distribution
Skewed distribution where ~20% of categories receive ~80% of queries. Realistic production-like workload.

## Configuration

Edit `config.yaml` to customize benchmark parameters:

```yaml
# Qdrant connection
qdrant:
  host: "localhost"
  port: 6333

# HNSW index parameters
hnsw:
  m: 16
  ef_construct: 100
  ef_search: 64

# Search parameters
search:
  top_k: 10
  distance_metric: "Cosine"

# Categories to test
categories:
  counts: [10, 100, 1000]

# Dataset configuration
dataset:
  name: "synthetic"  # or standard datasets
  synthetic:
    num_vectors: 100000
    dimensions: 128

# Benchmark execution
benchmark:
  num_queries: 1000
  batch_size: 100
  warmup_queries: 100
  repeat: 3
```

## Supported Datasets

| Dataset | Vectors | Dimensions | Distance |
|---------|---------|------------|----------|
| synthetic | configurable | configurable | configurable |
| dbpedia-openai-1M-angular | 1M | 1536 | cosine |
| deep-image-96-angular | 10M | 96 | cosine |
| gist-960-euclidean | 1M | 960 | euclidean |
| glove-100-angular | 1.2M | 100 | cosine |

## Metrics Collected

### Write Metrics
- Throughput (vectors/sec)
- Latency p50/p95

### Search Metrics
- Latency p50/p95/p99
- QPS (queries per second)
- Recall@k (when ground truth available)

### Resource Metrics
- RAM consumption
- Index build time

## Output

Results are saved to the `results/` directory:
- `benchmark_TIMESTAMP.json` - Raw metrics data
- `RESULTS.md` - Markdown summary report
- `*.png` - Comparison plots

## Project Structure

```
.
├── src/benchmark/
│   ├── __init__.py
│   ├── cli.py              # Command-line interface
│   ├── config.py           # Configuration management
│   ├── dataset.py          # Dataset loading/generation
│   ├── metrics.py          # Metrics collection
│   ├── qdrant_client_wrapper.py  # Qdrant scenarios
│   ├── query_patterns.py   # Query pattern implementations
│   ├── report.py           # Report generation
│   └── runner.py           # Benchmark orchestration
├── tests/                  # Unit tests
├── config.yaml             # Default configuration
├── docker-compose.yml      # Qdrant container setup
├── pyproject.toml          # Python package config
└── requirements.txt        # Dependencies
```

## Running Tests

```bash
pytest tests/ -v
```

## Hypotheses Being Tested

1. **Collection Overhead**: Multiple collections incur memory and I/O overhead compared to a single collection with filtering.

2. **Access Pattern Impact**: The cost difference between scenarios varies significantly based on query patterns (locality vs. randomness).

3. **Scaling Behavior**: Performance characteristics change differently as the number of categories increases.

## Example Results

After running the benchmark, you'll get comparison tables like:

```
--- uniform_random ---
Scenario     Categories   Search P50   Search P95   QPS          Insert P50
A            10           1.23         2.45         812.34       5.67
A            100          1.45         3.12         689.21       5.89
B            10           0.98         1.87         1021.56      4.32
B            100          1.12         2.34         892.45       4.56
```

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

MIT License
