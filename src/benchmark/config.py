"""Configuration management for the benchmark."""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml


@dataclass
class QdrantConfig:
    """Qdrant connection configuration."""

    host: str = "localhost"
    port: int = 6333
    grpc_port: int = 6334
    timeout: int = 300


@dataclass
class HNSWConfig:
    """HNSW index parameters."""

    m: int = 16
    ef_construct: int = 100
    ef_search: int = 64


@dataclass
class SearchConfig:
    """Search parameters."""

    top_k: int = 10
    distance_metric: str = "Cosine"


@dataclass
class SyntheticDatasetConfig:
    """Synthetic dataset configuration."""

    num_vectors: int = 100000
    dimensions: int = 128
    distance: str = "Cosine"


@dataclass
class DatasetConfig:
    """Dataset configuration."""

    name: str = "synthetic"
    synthetic: SyntheticDatasetConfig = field(default_factory=SyntheticDatasetConfig)


@dataclass
class BenchmarkConfig:
    """Benchmark execution parameters."""

    num_queries: int = 1000
    batch_size: int = 100
    warmup_queries: int = 100
    repeat: int = 3


@dataclass
class HotCategoryConfig:
    """Hot Category Loop test configuration."""

    category_index: int = 0


@dataclass
class BatchSweepConfig:
    """Category Batch Sweep test configuration."""

    queries_per_category: int = 100


@dataclass
class ZipfianConfig:
    """Zipfian Distribution test configuration."""

    alpha: float = 1.0


@dataclass
class QueryPatternsConfig:
    """Query pattern specific settings."""

    hot_category: HotCategoryConfig = field(default_factory=HotCategoryConfig)
    batch_sweep: BatchSweepConfig = field(default_factory=BatchSweepConfig)
    zipfian: ZipfianConfig = field(default_factory=ZipfianConfig)


@dataclass
class MonitoringConfig:
    """Resource monitoring configuration."""

    enabled: bool = True
    interval_seconds: int = 1


@dataclass
class OutputConfig:
    """Output configuration."""

    results_dir: str = "./results"
    save_raw_metrics: bool = True
    generate_plots: bool = True


@dataclass
class Config:
    """Main configuration class."""

    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    hnsw: HNSWConfig = field(default_factory=HNSWConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    categories: Dict = field(default_factory=lambda: {"counts": [10, 100, 1000]})
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    query_patterns: QueryPatternsConfig = field(default_factory=QueryPatternsConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        config = cls()

        if "qdrant" in data:
            config.qdrant = QdrantConfig(**data["qdrant"])

        if "hnsw" in data:
            config.hnsw = HNSWConfig(**data["hnsw"])

        if "search" in data:
            config.search = SearchConfig(**data["search"])

        if "categories" in data:
            config.categories = data["categories"]

        if "dataset" in data:
            ds_data = data["dataset"]
            synthetic_config = SyntheticDatasetConfig(
                **ds_data.get("synthetic", {})
            )
            config.dataset = DatasetConfig(
                name=ds_data.get("name", "synthetic"),
                synthetic=synthetic_config,
            )

        if "benchmark" in data:
            config.benchmark = BenchmarkConfig(**data["benchmark"])

        if "query_patterns" in data:
            qp_data = data["query_patterns"]
            config.query_patterns = QueryPatternsConfig(
                hot_category=HotCategoryConfig(**qp_data.get("hot_category", {})),
                batch_sweep=BatchSweepConfig(**qp_data.get("batch_sweep", {})),
                zipfian=ZipfianConfig(**qp_data.get("zipfian", {})),
            )

        if "monitoring" in data:
            config.monitoring = MonitoringConfig(**data["monitoring"])

        if "output" in data:
            config.output = OutputConfig(**data["output"])

        return config

    @classmethod
    def default(cls) -> "Config":
        """Create default configuration."""
        return cls()


def load_config(path: Optional[str] = None) -> Config:
    """Load configuration from file or return defaults."""
    if path and os.path.exists(path):
        return Config.from_yaml(path)
    
    # Try default locations
    default_paths = ["config.yaml", "config.yml", "benchmark_config.yaml"]
    for default_path in default_paths:
        if os.path.exists(default_path):
            return Config.from_yaml(default_path)
    
    return Config.default()
