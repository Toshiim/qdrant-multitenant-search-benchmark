"""Tests for configuration module."""

import os
import tempfile

import pytest

from benchmark.config import (
    Config,
    HNSWConfig,
    QdrantConfig,
    SearchConfig,
    load_config,
)


class TestConfig:
    """Tests for Config class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = Config.default()
        
        assert config.qdrant.host == "localhost"
        assert config.qdrant.port == 6333
        assert config.hnsw.m == 16
        assert config.hnsw.ef_construct == 100
        assert config.search.top_k == 10
        assert config.benchmark.num_queries == 1000

    def test_from_yaml(self):
        """Test loading configuration from YAML file."""
        yaml_content = """
qdrant:
  host: "test-host"
  port: 7333

hnsw:
  m: 32
  ef_construct: 200
  ef_search: 128

search:
  top_k: 20
  distance_metric: "Euclidean"

categories:
  counts: [5, 50]

benchmark:
  num_queries: 500
  batch_size: 50
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            config = Config.from_yaml(temp_path)
            
            assert config.qdrant.host == "test-host"
            assert config.qdrant.port == 7333
            assert config.hnsw.m == 32
            assert config.hnsw.ef_construct == 200
            assert config.hnsw.ef_search == 128
            assert config.search.top_k == 20
            assert config.search.distance_metric == "Euclidean"
            assert config.categories["counts"] == [5, 50]
            assert config.benchmark.num_queries == 500
            assert config.benchmark.batch_size == 50
        finally:
            os.unlink(temp_path)

    def test_load_config_default(self):
        """Test load_config returns default when no file exists."""
        config = load_config("/nonexistent/path.yaml")
        
        # Should return default config
        assert config.qdrant.host == "localhost"
        assert config.hnsw.m == 16


class TestQdrantConfig:
    """Tests for QdrantConfig class."""

    def test_defaults(self):
        """Test default values."""
        config = QdrantConfig()
        
        assert config.host == "localhost"
        assert config.port == 6333
        assert config.grpc_port == 6334
        assert config.timeout == 300


class TestHNSWConfig:
    """Tests for HNSWConfig class."""

    def test_defaults(self):
        """Test default values."""
        config = HNSWConfig()
        
        assert config.m == 16
        assert config.ef_construct == 100
        assert config.ef_search == 64


class TestSearchConfig:
    """Tests for SearchConfig class."""

    def test_defaults(self):
        """Test default values."""
        config = SearchConfig()
        
        assert config.top_k == 10
        assert config.distance_metric == "Cosine"
