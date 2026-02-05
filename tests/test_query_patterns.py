"""Tests for query patterns module."""

import numpy as np
import pytest

from benchmark.config import Config
from benchmark.query_patterns import (
    CategoryBatchSweep,
    HotCategoryLoop,
    InterleavedCategories,
    UniformRandomCategories,
    ZipfianDistribution,
    get_all_patterns,
    get_pattern_by_name,
)


class TestHotCategoryLoop:
    """Tests for HotCategoryLoop pattern."""

    def test_all_queries_same_category(self):
        """Test that all queries go to the same category."""
        pattern = HotCategoryLoop(category_index=2)
        query_vectors = np.random.randn(10, 32).astype(np.float32)
        
        queries = list(pattern.generate_queries(query_vectors, num_categories=5, num_queries=100))
        
        assert len(queries) == 100
        
        # All queries should be for category 2
        for vec, cat_id in queries:
            assert cat_id == 2

    def test_category_index_clamped(self):
        """Test category index is clamped to valid range."""
        pattern = HotCategoryLoop(category_index=10)  # > num_categories
        query_vectors = np.random.randn(10, 32).astype(np.float32)
        
        queries = list(pattern.generate_queries(query_vectors, num_categories=5, num_queries=10))
        
        # Should be clamped to max valid index (4)
        for vec, cat_id in queries:
            assert cat_id == 4

    def test_name_and_description(self):
        """Test pattern metadata."""
        pattern = HotCategoryLoop()
        
        assert pattern.name == "hot_category_loop"
        assert "single category" in pattern.description.lower()


class TestCategoryBatchSweep:
    """Tests for CategoryBatchSweep pattern."""

    def test_batch_structure(self):
        """Test queries are batched by category."""
        pattern = CategoryBatchSweep(queries_per_category=3)
        query_vectors = np.random.randn(20, 32).astype(np.float32)
        
        queries = list(pattern.generate_queries(query_vectors, num_categories=2, num_queries=12))
        
        assert len(queries) == 12
        
        # First 3 queries should be category 0
        for i in range(3):
            assert queries[i][1] == 0
        
        # Next 3 queries should be category 1
        for i in range(3, 6):
            assert queries[i][1] == 1
        
        # Next 3 should be category 0 again
        for i in range(6, 9):
            assert queries[i][1] == 0

    def test_partial_batch(self):
        """Test handling when num_queries < full cycle."""
        pattern = CategoryBatchSweep(queries_per_category=5)
        query_vectors = np.random.randn(10, 32).astype(np.float32)
        
        queries = list(pattern.generate_queries(query_vectors, num_categories=3, num_queries=7))
        
        assert len(queries) == 7


class TestInterleavedCategories:
    """Tests for InterleavedCategories pattern."""

    def test_round_robin(self):
        """Test round-robin category selection."""
        pattern = InterleavedCategories()
        query_vectors = np.random.randn(10, 32).astype(np.float32)
        
        queries = list(pattern.generate_queries(query_vectors, num_categories=3, num_queries=9))
        
        expected_categories = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        actual_categories = [cat_id for vec, cat_id in queries]
        
        assert actual_categories == expected_categories

    def test_name(self):
        """Test pattern name."""
        pattern = InterleavedCategories()
        assert pattern.name == "interleaved_categories"


class TestUniformRandomCategories:
    """Tests for UniformRandomCategories pattern."""

    def test_all_categories_used(self):
        """Test that all categories are eventually used."""
        pattern = UniformRandomCategories(seed=42)
        query_vectors = np.random.randn(10, 32).astype(np.float32)
        
        queries = list(pattern.generate_queries(query_vectors, num_categories=5, num_queries=1000))
        
        categories_used = set(cat_id for vec, cat_id in queries)
        
        # With 1000 queries and 5 categories, all should be used
        assert categories_used == {0, 1, 2, 3, 4}

    def test_reproducibility(self):
        """Test seed produces reproducible results."""
        query_vectors = np.random.randn(10, 32).astype(np.float32)
        
        pattern1 = UniformRandomCategories(seed=42)
        pattern2 = UniformRandomCategories(seed=42)
        
        queries1 = list(pattern1.generate_queries(query_vectors, num_categories=5, num_queries=100))
        queries2 = list(pattern2.generate_queries(query_vectors, num_categories=5, num_queries=100))
        
        cats1 = [c for v, c in queries1]
        cats2 = [c for v, c in queries2]
        
        assert cats1 == cats2


class TestZipfianDistribution:
    """Tests for ZipfianDistribution pattern."""

    def test_skewed_distribution(self):
        """Test that distribution is skewed toward early categories."""
        pattern = ZipfianDistribution(alpha=1.0, seed=42)
        query_vectors = np.random.randn(10, 32).astype(np.float32)
        
        queries = list(pattern.generate_queries(query_vectors, num_categories=10, num_queries=1000))
        
        category_counts = {}
        for vec, cat_id in queries:
            category_counts[cat_id] = category_counts.get(cat_id, 0) + 1
        
        # Category 0 should have more queries than category 9
        assert category_counts.get(0, 0) > category_counts.get(9, 0)

    def test_name_with_alpha(self):
        """Test pattern name includes alpha."""
        pattern = ZipfianDistribution(alpha=1.5)
        assert "1.5" in pattern.description


class TestPatternHelpers:
    """Tests for pattern helper functions."""

    def test_get_all_patterns(self):
        """Test getting all patterns."""
        config = Config.default()
        patterns = get_all_patterns(config)
        
        assert len(patterns) == 5
        
        pattern_names = [p.name for p in patterns]
        assert "hot_category_loop" in pattern_names
        assert "category_batch_sweep" in pattern_names
        assert "interleaved_categories" in pattern_names
        assert "uniform_random" in pattern_names
        assert "zipfian_distribution" in pattern_names

    def test_get_pattern_by_name(self):
        """Test getting specific pattern by name."""
        config = Config.default()
        
        pattern = get_pattern_by_name("hot_category_loop", config)
        assert isinstance(pattern, HotCategoryLoop)
        
        pattern = get_pattern_by_name("uniform_random", config)
        assert isinstance(pattern, UniformRandomCategories)

    def test_get_unknown_pattern(self):
        """Test error on unknown pattern name."""
        config = Config.default()
        
        with pytest.raises(ValueError) as exc_info:
            get_pattern_by_name("unknown_pattern", config)
        
        assert "Unknown pattern" in str(exc_info.value)
