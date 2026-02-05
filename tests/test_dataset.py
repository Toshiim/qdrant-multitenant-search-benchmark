"""Tests for dataset module."""

import numpy as np
import pytest

from benchmark.config import Config
from benchmark.dataset import (
    Dataset,
    assign_categories,
    batch_vectors,
    generate_synthetic_dataset,
    get_vectors_by_category,
)


class TestGenerateSyntheticDataset:
    """Tests for synthetic dataset generation."""

    def test_basic_generation(self):
        """Test basic synthetic dataset generation."""
        dataset = generate_synthetic_dataset(
            num_vectors=1000,
            dimensions=64,
            num_queries=100,
            distance="Cosine",
        )
        
        assert dataset.name == "synthetic-1000-64"
        assert dataset.vectors.shape == (1000, 64)
        assert dataset.queries.shape == (100, 64)
        assert dataset.dimensions == 64
        assert dataset.distance == "Cosine"

    def test_cosine_normalization(self):
        """Test vectors are normalized for cosine distance."""
        dataset = generate_synthetic_dataset(
            num_vectors=100,
            dimensions=32,
            distance="Cosine",
        )
        
        # Check that vectors are normalized (L2 norm ≈ 1)
        norms = np.linalg.norm(dataset.vectors, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(100), decimal=5)

    def test_euclidean_not_normalized(self):
        """Test vectors are not normalized for euclidean distance."""
        dataset = generate_synthetic_dataset(
            num_vectors=100,
            dimensions=32,
            distance="Euclidean",
        )
        
        # Vectors should NOT all be normalized
        norms = np.linalg.norm(dataset.vectors, axis=1)
        assert not np.allclose(norms, np.ones(100))

    def test_reproducibility(self):
        """Test seed produces reproducible results."""
        dataset1 = generate_synthetic_dataset(
            num_vectors=100,
            dimensions=32,
            seed=42,
        )
        dataset2 = generate_synthetic_dataset(
            num_vectors=100,
            dimensions=32,
            seed=42,
        )
        
        np.testing.assert_array_equal(dataset1.vectors, dataset2.vectors)


class TestAssignCategories:
    """Tests for category assignment."""

    def test_uniform_distribution(self):
        """Test uniform category distribution."""
        categories = assign_categories(
            num_vectors=10000,
            num_categories=10,
            distribution="uniform",
        )
        
        assert len(categories) == 10000
        assert categories.min() >= 0
        assert categories.max() < 10
        
        # Check roughly uniform distribution
        counts = np.bincount(categories)
        assert len(counts) == 10
        # Each category should have roughly 1000 vectors (±20%)
        for count in counts:
            assert 800 < count < 1200

    def test_zipfian_distribution(self):
        """Test zipfian category distribution."""
        categories = assign_categories(
            num_vectors=10000,
            num_categories=10,
            distribution="zipfian",
        )
        
        assert len(categories) == 10000
        assert categories.min() >= 0
        assert categories.max() < 10
        
        # Check that distribution is skewed (first categories have more)
        counts = np.bincount(categories, minlength=10)
        assert counts[0] > counts[-1]

    def test_reproducibility(self):
        """Test seed produces reproducible results."""
        cat1 = assign_categories(1000, 10, seed=42)
        cat2 = assign_categories(1000, 10, seed=42)
        
        np.testing.assert_array_equal(cat1, cat2)


class TestBatchVectors:
    """Tests for vector batching."""

    def test_basic_batching(self):
        """Test basic batch generation."""
        vectors = np.random.randn(100, 32).astype(np.float32)
        categories = np.random.randint(0, 5, size=100)
        
        batches = list(batch_vectors(vectors, categories, batch_size=25))
        
        assert len(batches) == 4  # 100 / 25 = 4 batches
        
        for batch_vecs, batch_ids, batch_cats in batches:
            assert len(batch_vecs) == 25
            assert len(batch_ids) == 25
            assert len(batch_cats) == 25

    def test_partial_last_batch(self):
        """Test handling of partial last batch."""
        vectors = np.random.randn(75, 32).astype(np.float32)
        categories = np.random.randint(0, 5, size=75)
        
        batches = list(batch_vectors(vectors, categories, batch_size=25))
        
        assert len(batches) == 3  # 75 / 25 = 3 batches
        
        # Check last batch is full
        assert len(batches[-1][0]) == 25


class TestGetVectorsByCategory:
    """Tests for category filtering."""

    def test_filter_by_category(self):
        """Test filtering vectors by category."""
        vectors = np.array([
            [1, 0],
            [2, 0],
            [3, 0],
            [4, 0],
            [5, 0],
        ], dtype=np.float32)
        categories = np.array([0, 1, 0, 1, 0])
        
        cat0_vectors, cat0_indices = get_vectors_by_category(vectors, categories, 0)
        
        assert len(cat0_vectors) == 3
        assert cat0_indices == [0, 2, 4]
        np.testing.assert_array_equal(
            cat0_vectors,
            np.array([[1, 0], [3, 0], [5, 0]], dtype=np.float32)
        )

    def test_empty_category(self):
        """Test filtering with no matches."""
        vectors = np.array([[1, 0], [2, 0]], dtype=np.float32)
        categories = np.array([0, 0])
        
        result_vectors, result_indices = get_vectors_by_category(vectors, categories, 5)
        
        assert len(result_vectors) == 0
        assert result_indices == []
