"""Query pattern implementations for benchmark tests."""

from abc import ABC, abstractmethod
from typing import Generator, List, Tuple

import numpy as np
from scipy.stats import zipfian


class QueryPattern(ABC):
    """Base class for query patterns."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return pattern name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Return pattern description."""
        pass

    @abstractmethod
    def generate_queries(
        self,
        query_vectors: np.ndarray,
        num_categories: int,
        num_queries: int,
    ) -> Generator[Tuple[np.ndarray, int], None, None]:
        """Generate (query_vector, category_id) pairs.
        
        Args:
            query_vectors: Pool of query vectors to use
            num_categories: Number of categories available
            num_queries: Number of queries to generate
        
        Yields:
            Tuples of (query_vector, category_id)
        """
        pass


class HotCategoryLoop(QueryPattern):
    """Hot Category Loop: All queries go to a single category."""

    def __init__(self, category_index: int = 0):
        self.category_index = category_index

    @property
    def name(self) -> str:
        return "hot_category_loop"

    @property
    def description(self) -> str:
        return "All queries target a single category (best-case locality)"

    def generate_queries(
        self,
        query_vectors: np.ndarray,
        num_categories: int,
        num_queries: int,
    ) -> Generator[Tuple[np.ndarray, int], None, None]:
        category = min(self.category_index, num_categories - 1)
        num_available = len(query_vectors)

        for i in range(num_queries):
            vector_idx = i % num_available
            yield query_vectors[vector_idx], category


class CategoryBatchSweep(QueryPattern):
    """Category Batch Sweep: Execute queries in blocks per category."""

    def __init__(self, queries_per_category: int = 100):
        self.queries_per_category = queries_per_category

    @property
    def name(self) -> str:
        return "category_batch_sweep"

    @property
    def description(self) -> str:
        return "Queries in sequential batches per category (partial cache warming)"

    def generate_queries(
        self,
        query_vectors: np.ndarray,
        num_categories: int,
        num_queries: int,
    ) -> Generator[Tuple[np.ndarray, int], None, None]:
        num_available = len(query_vectors)
        query_idx = 0
        generated = 0

        while generated < num_queries:
            for category in range(num_categories):
                for _ in range(self.queries_per_category):
                    if generated >= num_queries:
                        return
                    
                    vector_idx = query_idx % num_available
                    yield query_vectors[vector_idx], category
                    
                    query_idx += 1
                    generated += 1


class InterleavedCategories(QueryPattern):
    """Interleaved Categories: Round-robin across categories."""

    @property
    def name(self) -> str:
        return "interleaved_categories"

    @property
    def description(self) -> str:
        return "Round-robin category switching (worst-case locality)"

    def generate_queries(
        self,
        query_vectors: np.ndarray,
        num_categories: int,
        num_queries: int,
    ) -> Generator[Tuple[np.ndarray, int], None, None]:
        num_available = len(query_vectors)

        for i in range(num_queries):
            vector_idx = i % num_available
            category = i % num_categories
            yield query_vectors[vector_idx], category


class UniformRandomCategories(QueryPattern):
    """Uniform Random Categories: Random category selection."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    @property
    def name(self) -> str:
        return "uniform_random"

    @property
    def description(self) -> str:
        return "Uniformly random category selection (average case)"

    def generate_queries(
        self,
        query_vectors: np.ndarray,
        num_categories: int,
        num_queries: int,
    ) -> Generator[Tuple[np.ndarray, int], None, None]:
        np.random.seed(self.seed)
        num_available = len(query_vectors)

        categories = np.random.randint(0, num_categories, size=num_queries)
        
        for i in range(num_queries):
            vector_idx = i % num_available
            yield query_vectors[vector_idx], categories[i]


class ZipfianDistribution(QueryPattern):
    """Zipfian Distribution: Skewed distribution (hot/cold categories)."""

    def __init__(self, alpha: float = 1.0, seed: int = 42):
        self.alpha = alpha
        self.seed = seed

    @property
    def name(self) -> str:
        return "zipfian_distribution"

    @property
    def description(self) -> str:
        return f"Zipf distribution (alpha={self.alpha}, realistic hot/cold)"

    def generate_queries(
        self,
        query_vectors: np.ndarray,
        num_categories: int,
        num_queries: int,
    ) -> Generator[Tuple[np.ndarray, int], None, None]:
        np.random.seed(self.seed)
        num_available = len(query_vectors)

        # Generate Zipf-like distribution
        # Using manual probability calculation for stability
        ranks = np.arange(1, num_categories + 1)
        weights = 1.0 / (ranks ** self.alpha)
        probabilities = weights / weights.sum()

        categories = np.random.choice(
            num_categories,
            size=num_queries,
            p=probabilities,
        )

        for i in range(num_queries):
            vector_idx = i % num_available
            yield query_vectors[vector_idx], categories[i]


def get_all_patterns(config) -> List[QueryPattern]:
    """Get all query patterns with configuration."""
    return [
        HotCategoryLoop(category_index=config.query_patterns.hot_category.category_index),
        CategoryBatchSweep(queries_per_category=config.query_patterns.batch_sweep.queries_per_category),
        InterleavedCategories(),
        UniformRandomCategories(),
        ZipfianDistribution(alpha=config.query_patterns.zipfian.alpha),
    ]


def get_pattern_by_name(name: str, config) -> QueryPattern:
    """Get a specific query pattern by name."""
    patterns = {
        "hot_category_loop": HotCategoryLoop(
            category_index=config.query_patterns.hot_category.category_index
        ),
        "category_batch_sweep": CategoryBatchSweep(
            queries_per_category=config.query_patterns.batch_sweep.queries_per_category
        ),
        "interleaved_categories": InterleavedCategories(),
        "uniform_random": UniformRandomCategories(),
        "zipfian_distribution": ZipfianDistribution(
            alpha=config.query_patterns.zipfian.alpha
        ),
    }
    
    if name not in patterns:
        raise ValueError(f"Unknown pattern: {name}. Available: {list(patterns.keys())}")
    
    return patterns[name]
