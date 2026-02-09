"""Qdrant client wrapper for benchmark scenarios."""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    PointStruct,
    VectorParams,
)

from .config import Config


# Index cache reset thresholds
# These values are chosen to trigger Qdrant's optimizer to rebuild internal structures:
# - TEMP_INDEXING_THRESHOLD: Lower value to trigger immediate re-indexing
# - DEFAULT_INDEXING_THRESHOLD: Qdrant's default value to restore normal behavior
# Changing the indexing_threshold forces the optimizer to re-evaluate segments,
# which effectively clears cached HNSW graph data and resets search to "cold" state.
TEMP_INDEXING_THRESHOLD = 10000
DEFAULT_INDEXING_THRESHOLD = 20000


def get_distance_enum(distance_str: str) -> Distance:
    """Convert distance string to Qdrant Distance enum."""
    distance_map = {
        "cosine": Distance.COSINE,
        "euclidean": Distance.EUCLID,
        "dot": Distance.DOT,
    }
    return distance_map.get(distance_str.lower(), Distance.COSINE)


@dataclass
class InsertResult:
    """Result of an insert operation."""

    num_vectors: int
    duration_seconds: float
    throughput: float  # vectors/sec


@dataclass
class SearchResult:
    """Result of a search operation."""

    query_id: int
    category_id: int
    duration_seconds: float
    results: List[int]  # Point IDs


class ScenarioA:
    """Scenario A: Single collection with payload filtering."""

    COLLECTION_NAME = "benchmark_single"
    PAYLOAD_KEY = "category_id"

    def __init__(self, config: Config):
        self.config = config
        self.client = QdrantClient(
            host=config.qdrant.host,
            port=config.qdrant.port,
            timeout=config.qdrant.timeout,
        )
        self._collection_created = False

    def collection_exists(self) -> bool:
        """Check if collection exists and has data."""
        try:
            info = self.client.get_collection(self.COLLECTION_NAME)
            return info.points_count is not None and info.points_count > 0
        except Exception:
            return False

    def setup(self, dimensions: int, distance: str, force_recreate: bool = True) -> float:
        """Create collection and return time taken.
        
        Args:
            dimensions: Vector dimensions
            distance: Distance metric
            force_recreate: If False and collection exists, skip creation
        
        Returns:
            Time taken to create collection (0 if skipped)
        """
        # If not forcing recreate and collection exists, skip
        if not force_recreate and self.collection_exists():
            self._collection_created = True
            return 0.0

        start = time.perf_counter()

        # Delete if exists
        try:
            self.client.delete_collection(self.COLLECTION_NAME)
        except Exception:
            pass

        # Create collection
        self.client.create_collection(
            collection_name=self.COLLECTION_NAME,
            vectors_config=VectorParams(
                size=dimensions,
                distance=get_distance_enum(distance),
            ),
            hnsw_config=HnswConfigDiff(
                m=self.config.hnsw.m,
                ef_construct=self.config.hnsw.ef_construct,
            ),
        )

        # Create payload index for filtering
        self.client.create_payload_index(
            collection_name=self.COLLECTION_NAME,
            field_name=self.PAYLOAD_KEY,
            field_schema=models.PayloadSchemaType.INTEGER,
        )

        self._collection_created = True
        return time.perf_counter() - start

    def reset_index_cache(self) -> float:
        """Reset HNSW index to cold state by triggering re-optimization.
        
        This forces Qdrant to rebuild internal caches and reset the index
        to a "cold" state without re-uploading data.
        
        Returns:
            Time taken for the operation in seconds
        """
        start = time.perf_counter()
        
        # Update optimizers config to force index refresh
        self.client.update_collection(
            collection_name=self.COLLECTION_NAME,
            optimizer_config=OptimizersConfigDiff(
                indexing_threshold=TEMP_INDEXING_THRESHOLD,
            ),
        )
        
        # Wait for optimization to complete
        self._wait_for_green_status()
        
        # Restore original settings
        self.client.update_collection(
            collection_name=self.COLLECTION_NAME,
            optimizer_config=OptimizersConfigDiff(
                indexing_threshold=DEFAULT_INDEXING_THRESHOLD,
            ),
        )
        
        self._wait_for_green_status()
        
        return time.perf_counter() - start

    def _wait_for_green_status(self, timeout: int = 300):
        """Wait for collection to be in green/ready status."""
        start = time.time()
        while time.time() - start < timeout:
            info = self.client.get_collection(self.COLLECTION_NAME)
            if info.status == models.CollectionStatus.GREEN:
                return
            time.sleep(0.5)
        raise TimeoutError(f"Collection {self.COLLECTION_NAME} did not reach green status within {timeout}s")

    def insert(
        self,
        vectors: List[List[float]],
        ids: List[int],
        category_ids: List[int],
    ) -> InsertResult:
        """Insert vectors with category payload."""
        start = time.perf_counter()

        points = [
            PointStruct(
                id=id_,
                vector=vector,
                payload={self.PAYLOAD_KEY: cat_id},
            )
            for id_, vector, cat_id in zip(ids, vectors, category_ids)
        ]

        self.client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=points,
            wait=True,
        )

        duration = time.perf_counter() - start
        return InsertResult(
            num_vectors=len(vectors),
            duration_seconds=duration,
            throughput=len(vectors) / duration if duration > 0 else 0,
        )

    def search(
        self,
        query_vector: List[float],
        category_id: int,
        query_id: int = 0,
    ) -> SearchResult:
        """Search with category filter."""
        start = time.perf_counter()

        response = self.client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=query_vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key=self.PAYLOAD_KEY,
                        match=models.MatchValue(value=category_id),
                    )
                ]
            ),
            limit=self.config.search.top_k,
            search_params=models.SearchParams(
                hnsw_ef=self.config.hnsw.ef_search,
            ),
        )

        duration = time.perf_counter() - start
        return SearchResult(
            query_id=query_id,
            category_id=category_id,
            duration_seconds=duration,
            results=[point.id for point in response.points],
        )

    def get_collection_info(self) -> Dict:
        """Get collection statistics."""
        info = self.client.get_collection(self.COLLECTION_NAME)
        return {
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": str(info.status),
            "segments_count": len(info.segments or []),
        }

    def cleanup(self):
        """Delete collection."""
        try:
            self.client.delete_collection(self.COLLECTION_NAME)
        except Exception:
            pass


class ScenarioB:
    """Scenario B: Multiple collections (one per category)."""

    COLLECTION_PREFIX = "benchmark_cat_"

    def __init__(self, config: Config):
        self.config = config
        self.client = QdrantClient(
            host=config.qdrant.host,
            port=config.qdrant.port,
            timeout=config.qdrant.timeout,
        )
        self._collections: Dict[int, str] = {}
        self._num_categories: int = 0

    def _collection_name(self, category_id: int) -> str:
        """Generate collection name for category."""
        return f"{self.COLLECTION_PREFIX}{category_id}"

    def collections_exist(self, num_categories: int) -> bool:
        """Check if all collections exist and have data."""
        try:
            for cat_id in range(num_categories):
                name = self._collection_name(cat_id)
                info = self.client.get_collection(name)
                if info.points_count is None or info.points_count == 0:
                    return False
                self._collections[cat_id] = name
            self._num_categories = num_categories
            return True
        except Exception:
            return False

    def setup(self, dimensions: int, distance: str, num_categories: int, force_recreate: bool = True) -> float:
        """Create all collections and return time taken.
        
        Args:
            dimensions: Vector dimensions
            distance: Distance metric
            num_categories: Number of categories/collections
            force_recreate: If False and collections exist, skip creation
        
        Returns:
            Time taken to create collections (0 if skipped)
        """
        # If not forcing recreate and collections exist, skip
        if not force_recreate and self.collections_exist(num_categories):
            return 0.0

        start = time.perf_counter()

        # Delete existing collections
        for cat_id in range(num_categories):
            name = self._collection_name(cat_id)
            try:
                self.client.delete_collection(name)
            except Exception:
                pass

        # Create collections for each category
        for cat_id in range(num_categories):
            name = self._collection_name(cat_id)
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=dimensions,
                    distance=get_distance_enum(distance),
                ),
                hnsw_config=HnswConfigDiff(
                    m=self.config.hnsw.m,
                    ef_construct=self.config.hnsw.ef_construct,
                ),
            )
            self._collections[cat_id] = name

        self._num_categories = num_categories
        return time.perf_counter() - start

    def reset_index_cache(self) -> float:
        """Reset HNSW index to cold state for all collections.
        
        Returns:
            Time taken for the operation in seconds
        """
        start = time.perf_counter()
        
        for cat_id in self._collections:
            name = self._collection_name(cat_id)
            self.client.update_collection(
                collection_name=name,
                optimizer_config=OptimizersConfigDiff(
                    indexing_threshold=TEMP_INDEXING_THRESHOLD,
                ),
            )
        
        # Wait for all collections to be ready
        for cat_id in self._collections:
            self._wait_for_green_status(cat_id)
        
        # Restore original settings
        for cat_id in self._collections:
            name = self._collection_name(cat_id)
            self.client.update_collection(
                collection_name=name,
                optimizer_config=OptimizersConfigDiff(
                    indexing_threshold=DEFAULT_INDEXING_THRESHOLD,
                ),
            )
        
        for cat_id in self._collections:
            self._wait_for_green_status(cat_id)
        
        return time.perf_counter() - start

    def _wait_for_green_status(self, category_id: int, timeout: int = 300):
        """Wait for collection to be in green/ready status."""
        name = self._collection_name(category_id)
        start = time.time()
        while time.time() - start < timeout:
            info = self.client.get_collection(name)
            if info.status == models.CollectionStatus.GREEN:
                return
            time.sleep(0.5)
        raise TimeoutError(f"Collection {name} did not reach green status within {timeout}s")

    def insert(
        self,
        vectors: List[List[float]],
        ids: List[int],
        category_id: int,
    ) -> InsertResult:
        """Insert vectors into specific category collection."""
        start = time.perf_counter()

        name = self._collection_name(category_id)
        points = [
            PointStruct(id=id_, vector=vector)
            for id_, vector in zip(ids, vectors)
        ]

        self.client.upsert(
            collection_name=name,
            points=points,
            wait=True,
        )

        duration = time.perf_counter() - start
        return InsertResult(
            num_vectors=len(vectors),
            duration_seconds=duration,
            throughput=len(vectors) / duration if duration > 0 else 0,
        )

    def search(
        self,
        query_vector: List[float],
        category_id: int,
        query_id: int = 0,
    ) -> SearchResult:
        """Search in specific category collection."""
        start = time.perf_counter()

        name = self._collection_name(category_id)
        response = self.client.query_points(
            collection_name=name,
            query=query_vector,
            limit=self.config.search.top_k,
            search_params=models.SearchParams(
                hnsw_ef=self.config.hnsw.ef_search,
            ),
        )

        duration = time.perf_counter() - start
        return SearchResult(
            query_id=query_id,
            category_id=category_id,
            duration_seconds=duration,
            results=[point.id for point in response.points],
        )

    def get_collection_info(self, category_id: int) -> Dict:
        """Get collection statistics for a category."""
        name = self._collection_name(category_id)
        info = self.client.get_collection(name)
        return {
            "category_id": category_id,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": str(info.status),
        }

    def get_all_collections_info(self) -> List[Dict]:
        """Get statistics for all collections."""
        return [
            self.get_collection_info(cat_id)
            for cat_id in sorted(self._collections.keys())
        ]

    def cleanup(self):
        """Delete all collections."""
        for cat_id in list(self._collections.keys()):
            try:
                self.client.delete_collection(self._collection_name(cat_id))
            except Exception:
                pass
        self._collections.clear()


class BaselineScenario:
    """Baseline: Single collection without payload filtering."""

    COLLECTION_NAME = "benchmark_baseline"

    def __init__(self, config: Config):
        self.config = config
        self.client = QdrantClient(
            host=config.qdrant.host,
            port=config.qdrant.port,
            timeout=config.qdrant.timeout,
        )

    def collection_exists(self) -> bool:
        """Check if collection exists and has data."""
        try:
            info = self.client.get_collection(self.COLLECTION_NAME)
            return info.points_count is not None and info.points_count > 0
        except Exception:
            return False

    def setup(self, dimensions: int, distance: str, force_recreate: bool = True) -> float:
        """Create collection and return time taken.
        
        Args:
            dimensions: Vector dimensions
            distance: Distance metric
            force_recreate: If False and collection exists, skip creation
        
        Returns:
            Time taken to create collection (0 if skipped)
        """
        # If not forcing recreate and collection exists, skip
        if not force_recreate and self.collection_exists():
            return 0.0

        start = time.perf_counter()

        try:
            self.client.delete_collection(self.COLLECTION_NAME)
        except Exception:
            pass

        self.client.create_collection(
            collection_name=self.COLLECTION_NAME,
            vectors_config=VectorParams(
                size=dimensions,
                distance=get_distance_enum(distance),
            ),
            hnsw_config=HnswConfigDiff(
                m=self.config.hnsw.m,
                ef_construct=self.config.hnsw.ef_construct,
            ),
        )

        return time.perf_counter() - start

    def reset_index_cache(self) -> float:
        """Reset HNSW index to cold state by triggering re-optimization.
        
        Returns:
            Time taken for the operation in seconds
        """
        start = time.perf_counter()
        
        self.client.update_collection(
            collection_name=self.COLLECTION_NAME,
            optimizer_config=OptimizersConfigDiff(
                indexing_threshold=TEMP_INDEXING_THRESHOLD,
            ),
        )
        
        self._wait_for_green_status()
        
        self.client.update_collection(
            collection_name=self.COLLECTION_NAME,
            optimizer_config=OptimizersConfigDiff(
                indexing_threshold=DEFAULT_INDEXING_THRESHOLD,
            ),
        )
        
        self._wait_for_green_status()
        
        return time.perf_counter() - start

    def _wait_for_green_status(self, timeout: int = 300):
        """Wait for collection to be in green/ready status."""
        start = time.time()
        while time.time() - start < timeout:
            info = self.client.get_collection(self.COLLECTION_NAME)
            if info.status == models.CollectionStatus.GREEN:
                return
            time.sleep(0.5)
        raise TimeoutError(f"Collection {self.COLLECTION_NAME} did not reach green status within {timeout}s")

    def insert(
        self,
        vectors: List[List[float]],
        ids: List[int],
    ) -> InsertResult:
        """Insert vectors without payload."""
        start = time.perf_counter()

        points = [
            PointStruct(id=id_, vector=vector)
            for id_, vector in zip(ids, vectors)
        ]

        self.client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=points,
            wait=True,
        )

        duration = time.perf_counter() - start
        return InsertResult(
            num_vectors=len(vectors),
            duration_seconds=duration,
            throughput=len(vectors) / duration if duration > 0 else 0,
        )

    def search(
        self,
        query_vector: List[float],
        query_id: int = 0,
    ) -> SearchResult:
        """Search without filtering."""
        start = time.perf_counter()

        response = self.client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=query_vector,
            limit=self.config.search.top_k,
            search_params=models.SearchParams(
                hnsw_ef=self.config.hnsw.ef_search,
            ),
        )

        duration = time.perf_counter() - start
        return SearchResult(
            query_id=query_id,
            category_id=-1,  # No category for baseline
            duration_seconds=duration,
            results=[point.id for point in response.points],
        )

    def get_collection_info(self) -> Dict:
        """Get collection statistics."""
        info = self.client.get_collection(self.COLLECTION_NAME)
        return {
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": str(info.status),
        }

    def cleanup(self):
        """Delete collection."""
        try:
            self.client.delete_collection(self.COLLECTION_NAME)
        except Exception:
            pass
