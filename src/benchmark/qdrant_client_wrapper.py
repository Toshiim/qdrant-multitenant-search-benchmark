"""Qdrant client wrapper for benchmark scenarios."""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance,
    HnswConfigDiff,
    PointStruct,
    VectorParams,
)

from .config import Config


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

    def setup(self, dimensions: int, distance: str) -> float:
        """Create collection and return time taken."""
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

    def _collection_name(self, category_id: int) -> str:
        """Generate collection name for category."""
        return f"{self.COLLECTION_PREFIX}{category_id}"

    def setup(self, dimensions: int, distance: str, num_categories: int) -> float:
        """Create all collections and return time taken."""
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

        return time.perf_counter() - start

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

    def setup(self, dimensions: int, distance: str) -> float:
        """Create collection and return time taken."""
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
