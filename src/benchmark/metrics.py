"""Metrics collection and computation utilities."""

import statistics
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import psutil


@dataclass
class LatencyMetrics:
    """Latency statistics."""

    count: int
    total_seconds: float
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float

    @classmethod
    def from_durations(cls, durations: List[float]) -> "LatencyMetrics":
        """Compute latency metrics from list of durations in seconds."""
        if not durations:
            return cls(
                count=0,
                total_seconds=0,
                mean_ms=0,
                p50_ms=0,
                p95_ms=0,
                p99_ms=0,
                min_ms=0,
                max_ms=0,
            )

        durations_ms = [d * 1000 for d in durations]
        sorted_ms = sorted(durations_ms)

        return cls(
            count=len(durations),
            total_seconds=sum(durations),
            mean_ms=statistics.mean(durations_ms),
            p50_ms=np.percentile(sorted_ms, 50),
            p95_ms=np.percentile(sorted_ms, 95),
            p99_ms=np.percentile(sorted_ms, 99),
            min_ms=min(durations_ms),
            max_ms=max(durations_ms),
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "count": self.count,
            "total_seconds": round(self.total_seconds, 3),
            "mean_ms": round(self.mean_ms, 3),
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
            "min_ms": round(self.min_ms, 3),
            "max_ms": round(self.max_ms, 3),
        }


@dataclass
class ThroughputMetrics:
    """Throughput statistics."""

    total_operations: int
    total_seconds: float
    ops_per_second: float
    vectors_per_second: float  # For insert operations

    @classmethod
    def from_operations(
        cls,
        num_operations: int,
        total_seconds: float,
        num_vectors: int = 0,
    ) -> "ThroughputMetrics":
        """Compute throughput metrics."""
        ops_per_sec = num_operations / total_seconds if total_seconds > 0 else 0
        vectors_per_sec = num_vectors / total_seconds if total_seconds > 0 else 0

        return cls(
            total_operations=num_operations,
            total_seconds=total_seconds,
            ops_per_second=ops_per_sec,
            vectors_per_second=vectors_per_sec,
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "total_operations": self.total_operations,
            "total_seconds": round(self.total_seconds, 3),
            "ops_per_second": round(self.ops_per_second, 2),
            "vectors_per_second": round(self.vectors_per_second, 2),
        }


@dataclass
class RecallMetrics:
    """Recall statistics for search quality."""

    recall_at_k: float
    num_queries: int
    k: int

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            f"recall@{self.k}": round(self.recall_at_k, 4),
            "num_queries": self.num_queries,
            "k": self.k,
        }


@dataclass
class ResourceMetrics:
    """Resource usage metrics."""

    ram_mb: float
    ram_percent: float
    cpu_percent: float
    timestamp: float

    @classmethod
    def current(cls) -> "ResourceMetrics":
        """Get current resource usage."""
        process = psutil.Process()
        mem_info = process.memory_info()

        return cls(
            ram_mb=mem_info.rss / (1024 * 1024),
            ram_percent=process.memory_percent(),
            cpu_percent=process.cpu_percent(interval=0.1),
            timestamp=time.time(),
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "ram_mb": round(self.ram_mb, 2),
            "ram_percent": round(self.ram_percent, 2),
            "cpu_percent": round(self.cpu_percent, 2),
        }


@dataclass
class InsertMetrics:
    """Combined insert operation metrics."""

    latency: LatencyMetrics
    throughput: ThroughputMetrics
    total_vectors: int
    index_build_time_seconds: float

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "latency": self.latency.to_dict(),
            "throughput": self.throughput.to_dict(),
            "total_vectors": self.total_vectors,
            "index_build_time_seconds": round(self.index_build_time_seconds, 3),
        }


@dataclass
class SearchMetrics:
    """Combined search operation metrics."""

    latency: LatencyMetrics
    qps: float  # Queries per second
    recall: Optional[RecallMetrics] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = {
            "latency": self.latency.to_dict(),
            "qps": round(self.qps, 2),
        }
        if self.recall:
            result["recall"] = self.recall.to_dict()
        return result


@dataclass
class TestMetrics:
    """Metrics for a single test run."""

    test_name: str
    scenario: str  # "A", "B", or "baseline"
    num_categories: int
    insert_metrics: InsertMetrics
    search_metrics: SearchMetrics
    resource_metrics: ResourceMetrics
    setup_time_seconds: float
    total_time_seconds: float

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "test_name": self.test_name,
            "scenario": self.scenario,
            "num_categories": self.num_categories,
            "insert": self.insert_metrics.to_dict(),
            "search": self.search_metrics.to_dict(),
            "resources": self.resource_metrics.to_dict(),
            "setup_time_seconds": round(self.setup_time_seconds, 3),
            "total_time_seconds": round(self.total_time_seconds, 3),
        }


def compute_recall(
    results: List[List[int]],
    ground_truth: np.ndarray,
    k: int,
) -> RecallMetrics:
    """Compute recall@k against ground truth.
    
    Args:
        results: List of result ID lists from search
        ground_truth: Array of true neighbor IDs for each query
        k: Number of results to consider
    
    Returns:
        RecallMetrics with computed recall
    """
    if ground_truth is None or len(results) == 0:
        return RecallMetrics(recall_at_k=0.0, num_queries=0, k=k)

    total_recall = 0.0
    num_queries = min(len(results), len(ground_truth))

    for i in range(num_queries):
        result_set = set(results[i][:k])
        truth_set = set(ground_truth[i][:k])

        if len(truth_set) > 0:
            intersection = len(result_set & truth_set)
            total_recall += intersection / len(truth_set)

    avg_recall = total_recall / num_queries if num_queries > 0 else 0.0

    return RecallMetrics(
        recall_at_k=avg_recall,
        num_queries=num_queries,
        k=k,
    )


class MetricsCollector:
    """Collect and aggregate metrics during benchmark execution."""

    def __init__(self):
        self.insert_durations: List[float] = []
        self.search_durations: List[float] = []
        self.search_results: List[List[int]] = []
        self.resource_samples: List[ResourceMetrics] = []
        self._start_time: Optional[float] = None
        self._vectors_inserted = 0

    def start(self):
        """Mark the start of metric collection."""
        self._start_time = time.perf_counter()

    def record_insert(self, duration: float, num_vectors: int):
        """Record an insert operation."""
        self.insert_durations.append(duration)
        self._vectors_inserted += num_vectors

    def record_search(self, duration: float, results: List[int]):
        """Record a search operation."""
        self.search_durations.append(duration)
        self.search_results.append(results)

    def sample_resources(self):
        """Take a resource usage sample."""
        self.resource_samples.append(ResourceMetrics.current())

    def compute_insert_metrics(self, index_build_time: float = 0.0) -> InsertMetrics:
        """Compute insert metrics from collected data."""
        latency = LatencyMetrics.from_durations(self.insert_durations)
        throughput = ThroughputMetrics.from_operations(
            num_operations=len(self.insert_durations),
            total_seconds=latency.total_seconds,
            num_vectors=self._vectors_inserted,
        )

        return InsertMetrics(
            latency=latency,
            throughput=throughput,
            total_vectors=self._vectors_inserted,
            index_build_time_seconds=index_build_time,
        )

    def compute_search_metrics(
        self,
        ground_truth: Optional[np.ndarray] = None,
        k: int = 10,
    ) -> SearchMetrics:
        """Compute search metrics from collected data."""
        latency = LatencyMetrics.from_durations(self.search_durations)
        qps = latency.count / latency.total_seconds if latency.total_seconds > 0 else 0

        recall = None
        if ground_truth is not None:
            recall = compute_recall(self.search_results, ground_truth, k)

        return SearchMetrics(
            latency=latency,
            qps=qps,
            recall=recall,
        )

    def get_latest_resource_metrics(self) -> ResourceMetrics:
        """Get most recent resource sample or current if none."""
        if self.resource_samples:
            return self.resource_samples[-1]
        return ResourceMetrics.current()

    def get_elapsed_time(self) -> float:
        """Get elapsed time since start."""
        if self._start_time is None:
            return 0.0
        return time.perf_counter() - self._start_time

    def reset(self):
        """Reset all collected metrics."""
        self.insert_durations.clear()
        self.search_durations.clear()
        self.search_results.clear()
        self.resource_samples.clear()
        self._start_time = None
        self._vectors_inserted = 0
