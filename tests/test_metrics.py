"""Tests for metrics module."""

import numpy as np
import pytest

from benchmark.metrics import (
    LatencyMetrics,
    MetricsCollector,
    RecallMetrics,
    ThroughputMetrics,
    compute_recall,
)


class TestLatencyMetrics:
    """Tests for LatencyMetrics class."""

    def test_from_empty_list(self):
        """Test handling of empty duration list."""
        metrics = LatencyMetrics.from_durations([])
        
        assert metrics.count == 0
        assert metrics.total_seconds == 0
        assert metrics.mean_ms == 0
        assert metrics.p50_ms == 0

    def test_from_single_duration(self):
        """Test handling of single duration."""
        metrics = LatencyMetrics.from_durations([0.1])  # 100ms
        
        assert metrics.count == 1
        assert metrics.total_seconds == 0.1
        assert metrics.mean_ms == 100
        assert metrics.p50_ms == 100

    def test_from_multiple_durations(self):
        """Test calculation from multiple durations."""
        # 10, 20, 30, 40, 50 ms
        durations = [0.01, 0.02, 0.03, 0.04, 0.05]
        metrics = LatencyMetrics.from_durations(durations)
        
        assert metrics.count == 5
        assert abs(metrics.total_seconds - 0.15) < 0.001
        assert abs(metrics.mean_ms - 30) < 0.1
        assert metrics.min_ms == 10
        assert metrics.max_ms == 50

    def test_percentiles(self):
        """Test percentile calculations."""
        # Create durations with known distribution
        durations = [i / 1000 for i in range(1, 101)]  # 1-100ms
        metrics = LatencyMetrics.from_durations(durations)
        
        # P50 should be around 50ms
        assert 49 < metrics.p50_ms < 51
        # P95 should be around 95ms
        assert 94 < metrics.p95_ms < 96

    def test_to_dict(self):
        """Test dictionary conversion."""
        metrics = LatencyMetrics.from_durations([0.01, 0.02])
        result = metrics.to_dict()
        
        assert "count" in result
        assert "mean_ms" in result
        assert "p50_ms" in result
        assert "p95_ms" in result


class TestThroughputMetrics:
    """Tests for ThroughputMetrics class."""

    def test_basic_calculation(self):
        """Test basic throughput calculation."""
        metrics = ThroughputMetrics.from_operations(
            num_operations=100,
            total_seconds=10,
            num_vectors=1000,
        )
        
        assert metrics.total_operations == 100
        assert metrics.total_seconds == 10
        assert metrics.ops_per_second == 10  # 100 ops / 10 sec
        assert metrics.vectors_per_second == 100  # 1000 vecs / 10 sec

    def test_zero_time(self):
        """Test handling of zero time."""
        metrics = ThroughputMetrics.from_operations(
            num_operations=100,
            total_seconds=0,
        )
        
        assert metrics.ops_per_second == 0
        assert metrics.vectors_per_second == 0


class TestRecallMetrics:
    """Tests for RecallMetrics class."""

    def test_to_dict(self):
        """Test dictionary conversion."""
        metrics = RecallMetrics(recall_at_k=0.95, num_queries=100, k=10)
        result = metrics.to_dict()
        
        assert result["recall@10"] == 0.95
        assert result["num_queries"] == 100


class TestComputeRecall:
    """Tests for compute_recall function."""

    def test_perfect_recall(self):
        """Test when all results match ground truth."""
        results = [[0, 1, 2], [3, 4, 5]]
        ground_truth = np.array([[0, 1, 2], [3, 4, 5]])
        
        recall = compute_recall(results, ground_truth, k=3)
        
        assert recall.recall_at_k == 1.0
        assert recall.num_queries == 2

    def test_partial_recall(self):
        """Test partial recall."""
        results = [[0, 1, 9], [3, 4, 9]]  # 2/3 correct each
        ground_truth = np.array([[0, 1, 2], [3, 4, 5]])
        
        recall = compute_recall(results, ground_truth, k=3)
        
        # Each query has 2/3 recall, average is 2/3
        assert abs(recall.recall_at_k - 2/3) < 0.01

    def test_zero_recall(self):
        """Test when no results match."""
        results = [[10, 11, 12], [13, 14, 15]]
        ground_truth = np.array([[0, 1, 2], [3, 4, 5]])
        
        recall = compute_recall(results, ground_truth, k=3)
        
        assert recall.recall_at_k == 0.0

    def test_no_ground_truth(self):
        """Test handling of missing ground truth."""
        results = [[0, 1, 2]]
        
        recall = compute_recall(results, None, k=3)
        
        assert recall.recall_at_k == 0.0
        assert recall.num_queries == 0


class TestMetricsCollector:
    """Tests for MetricsCollector class."""

    def test_record_inserts(self):
        """Test recording insert operations."""
        collector = MetricsCollector()
        collector.start()
        
        collector.record_insert(0.01, 100)
        collector.record_insert(0.02, 100)
        
        metrics = collector.compute_insert_metrics()
        
        assert metrics.total_vectors == 200
        assert metrics.latency.count == 2

    def test_record_searches(self):
        """Test recording search operations."""
        collector = MetricsCollector()
        collector.start()
        
        collector.record_search(0.005, [1, 2, 3])
        collector.record_search(0.010, [4, 5, 6])
        
        metrics = collector.compute_search_metrics()
        
        assert metrics.latency.count == 2
        assert len(collector.search_results) == 2

    def test_reset(self):
        """Test resetting collector."""
        collector = MetricsCollector()
        collector.start()
        collector.record_insert(0.01, 100)
        collector.record_search(0.005, [1, 2, 3])
        
        collector.reset()
        
        assert len(collector.insert_durations) == 0
        assert len(collector.search_durations) == 0
        assert len(collector.search_results) == 0
