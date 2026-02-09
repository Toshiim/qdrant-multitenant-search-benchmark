"""Benchmark runner - orchestrates test execution."""

import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

from .config import Config
from .dataset import Dataset, assign_categories, batch_vectors
from .metrics import MetricsCollector, TestMetrics, InsertMetrics, SearchMetrics, ResourceMetrics, LatencyMetrics, ThroughputMetrics
from .qdrant_client_wrapper import BaselineScenario, ScenarioA, ScenarioB
from .query_patterns import QueryPattern, get_all_patterns


@dataclass
class BenchmarkResult:
    """Result of a complete benchmark run."""

    config: Dict
    results: List[TestMetrics]
    timestamp: str

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "config": self.config,
            "timestamp": self.timestamp,
            "results": [r.to_dict() for r in self.results],
        }

    def save(self, path: str):
        """Save results to JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class BenchmarkRunner:
    """Main benchmark runner."""

    def __init__(self, config: Config, dataset: Dataset):
        self.config = config
        self.dataset = dataset
        self.results: List[TestMetrics] = []

    def _insert_data_scenario_a(
        self,
        scenario: ScenarioA,
        vectors: np.ndarray,
        category_ids: np.ndarray,
        collector: MetricsCollector,
    ) -> float:
        """Insert data for Scenario A (single collection)."""
        setup_time = scenario.setup(self.dataset.dimensions, self.dataset.distance)
        
        for batch_vecs, batch_ids, batch_cats in tqdm(
            batch_vectors(vectors, category_ids, self.config.benchmark.batch_size),
            desc="Inserting (Scenario A)",
            total=(len(vectors) // self.config.benchmark.batch_size) + 1,
        ):
            result = scenario.insert(
                vectors=batch_vecs.tolist(),
                ids=batch_ids,
                category_ids=batch_cats,
            )
            collector.record_insert(result.duration_seconds, result.num_vectors)

        return setup_time

    def _insert_data_scenario_b(
        self,
        scenario: ScenarioB,
        vectors: np.ndarray,
        category_ids: np.ndarray,
        num_categories: int,
        collector: MetricsCollector,
    ) -> float:
        """Insert data for Scenario B (multiple collections)."""
        setup_time = scenario.setup(
            self.dataset.dimensions,
            self.dataset.distance,
            num_categories,
        )

        # Group vectors by category
        for cat_id in tqdm(range(num_categories), desc="Inserting (Scenario B)"):
            mask = category_ids == cat_id
            cat_vectors = vectors[mask]
            cat_ids = np.where(mask)[0].tolist()

            if len(cat_vectors) == 0:
                continue

            # Insert in batches
            for start in range(0, len(cat_vectors), self.config.benchmark.batch_size):
                end = min(start + self.config.benchmark.batch_size, len(cat_vectors))
                batch_vecs = cat_vectors[start:end]
                batch_ids = cat_ids[start:end]

                result = scenario.insert(
                    vectors=batch_vecs.tolist(),
                    ids=batch_ids,
                    category_id=cat_id,
                )
                collector.record_insert(result.duration_seconds, result.num_vectors)

        return setup_time

    def _insert_data_baseline(
        self,
        scenario: BaselineScenario,
        vectors: np.ndarray,
        collector: MetricsCollector,
    ) -> float:
        """Insert data for baseline scenario."""
        setup_time = scenario.setup(self.dataset.dimensions, self.dataset.distance)

        ids = list(range(len(vectors)))
        for start in tqdm(
            range(0, len(vectors), self.config.benchmark.batch_size),
            desc="Inserting (Baseline)",
        ):
            end = min(start + self.config.benchmark.batch_size, len(vectors))
            batch_vecs = vectors[start:end]
            batch_ids = ids[start:end]

            result = scenario.insert(
                vectors=batch_vecs.tolist(),
                ids=batch_ids,
            )
            collector.record_insert(result.duration_seconds, result.num_vectors)

        return setup_time

    def _run_warmup(
        self,
        scenario,
        num_categories: int,
    ) -> float:
        """Run warmup queries to warm up indexes before search tests.
        
        Returns:
            Duration of warmup in seconds.
        """
        print(f"  Running warmup ({self.config.benchmark.warmup_queries} queries)...")
        start_time = time.perf_counter()
        
        # Use uniform distribution for warmup to touch all categories
        from .query_patterns import UniformRandomCategories
        warmup_pattern = UniformRandomCategories(seed=0)
        
        warmup_gen = warmup_pattern.generate_queries(
            self.dataset.queries,
            num_categories,
            self.config.benchmark.warmup_queries,
        )
        
        for query_vec, cat_id in tqdm(
            warmup_gen,
            desc="Warmup",
            total=self.config.benchmark.warmup_queries,
        ):
            if isinstance(scenario, BaselineScenario):
                scenario.search(query_vec.tolist())
            else:
                scenario.search(query_vec.tolist(), cat_id)
        
        duration = time.perf_counter() - start_time
        print(f"  Warmup completed in {duration:.2f}s")
        return duration

    def _run_search_pattern(
        self,
        scenario,
        pattern: QueryPattern,
        num_categories: int,
        collector: MetricsCollector,
    ):
        """Run search queries according to pattern (without warmup)."""
        # Actual measurement
        query_gen = pattern.generate_queries(
            self.dataset.queries,
            num_categories,
            self.config.benchmark.num_queries,
        )

        for i, (query_vec, cat_id) in enumerate(tqdm(
            query_gen,
            desc=f"Searching ({pattern.name})",
            total=self.config.benchmark.num_queries,
        )):
            if isinstance(scenario, BaselineScenario):
                result = scenario.search(query_vec.tolist(), query_id=i)
            else:
                result = scenario.search(query_vec.tolist(), cat_id, query_id=i)
            
            collector.record_search(result.duration_seconds, result.results)

    def run_scenario_a(
        self,
        pattern: QueryPattern,
        num_categories: int,
        category_ids: np.ndarray,
    ) -> TestMetrics:
        """Run benchmark for Scenario A (legacy method for backwards compatibility)."""
        print(f"\n=== Running Scenario A: {pattern.name} ({num_categories} categories) ===")
        
        scenario = ScenarioA(self.config)
        collector = MetricsCollector()
        collector.start()

        try:
            # Insert phase
            setup_time = self._insert_data_scenario_a(
                scenario,
                self.dataset.vectors,
                category_ids,
                collector,
            )
            index_build_time = setup_time
            insert_metrics = collector.compute_insert_metrics(index_build_time)

            # Warmup phase
            self._run_warmup(scenario, num_categories)

            # Search phase
            collector.reset()
            collector.start()
            self._run_search_pattern(scenario, pattern, num_categories, collector)
            
            search_metrics = collector.compute_search_metrics(
                self.dataset.neighbors,
                self.config.search.top_k,
            )

            resource_metrics = collector.get_latest_resource_metrics()
            total_time = collector.get_elapsed_time()

            return TestMetrics(
                test_name=pattern.name,
                scenario="A",
                num_categories=num_categories,
                insert_metrics=insert_metrics,
                search_metrics=search_metrics,
                resource_metrics=resource_metrics,
                setup_time_seconds=setup_time,
                total_time_seconds=total_time,
            )
        finally:
            scenario.cleanup()

    def run_scenario_b(
        self,
        pattern: QueryPattern,
        num_categories: int,
        category_ids: np.ndarray,
    ) -> TestMetrics:
        """Run benchmark for Scenario B (legacy method for backwards compatibility)."""
        print(f"\n=== Running Scenario B: {pattern.name} ({num_categories} categories) ===")
        
        scenario = ScenarioB(self.config)
        collector = MetricsCollector()
        collector.start()

        try:
            # Insert phase
            setup_time = self._insert_data_scenario_b(
                scenario,
                self.dataset.vectors,
                category_ids,
                num_categories,
                collector,
            )
            index_build_time = setup_time
            insert_metrics = collector.compute_insert_metrics(index_build_time)

            # Warmup phase
            self._run_warmup(scenario, num_categories)

            # Search phase
            collector.reset()
            collector.start()
            self._run_search_pattern(scenario, pattern, num_categories, collector)
            
            search_metrics = collector.compute_search_metrics(
                self.dataset.neighbors,
                self.config.search.top_k,
            )

            resource_metrics = collector.get_latest_resource_metrics()
            total_time = collector.get_elapsed_time()

            return TestMetrics(
                test_name=pattern.name,
                scenario="B",
                num_categories=num_categories,
                insert_metrics=insert_metrics,
                search_metrics=search_metrics,
                resource_metrics=resource_metrics,
                setup_time_seconds=setup_time,
                total_time_seconds=total_time,
            )
        finally:
            scenario.cleanup()

    def run_baseline(self, pattern: QueryPattern) -> TestMetrics:
        """Run baseline benchmark (no categories)."""
        print(f"\n=== Running Baseline: {pattern.name} ===")
        
        scenario = BaselineScenario(self.config)
        collector = MetricsCollector()
        collector.start()

        try:
            # Insert phase
            setup_time = self._insert_data_baseline(
                scenario,
                self.dataset.vectors,
                collector,
            )
            index_build_time = setup_time
            insert_metrics = collector.compute_insert_metrics(index_build_time)

            # Warmup phase
            self._run_warmup(scenario, 1)

            # Search phase
            collector.reset()
            collector.start()
            self._run_search_pattern(scenario, pattern, 1, collector)
            
            search_metrics = collector.compute_search_metrics(
                self.dataset.neighbors,
                self.config.search.top_k,
            )

            resource_metrics = collector.get_latest_resource_metrics()
            total_time = collector.get_elapsed_time()

            return TestMetrics(
                test_name=pattern.name,
                scenario="baseline",
                num_categories=0,
                insert_metrics=insert_metrics,
                search_metrics=search_metrics,
                resource_metrics=resource_metrics,
                setup_time_seconds=setup_time,
                total_time_seconds=total_time,
            )
        finally:
            scenario.cleanup()

    def _create_empty_insert_metrics(self) -> InsertMetrics:
        """Create empty insert metrics for search-only tests."""
        return InsertMetrics(
            latency=LatencyMetrics(
                count=0, total_seconds=0, mean_ms=0, p50_ms=0,
                p95_ms=0, p99_ms=0, min_ms=0, max_ms=0
            ),
            throughput=ThroughputMetrics(
                total_operations=0, total_seconds=0,
                ops_per_second=0, vectors_per_second=0
            ),
            total_vectors=0,
            index_build_time_seconds=0,
        )

    def run_all(
        self,
        patterns: Optional[List[QueryPattern]] = None,
        category_counts: Optional[List[int]] = None,
        include_baseline: bool = True,
    ) -> BenchmarkResult:
        """Run complete benchmark suite.
        
        Optimized structure: For each category count, data is loaded once per scenario,
        then all search patterns are run against the loaded data.
        
        Structure for each category count:
        1. Scenario A:
           - Insert test (load data)
           - Warmup (warm indexes)
           - All search pattern tests
        2. Scenario B:
           - Insert test (load data)
           - Warmup (warm indexes)
           - All search pattern tests
        """
        if patterns is None:
            patterns = get_all_patterns(self.config)
        
        if category_counts is None:
            category_counts = self.config.categories.get("counts", [10, 100])

        all_results: List[TestMetrics] = []
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Run baseline
        if include_baseline:
            # Use uniform random pattern for baseline (find by name, fallback to first)
            baseline_pattern = None
            for p in patterns:
                if p.name == "uniform_random":
                    baseline_pattern = p
                    break
            if baseline_pattern is None:
                baseline_pattern = patterns[0]
            result = self.run_baseline(baseline_pattern)
            all_results.append(result)

        # Run for each category count
        for num_categories in category_counts:
            print(f"\n{'='*60}")
            print(f"Testing with {num_categories} categories")
            print('='*60)

            # Generate category assignments
            category_ids = assign_categories(
                len(self.dataset.vectors),
                num_categories,
                distribution="uniform",
            )

            # ==================== SCENARIO A ====================
            print(f"\n--- Scenario A ({num_categories} categories) ---")
            scenario_a = ScenarioA(self.config)
            
            try:
                # Step 1: Insert test
                print("\n[1/3] Insert Test (Scenario A)")
                insert_collector = MetricsCollector()
                insert_collector.start()
                
                setup_time_a = self._insert_data_scenario_a(
                    scenario_a,
                    self.dataset.vectors,
                    category_ids,
                    insert_collector,
                )
                insert_metrics_a = insert_collector.compute_insert_metrics(setup_time_a)
                
                # Record insert test result
                all_results.append(TestMetrics(
                    test_name="insert",
                    scenario="A",
                    num_categories=num_categories,
                    insert_metrics=insert_metrics_a,
                    search_metrics=SearchMetrics(
                        latency=LatencyMetrics(
                            count=0, total_seconds=0, mean_ms=0, p50_ms=0,
                            p95_ms=0, p99_ms=0, min_ms=0, max_ms=0
                        ),
                        qps=0,
                        recall=None,
                    ),
                    resource_metrics=insert_collector.get_latest_resource_metrics(),
                    setup_time_seconds=setup_time_a,
                    total_time_seconds=insert_collector.get_elapsed_time(),
                ))
                
                # Step 2: Warmup
                print("\n[2/3] Warmup (Scenario A)")
                warmup_duration_a = self._run_warmup(scenario_a, num_categories)
                
                # Step 3: Run all search patterns
                print("\n[3/3] Search Tests (Scenario A)")
                for pattern in patterns:
                    for repeat_idx in range(self.config.benchmark.repeat):
                        print(f"\n  Running {pattern.name} (repeat {repeat_idx + 1}/{self.config.benchmark.repeat})")
                        
                        search_collector = MetricsCollector()
                        search_collector.start()
                        
                        self._run_search_pattern(
                            scenario_a, pattern, num_categories, search_collector
                        )
                        
                        search_metrics = search_collector.compute_search_metrics(
                            self.dataset.neighbors,
                            self.config.search.top_k,
                        )
                        
                        all_results.append(TestMetrics(
                            test_name=pattern.name,
                            scenario="A",
                            num_categories=num_categories,
                            insert_metrics=self._create_empty_insert_metrics(),
                            search_metrics=search_metrics,
                            resource_metrics=search_collector.get_latest_resource_metrics(),
                            setup_time_seconds=0,
                            total_time_seconds=search_collector.get_elapsed_time(),
                        ))
            finally:
                scenario_a.cleanup()

            # ==================== SCENARIO B ====================
            print(f"\n--- Scenario B ({num_categories} categories) ---")
            scenario_b = ScenarioB(self.config)
            
            try:
                # Step 1: Insert test
                print("\n[1/3] Insert Test (Scenario B)")
                insert_collector = MetricsCollector()
                insert_collector.start()
                
                setup_time_b = self._insert_data_scenario_b(
                    scenario_b,
                    self.dataset.vectors,
                    category_ids,
                    num_categories,
                    insert_collector,
                )
                insert_metrics_b = insert_collector.compute_insert_metrics(setup_time_b)
                
                # Record insert test result
                all_results.append(TestMetrics(
                    test_name="insert",
                    scenario="B",
                    num_categories=num_categories,
                    insert_metrics=insert_metrics_b,
                    search_metrics=SearchMetrics(
                        latency=LatencyMetrics(
                            count=0, total_seconds=0, mean_ms=0, p50_ms=0,
                            p95_ms=0, p99_ms=0, min_ms=0, max_ms=0
                        ),
                        qps=0,
                        recall=None,
                    ),
                    resource_metrics=insert_collector.get_latest_resource_metrics(),
                    setup_time_seconds=setup_time_b,
                    total_time_seconds=insert_collector.get_elapsed_time(),
                ))
                
                # Step 2: Warmup
                print("\n[2/3] Warmup (Scenario B)")
                warmup_duration_b = self._run_warmup(scenario_b, num_categories)
                
                # Step 3: Run all search patterns
                print("\n[3/3] Search Tests (Scenario B)")
                for pattern in patterns:
                    for repeat_idx in range(self.config.benchmark.repeat):
                        print(f"\n  Running {pattern.name} (repeat {repeat_idx + 1}/{self.config.benchmark.repeat})")
                        
                        search_collector = MetricsCollector()
                        search_collector.start()
                        
                        self._run_search_pattern(
                            scenario_b, pattern, num_categories, search_collector
                        )
                        
                        search_metrics = search_collector.compute_search_metrics(
                            self.dataset.neighbors,
                            self.config.search.top_k,
                        )
                        
                        all_results.append(TestMetrics(
                            test_name=pattern.name,
                            scenario="B",
                            num_categories=num_categories,
                            insert_metrics=self._create_empty_insert_metrics(),
                            search_metrics=search_metrics,
                            resource_metrics=search_collector.get_latest_resource_metrics(),
                            setup_time_seconds=0,
                            total_time_seconds=search_collector.get_elapsed_time(),
                        ))
            finally:
                scenario_b.cleanup()

        # Create result object
        config_dict = {
            "qdrant": {
                "host": self.config.qdrant.host,
                "port": self.config.qdrant.port,
            },
            "hnsw": {
                "m": self.config.hnsw.m,
                "ef_construct": self.config.hnsw.ef_construct,
                "ef_search": self.config.hnsw.ef_search,
            },
            "search": {
                "top_k": self.config.search.top_k,
                "distance": self.config.search.distance_metric,
            },
            "dataset": {
                "name": self.dataset.name,
                "num_vectors": len(self.dataset.vectors),
                "dimensions": self.dataset.dimensions,
                "distance": self.dataset.distance,
            },
            "benchmark": {
                "num_queries": self.config.benchmark.num_queries,
                "batch_size": self.config.benchmark.batch_size,
                "warmup_queries": self.config.benchmark.warmup_queries,
                "repeat": self.config.benchmark.repeat,
            },
            "category_counts": category_counts,
        }

        self.results = all_results
        return BenchmarkResult(
            config=config_dict,
            results=all_results,
            timestamp=timestamp,
        )
