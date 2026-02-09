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
from .metrics import MetricsCollector, TestMetrics, InsertMetrics, LatencyMetrics, ThroughputMetrics, ResourceMetrics
from .qdrant_client_wrapper import BaselineScenario, ScenarioA, ScenarioB
from .query_patterns import QueryPattern, get_all_patterns


@dataclass
class LoadMetrics:
    """Metrics for collection loading operation."""

    scenario: str
    num_categories: int
    setup_time_seconds: float
    insert_metrics: InsertMetrics
    total_time_seconds: float
    resource_metrics: ResourceMetrics

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "test_name": "collection_load",
            "scenario": self.scenario,
            "num_categories": self.num_categories,
            "setup_time_seconds": round(self.setup_time_seconds, 3),
            "insert": self.insert_metrics.to_dict(),
            "total_time_seconds": round(self.total_time_seconds, 3),
            "resources": self.resource_metrics.to_dict(),
        }


@dataclass
class BenchmarkResult:
    """Result of a complete benchmark run."""

    config: Dict
    results: List[TestMetrics]
    load_results: List[LoadMetrics]
    timestamp: str

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "config": self.config,
            "timestamp": self.timestamp,
            "load_results": [r.to_dict() for r in self.load_results],
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
        self.load_results: List[LoadMetrics] = []

    def _insert_data_scenario_a(
        self,
        scenario: ScenarioA,
        vectors: np.ndarray,
        category_ids: np.ndarray,
        collector: MetricsCollector,
        force_recreate: bool = True,
    ) -> float:
        """Insert data for Scenario A (single collection)."""
        setup_time = scenario.setup(self.dataset.dimensions, self.dataset.distance, force_recreate=force_recreate)
        
        # If skipping load and collection exists, don't insert
        if setup_time == 0.0 and not force_recreate:
            return setup_time
        
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
        force_recreate: bool = True,
    ) -> float:
        """Insert data for Scenario B (multiple collections)."""
        setup_time = scenario.setup(
            self.dataset.dimensions,
            self.dataset.distance,
            num_categories,
            force_recreate=force_recreate,
        )

        # If skipping load and collections exist, don't insert
        if setup_time == 0.0 and not force_recreate:
            return setup_time

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
        force_recreate: bool = True,
    ) -> float:
        """Insert data for baseline scenario."""
        setup_time = scenario.setup(self.dataset.dimensions, self.dataset.distance, force_recreate=force_recreate)

        # If skipping load and collection exists, don't insert
        if setup_time == 0.0 and not force_recreate:
            return setup_time

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

    def _run_search_pattern(
        self,
        scenario,
        pattern: QueryPattern,
        num_categories: int,
        collector: MetricsCollector,
    ):
        """Run search queries according to pattern."""
        # Warmup
        warmup_gen = pattern.generate_queries(
            self.dataset.queries,
            num_categories,
            self.config.benchmark.warmup_queries,
        )
        for query_vec, cat_id in warmup_gen:
            if hasattr(scenario, "search"):
                if isinstance(scenario, BaselineScenario):
                    scenario.search(query_vec.tolist())
                else:
                    scenario.search(query_vec.tolist(), cat_id)

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
        skip_load: bool = False,
        reset_cache: bool = False,
        cleanup_after: bool = True,
    ) -> TestMetrics:
        """Run benchmark for Scenario A.
        
        Args:
            pattern: Query pattern to use
            num_categories: Number of categories
            category_ids: Category assignments for vectors
            skip_load: If True, skip loading if collection exists
            reset_cache: If True, reset index cache before search (cold start)
            cleanup_after: If True, cleanup collection after test
        """
        print(f"\n=== Running Scenario A: {pattern.name} ({num_categories} categories) ===")
        
        scenario = ScenarioA(self.config)
        collector = MetricsCollector()
        collector.start()

        try:
            # Insert phase
            force_recreate = not skip_load
            setup_time = self._insert_data_scenario_a(
                scenario,
                self.dataset.vectors,
                category_ids,
                collector,
                force_recreate=force_recreate,
            )
            index_build_time = setup_time
            insert_metrics = collector.compute_insert_metrics(index_build_time)

            # Reset index cache if requested (simulate cold start)
            if reset_cache and skip_load:
                print("  Resetting index cache for cold start...")
                reset_time = scenario.reset_index_cache()
                print(f"  Index cache reset completed in {reset_time:.2f}s")

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
            if cleanup_after:
                scenario.cleanup()

    def run_scenario_b(
        self,
        pattern: QueryPattern,
        num_categories: int,
        category_ids: np.ndarray,
        skip_load: bool = False,
        reset_cache: bool = False,
        cleanup_after: bool = True,
    ) -> TestMetrics:
        """Run benchmark for Scenario B.
        
        Args:
            pattern: Query pattern to use
            num_categories: Number of categories
            category_ids: Category assignments for vectors
            skip_load: If True, skip loading if collections exist
            reset_cache: If True, reset index cache before search (cold start)
            cleanup_after: If True, cleanup collections after test
        """
        print(f"\n=== Running Scenario B: {pattern.name} ({num_categories} categories) ===")
        
        scenario = ScenarioB(self.config)
        collector = MetricsCollector()
        collector.start()

        try:
            # Insert phase
            force_recreate = not skip_load
            setup_time = self._insert_data_scenario_b(
                scenario,
                self.dataset.vectors,
                category_ids,
                num_categories,
                collector,
                force_recreate=force_recreate,
            )
            index_build_time = setup_time
            insert_metrics = collector.compute_insert_metrics(index_build_time)

            # Reset index cache if requested (simulate cold start)
            if reset_cache and skip_load:
                print("  Resetting index cache for cold start...")
                reset_time = scenario.reset_index_cache()
                print(f"  Index cache reset completed in {reset_time:.2f}s")

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
            if cleanup_after:
                scenario.cleanup()

    def run_baseline(
        self,
        pattern: QueryPattern,
        skip_load: bool = False,
        reset_cache: bool = False,
        cleanup_after: bool = True,
    ) -> TestMetrics:
        """Run baseline benchmark (no categories).
        
        Args:
            pattern: Query pattern to use
            skip_load: If True, skip loading if collection exists
            reset_cache: If True, reset index cache before search (cold start)
            cleanup_after: If True, cleanup collection after test
        """
        print(f"\n=== Running Baseline: {pattern.name} ===")
        
        scenario = BaselineScenario(self.config)
        collector = MetricsCollector()
        collector.start()

        try:
            # Insert phase
            force_recreate = not skip_load
            setup_time = self._insert_data_baseline(
                scenario,
                self.dataset.vectors,
                collector,
                force_recreate=force_recreate,
            )
            index_build_time = setup_time
            insert_metrics = collector.compute_insert_metrics(index_build_time)

            # Reset index cache if requested (simulate cold start)
            if reset_cache and skip_load:
                print("  Resetting index cache for cold start...")
                reset_time = scenario.reset_index_cache()
                print(f"  Index cache reset completed in {reset_time:.2f}s")

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
            if cleanup_after:
                scenario.cleanup()

    def load_collections(
        self,
        category_counts: Optional[List[int]] = None,
        include_baseline: bool = True,
        scenarios: str = "both",
    ) -> List[LoadMetrics]:
        """Load collections and return load metrics separately.
        
        This method only loads data into collections without running search tests.
        It's useful for separating load time metrics from search metrics.
        
        Args:
            category_counts: List of category counts to test
            include_baseline: Whether to load baseline collection
            scenarios: Which scenarios to load ("A", "B", or "both")
        
        Returns:
            List of LoadMetrics for each loaded configuration
        """
        if category_counts is None:
            category_counts = self.config.categories.get("counts", [10, 100])

        load_results: List[LoadMetrics] = []

        # Load baseline
        if include_baseline:
            print("\n=== Loading Baseline Collection ===")
            scenario = BaselineScenario(self.config)
            collector = MetricsCollector()
            collector.start()

            setup_time = self._insert_data_baseline(
                scenario,
                self.dataset.vectors,
                collector,
                force_recreate=True,
            )
            insert_metrics = collector.compute_insert_metrics(setup_time)
            resource_metrics = collector.get_latest_resource_metrics()
            total_time = collector.get_elapsed_time()

            load_results.append(LoadMetrics(
                scenario="baseline",
                num_categories=0,
                setup_time_seconds=setup_time,
                insert_metrics=insert_metrics,
                total_time_seconds=total_time,
                resource_metrics=resource_metrics,
            ))

        # Load for each category count
        for num_categories in category_counts:
            print(f"\n{'='*60}")
            print(f"Loading collections for {num_categories} categories")
            print('='*60)

            # Generate category assignments
            category_ids = assign_categories(
                len(self.dataset.vectors),
                num_categories,
                distribution="uniform",
            )

            # Load Scenario A
            if scenarios in ("A", "both"):
                print(f"\n--- Loading Scenario A ({num_categories} categories) ---")
                scenario_a = ScenarioA(self.config)
                collector_a = MetricsCollector()
                collector_a.start()

                setup_time_a = self._insert_data_scenario_a(
                    scenario_a,
                    self.dataset.vectors,
                    category_ids,
                    collector_a,
                    force_recreate=True,
                )
                insert_metrics_a = collector_a.compute_insert_metrics(setup_time_a)
                resource_metrics_a = collector_a.get_latest_resource_metrics()
                total_time_a = collector_a.get_elapsed_time()

                load_results.append(LoadMetrics(
                    scenario="A",
                    num_categories=num_categories,
                    setup_time_seconds=setup_time_a,
                    insert_metrics=insert_metrics_a,
                    total_time_seconds=total_time_a,
                    resource_metrics=resource_metrics_a,
                ))

            # Load Scenario B
            if scenarios in ("B", "both"):
                print(f"\n--- Loading Scenario B ({num_categories} categories) ---")
                scenario_b = ScenarioB(self.config)
                collector_b = MetricsCollector()
                collector_b.start()

                setup_time_b = self._insert_data_scenario_b(
                    scenario_b,
                    self.dataset.vectors,
                    category_ids,
                    num_categories,
                    collector_b,
                    force_recreate=True,
                )
                insert_metrics_b = collector_b.compute_insert_metrics(setup_time_b)
                resource_metrics_b = collector_b.get_latest_resource_metrics()
                total_time_b = collector_b.get_elapsed_time()

                load_results.append(LoadMetrics(
                    scenario="B",
                    num_categories=num_categories,
                    setup_time_seconds=setup_time_b,
                    insert_metrics=insert_metrics_b,
                    total_time_seconds=total_time_b,
                    resource_metrics=resource_metrics_b,
                ))

        self.load_results = load_results
        return load_results

    def run_all(
        self,
        patterns: Optional[List[QueryPattern]] = None,
        category_counts: Optional[List[int]] = None,
        include_baseline: bool = True,
        skip_load: bool = False,
        reset_cache: bool = False,
    ) -> BenchmarkResult:
        """Run complete benchmark suite.
        
        Args:
            patterns: Query patterns to run (all if None)
            category_counts: Category counts to test (from config if None)
            include_baseline: Whether to include baseline test
            skip_load: If True, skip loading collections if they exist (faster)
            reset_cache: If True, reset index cache before each search test (cold start)
        """
        if patterns is None:
            patterns = get_all_patterns(self.config)
        
        if category_counts is None:
            category_counts = self.config.categories.get("counts", [10, 100])

        all_results: List[TestMetrics] = []
        all_load_results: List[LoadMetrics] = []
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # If not skipping load, run load phase and collect metrics
        if not skip_load:
            print("\n" + "="*60)
            print("Load phase: Creating and populating collections")
            print("="*60)
            all_load_results = self.load_collections(
                category_counts=category_counts,
                include_baseline=include_baseline,
            )

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
            
            result = self.run_baseline(
                baseline_pattern,
                skip_load=skip_load,
                reset_cache=reset_cache,
                cleanup_after=False,  # Keep collection for potential re-run
            )
            all_results.append(result)

        # Run for each category count
        for num_categories in category_counts:
            print(f"\n{'='*60}")
            if skip_load:
                print(f"Testing with {num_categories} categories (using existing collections)")
            else:
                print(f"Search phase: Testing with {num_categories} categories")
            print('='*60)

            # Generate category assignments
            category_ids = assign_categories(
                len(self.dataset.vectors),
                num_categories,
                distribution="uniform",
            )

            # Run each pattern for both scenarios
            for pattern in patterns:
                # Run Scenario A
                for _ in range(self.config.benchmark.repeat):
                    result_a = self.run_scenario_a(
                        pattern,
                        num_categories,
                        category_ids,
                        skip_load=skip_load,
                        reset_cache=reset_cache,
                        cleanup_after=False,  # Keep collection for potential re-run
                    )
                    all_results.append(result_a)

                # Run Scenario B
                for _ in range(self.config.benchmark.repeat):
                    result_b = self.run_scenario_b(
                        pattern,
                        num_categories,
                        category_ids,
                        skip_load=skip_load,
                        reset_cache=reset_cache,
                        cleanup_after=False,  # Keep collection for potential re-run
                    )
                    all_results.append(result_b)

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
            "skip_load": skip_load,
            "reset_cache": reset_cache,
        }

        self.results = all_results
        self.load_results = all_load_results
        return BenchmarkResult(
            config=config_dict,
            results=all_results,
            load_results=all_load_results,
            timestamp=timestamp,
        )
