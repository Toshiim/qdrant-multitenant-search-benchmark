"""Dataset loading and generation utilities."""

import hashlib
import os
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import h5py
import numpy as np
import requests
from tqdm import tqdm

from .config import Config


# Standard benchmark datasets with download URLs
DATASET_INFO = {
    "dbpedia-openai-1M-angular": {
        "url": "http://ann-benchmarks.com/dbpedia-openai-1000k-angular.hdf5",
        "num_vectors": 1000000,
        "dimensions": 1536,
        "distance": "Cosine",
    },
    "deep-image-96-angular": {
        "url": "http://ann-benchmarks.com/deep-image-96-angular.hdf5",
        "num_vectors": 9990000,
        "dimensions": 96,
        "distance": "Cosine",
    },
    "gist-960-euclidean": {
        "url": "http://ann-benchmarks.com/gist-960-euclidean.hdf5",
        "num_vectors": 1000000,
        "dimensions": 960,
        "distance": "Euclidean",
    },
    "glove-100-angular": {
        "url": "http://ann-benchmarks.com/glove-100-angular.hdf5",
        "num_vectors": 1183514,
        "dimensions": 100,
        "distance": "Cosine",
    },
}


@dataclass
class Dataset:
    """Dataset container."""

    name: str
    vectors: np.ndarray
    queries: np.ndarray
    dimensions: int
    distance: str
    neighbors: Optional[np.ndarray] = None  # Ground truth for recall calculation


def download_file(url: str, destination: str, chunk_size: int = 8192) -> None:
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get("content-length", 0))
    
    os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
    
    with open(destination, "wb") as f:
        with tqdm(total=total_size, unit="iB", unit_scale=True, desc=f"Downloading {os.path.basename(destination)}") as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                size = f.write(chunk)
                pbar.update(size)


def load_hdf5_dataset(path: str) -> Dataset:
    """Load dataset from HDF5 file (ANN-benchmarks format)."""
    with h5py.File(path, "r") as f:
        vectors = np.array(f["train"])
        queries = np.array(f["test"])
        neighbors = np.array(f["neighbors"]) if "neighbors" in f else None
        
        # Determine distance metric from filename or metadata
        if "angular" in path.lower() or "cosine" in path.lower():
            distance = "Cosine"
        elif "euclidean" in path.lower():
            distance = "Euclidean"
        else:
            distance = "Cosine"  # Default
        
        return Dataset(
            name=os.path.basename(path),
            vectors=vectors,
            queries=queries,
            dimensions=vectors.shape[1],
            distance=distance,
            neighbors=neighbors,
        )


def generate_synthetic_dataset(
    num_vectors: int,
    dimensions: int,
    num_queries: int = 1000,
    distance: str = "Cosine",
    seed: int = 42,
) -> Dataset:
    """Generate synthetic random dataset.
    
    Note: Vectors are NOT normalized for Cosine distance because Qdrant
    automatically normalizes vectors internally when using Cosine distance.
    Pre-normalization would be redundant.
    """
    np.random.seed(seed)
    
    vectors = np.random.randn(num_vectors, dimensions).astype(np.float32)
    queries = np.random.randn(num_queries, dimensions).astype(np.float32)
    
    return Dataset(
        name=f"synthetic-{num_vectors}-{dimensions}",
        vectors=vectors,
        queries=queries,
        dimensions=dimensions,
        distance=distance,
    )


def load_dataset(config: Config, data_dir: str = "./data") -> Dataset:
    """Load or generate dataset based on configuration."""
    dataset_name = config.dataset.name
    
    if dataset_name == "synthetic":
        return generate_synthetic_dataset(
            num_vectors=config.dataset.synthetic.num_vectors,
            dimensions=config.dataset.synthetic.dimensions,
            num_queries=config.benchmark.num_queries,
            distance=config.dataset.synthetic.distance,
        )
    
    if dataset_name not in DATASET_INFO:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_INFO.keys())}")
    
    info = DATASET_INFO[dataset_name]
    
    # Check if dataset file exists
    filename = f"{dataset_name}.hdf5"
    filepath = os.path.join(data_dir, filename)
    
    if not os.path.exists(filepath):
        print(f"Dataset {dataset_name} not found. Downloading...")
        download_file(info["url"], filepath)
    
    dataset = load_hdf5_dataset(filepath)
    
    return dataset


def assign_categories(
    num_vectors: int,
    num_categories: int,
    distribution: str = "uniform",
    seed: int = 42,
) -> np.ndarray:
    """Assign category IDs to vectors.
    
    Args:
        num_vectors: Number of vectors
        num_categories: Number of categories
        distribution: How to distribute vectors ("uniform", "zipfian")
        seed: Random seed
    
    Returns:
        Array of category IDs (integers from 0 to num_categories-1)
    """
    np.random.seed(seed)
    
    if distribution == "uniform":
        # Equal distribution across categories
        return np.random.randint(0, num_categories, size=num_vectors)
    elif distribution == "zipfian":
        # Zipfian distribution - some categories have more vectors
        # Generate zipfian weights manually
        alpha = 1.5
        weights = np.array([1.0 / (i ** alpha) for i in range(1, num_categories + 1)])
        weights = weights / weights.sum()
        
        return np.random.choice(num_categories, size=num_vectors, p=weights)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


def batch_vectors(
    vectors: np.ndarray,
    category_ids: np.ndarray,
    batch_size: int,
) -> Iterator[Tuple[np.ndarray, np.ndarray, List[int]]]:
    """Yield batches of vectors with their IDs and category IDs.
    
    Args:
        vectors: Vector array
        category_ids: Category ID for each vector
        batch_size: Size of each batch
    
    Yields:
        Tuples of (vector_batch, id_batch, category_id_batch)
    """
    num_vectors = len(vectors)
    
    for start_idx in range(0, num_vectors, batch_size):
        end_idx = min(start_idx + batch_size, num_vectors)
        
        batch_vectors = vectors[start_idx:end_idx]
        batch_ids = list(range(start_idx, end_idx))
        batch_categories = category_ids[start_idx:end_idx].tolist()
        
        yield batch_vectors, batch_ids, batch_categories


def get_vectors_by_category(
    vectors: np.ndarray,
    category_ids: np.ndarray,
    category: int,
) -> Tuple[np.ndarray, List[int]]:
    """Get all vectors belonging to a specific category.
    
    Args:
        vectors: Vector array
        category_ids: Category ID for each vector
        category: Target category ID
    
    Returns:
        Tuple of (vectors_in_category, original_indices)
    """
    mask = category_ids == category
    indices = np.where(mask)[0].tolist()
    return vectors[mask], indices
