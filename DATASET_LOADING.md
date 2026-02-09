# How Dataset Loading Works

This document explains how the benchmark loads and processes datasets for testing.

## Overview

The dataset loading system supports two types of datasets:
1. **Synthetic datasets**: Generated on-the-fly using random vectors
2. **Standard benchmark datasets**: Downloaded from ann-benchmarks.com in HDF5 format

## Dataset Structure

All datasets are represented by the `Dataset` dataclass (defined in `src/benchmark/dataset.py`):

```python
@dataclass
class Dataset:
    name: str                           # Dataset identifier
    vectors: np.ndarray                 # Training vectors (N x D array)
    queries: np.ndarray                 # Query vectors (M x D array)
    dimensions: int                     # Vector dimensionality (D)
    distance: str                       # Distance metric ("Cosine" or "Euclidean")
    neighbors: Optional[np.ndarray]     # Ground truth neighbors for recall calculation
```

## Loading Process

### 1. Configuration

Dataset configuration is specified in `config.yaml`:

```yaml
dataset:
  name: "synthetic"  # or "dbpedia-openai-1M-angular", etc.
  synthetic:
    num_vectors: 100000
    dimensions: 128
    distance: "Cosine"
```

### 2. Main Loading Function

The `load_dataset(config, data_dir)` function (in `dataset.py`) handles all dataset loading:

```python
def load_dataset(config: Config, data_dir: str = "./data") -> Dataset:
    dataset_name = config.dataset.name
    
    if dataset_name == "synthetic":
        # Generate synthetic data
        return generate_synthetic_dataset(...)
    
    # Otherwise, load from HDF5 file
    # Download if needed
    # Parse and return
```

### 3. Synthetic Dataset Generation

For synthetic datasets:

1. **Vector Generation**: Uses `numpy.random.randn()` to generate random vectors
2. **Normalization**: If using Cosine distance, vectors are normalized to unit length
3. **Queries**: Generates separate query vectors using the same process

```python
def generate_synthetic_dataset(num_vectors, dimensions, num_queries, distance, seed):
    np.random.seed(seed)
    
    vectors = np.random.randn(num_vectors, dimensions).astype(np.float32)
    queries = np.random.randn(num_queries, dimensions).astype(np.float32)
    
    if distance == "Cosine":
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    
    return Dataset(...)
```

### 4. Standard Dataset Loading

For standard benchmark datasets (e.g., from ann-benchmarks.com):

#### Step 1: Check if dataset exists locally
```python
filename = f"{dataset_name}.hdf5"
filepath = os.path.join(data_dir, filename)

if not os.path.exists(filepath):
    print(f"Dataset {dataset_name} not found. Downloading...")
    download_file(info["url"], filepath)
```

#### Step 2: Download if needed
The `download_file()` function:
- Makes HTTP GET request with streaming
- Shows progress bar using `tqdm`
- Saves to local `./data/` directory
- Uses chunked reading (8KB chunks) for memory efficiency

#### Step 3: Parse HDF5 file
The `load_hdf5_dataset()` function reads the HDF5 file:

```python
def load_hdf5_dataset(path: str) -> Dataset:
    with h5py.File(path, "r") as f:
        vectors = np.array(f["train"])      # Training vectors
        queries = np.array(f["test"])       # Test/query vectors
        neighbors = np.array(f["neighbors"]) if "neighbors" in f else None
        
        # Infer distance metric from filename
        if "angular" in path.lower() or "cosine" in path.lower():
            distance = "Cosine"
        elif "euclidean" in path.lower():
            distance = "Euclidean"
        else:
            distance = "Cosine"  # Default
        
        return Dataset(...)
```

**HDF5 File Structure:**
- `train`: Training vectors (used for indexing)
- `test`: Query vectors (used for search)
- `neighbors`: Ground truth nearest neighbors (for recall calculation)

### 5. Category Assignment

After loading the dataset, vectors are assigned to categories:

```python
def assign_categories(num_vectors, num_categories, distribution, seed):
    np.random.seed(seed)
    
    if distribution == "uniform":
        # Equal distribution across categories
        return np.random.randint(0, num_categories, size=num_vectors)
    
    elif distribution == "zipfian":
        # Zipfian distribution - some categories have more vectors
        alpha = 1.5
        weights = np.array([1.0 / (i ** alpha) for i in range(1, num_categories + 1)])
        weights = weights / weights.sum()
        return np.random.choice(num_categories, size=num_vectors, p=weights)
```

**Distribution Types:**
- **Uniform**: Each vector has equal probability of belonging to any category
- **Zipfian**: Skewed distribution where some categories have many more vectors (realistic for multi-tenant scenarios)

### 6. Batching

During insertion, vectors are processed in batches:

```python
def batch_vectors(vectors, category_ids, batch_size):
    for start_idx in range(0, len(vectors), batch_size):
        end_idx = min(start_idx + batch_size, len(vectors))
        
        batch_vectors = vectors[start_idx:end_idx]
        batch_ids = list(range(start_idx, end_idx))
        batch_categories = category_ids[start_idx:end_idx].tolist()
        
        yield batch_vectors, batch_ids, batch_categories
```

This yields tuples of `(vectors, ids, category_ids)` for efficient batch insertion.

## Available Datasets

### Synthetic
- **Generated on-the-fly**
- Configurable size, dimensions, and distance metric
- Fast, no download required
- No ground truth for recall calculation

### Standard Benchmark Datasets

| Dataset | Vectors | Dimensions | Distance | Size |
|---------|---------|------------|----------|------|
| dbpedia-openai-1M-angular | 1M | 1536 | Cosine | ~6GB |
| deep-image-96-angular | 10M | 96 | Cosine | ~4GB |
| gist-960-euclidean | 1M | 960 | Euclidean | ~4GB |
| glove-100-angular | 1.2M | 100 | Cosine | ~500MB |

All datasets are downloaded from ann-benchmarks.com (using HTTP URLs as provided by the benchmark dataset repository)

**Note**: The dataset URLs use HTTP protocol as provided by the ann-benchmarks repository. Downloads are for publicly available benchmark datasets.

## Usage in Benchmark

1. **CLI calls** `load_dataset(config)` in `cli.py`
2. **Dataset object** is passed to `BenchmarkRunner`
3. **Runner** assigns categories using `assign_categories()`
4. **Vectors are batched** and inserted into Qdrant
5. **Query vectors** are used for search patterns

## Caching

- Downloaded datasets are cached in `./data/` directory
- Once downloaded, datasets are reused across benchmark runs
- No re-download unless file is deleted

## Memory Considerations

- Large datasets (10M+ vectors) can use significant RAM
- HDF5 files are memory-mapped by `h5py` for efficiency
- Batch processing helps manage memory during insertion
- Query vectors are kept in memory for fast access

## Error Handling

- **Invalid dataset name**: Raises `ValueError` with available options
- **Download failure**: HTTP errors are propagated from `requests.get()`
- **HDF5 parse errors**: File corruption or invalid format raises `h5py` exceptions
- **Missing fields**: `neighbors` field is optional (None if not present)

## Future Enhancements

Possible improvements to dataset loading:
1. Support for custom dataset formats (CSV, numpy arrays)
2. Dataset preprocessing (normalization, dimensionality reduction)
3. Partial dataset loading for very large datasets
4. Dataset validation and integrity checks
5. Dataset statistics and visualization
