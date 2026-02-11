# How Dataset Loading Works

This document explains how the benchmark loads and processes datasets for testing.

## Overview

The dataset loading system supports three types of datasets:
1. **Synthetic datasets**: Generated on-the-fly using random vectors
2. **Hugging Face datasets**: Downloaded from Hugging Face Hub (recommended)
3. **HDF5 benchmark datasets**: Downloaded from ann-benchmarks.com in HDF5 format

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
  name: "synthetic"  # or "dbpedia-entities-openai-1M", "dbpedia-openai-1M-angular", etc.
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
2. **No Pre-normalization**: Vectors are NOT normalized because Qdrant automatically normalizes vectors internally when using Cosine distance
3. **Queries**: Generates separate query vectors using the same process

```python
def generate_synthetic_dataset(num_vectors, dimensions, num_queries, distance, seed):
    np.random.seed(seed)
    
    vectors = np.random.randn(num_vectors, dimensions).astype(np.float32)
    queries = np.random.randn(num_queries, dimensions).astype(np.float32)
    
    # Note: No normalization needed for Cosine distance
    # Qdrant handles normalization internally
    
    return Dataset(...)
```

**Important**: Qdrant automatically normalizes vectors when using `Distance.COSINE`, so pre-normalizing vectors would be redundant and waste computation time. The benchmark sends raw vectors to Qdrant for more accurate performance measurements.

### 4. Hugging Face Dataset Loading (Recommended)

For datasets hosted on Hugging Face Hub (e.g., `KShivendu/dbpedia-entities-openai-1M`):

#### Step 1: Load dataset from Hugging Face
```python
from datasets import load_dataset as hf_load_dataset

# Load the dataset with the specified split
dataset = hf_load_dataset(repo_id, split="train", cache_dir=cache_dir)
```

#### Step 2: Extract embeddings
```python
# Extract embeddings from the specified column
embeddings = np.array(dataset[embedding_column])

# Convert to float32 if needed
if embeddings.dtype != np.float32:
    embeddings = embeddings.astype(np.float32)

# Handle list of lists format
if len(embeddings.shape) == 1:
    embeddings = np.array([np.array(emb, dtype=np.float32) for emb in embeddings])
```

#### Step 3: Sample queries
```python
# Sample query vectors from the main dataset
if len(embeddings) > num_queries:
    indices = np.random.choice(len(embeddings), size=num_queries, replace=False)
    queries = embeddings[indices]
else:
    queries = embeddings.copy()
```

**Benefits of Hugging Face datasets:**
- More reliable download infrastructure
- Built-in caching and resume capability
- Easy to add new datasets
- Community-maintained datasets
- No need for HDF5 files

### 5. HDF5 Dataset Loading (Legacy)

For standard benchmark datasets (e.g., from ann-benchmarks.com):

**Note**: ann-benchmarks.com URLs may not always be accessible. Consider using Hugging Face datasets instead.

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

### 6. Category Assignment

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

### 7. Batching

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

### Hugging Face Datasets (Recommended)

| Dataset | Vectors | Dimensions | Distance | Repository |
|---------|---------|------------|----------|------------|
| dbpedia-entities-openai-1M | 1M | 1536 | Cosine | [KShivendu/dbpedia-entities-openai-1M](https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M) |

**Benefits:**
- Reliable download infrastructure with automatic resume
- Built-in caching (datasets are cached in `./data/huggingface_cache/`)
- Easy to extend with more datasets
- Community-maintained and updated

**Usage:**
```yaml
dataset:
  name: "dbpedia-entities-openai-1M"
```

### HDF5 Benchmark Datasets (Legacy)

| Dataset | Vectors | Dimensions | Distance | Size |
|---------|---------|------------|----------|------|
| dbpedia-openai-1M-angular | 1M | 1536 | Cosine | ~6GB |
| deep-image-96-angular | 10M | 96 | Cosine | ~4GB |
| gist-960-euclidean | 1M | 960 | Euclidean | ~4GB |
| glove-100-angular | 1.2M | 100 | Cosine | ~500MB |

**Note:** These datasets are downloaded from ann-benchmarks.com which may not always be accessible. For reliable dataset loading, use Hugging Face datasets instead.

All datasets are downloaded from ann-benchmarks.com (using HTTP URLs as provided by the benchmark dataset repository)

**Security Note**: The dataset URLs currently use HTTP protocol as provided by the ann-benchmarks repository. Downloads are for publicly available benchmark datasets. For production use, consider:
1. Verifying dataset integrity using checksums after download
2. Downloading datasets once and storing them in a trusted location
3. Using a local mirror with HTTPS support

## Usage in Benchmark

1. **CLI calls** `load_dataset(config)` in `cli.py`
2. **Dataset object** is passed to `BenchmarkRunner`
3. **Runner** assigns categories using `assign_categories()`
4. **Vectors are batched** and inserted into Qdrant
5. **Query vectors** are used for search patterns

## Caching

- **HDF5 datasets**: Cached in `./data/` directory
- **Hugging Face datasets**: Cached in `./data/huggingface_cache/` directory
- Once downloaded, datasets are reused across benchmark runs
- No re-download unless cache is deleted
- Hugging Face datasets support automatic resume on interrupted downloads

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

## Adding New Datasets

### Adding a Hugging Face Dataset

To add a new Hugging Face dataset:

1. Find a dataset on Hugging Face Hub with embedding vectors
2. Add it to `HF_DATASET_INFO` in `src/benchmark/dataset.py`:

```python
HF_DATASET_INFO = {
    "your-dataset-name": {
        "repo_id": "username/dataset-name",
        "embedding_column": "embeddings",  # Column containing vectors
        "dimensions": 768,                  # Vector dimensions
        "distance": "Cosine",              # Distance metric
        "split": "train",                  # Dataset split to use
    },
}
```

3. Use it in your config:

```yaml
dataset:
  name: "your-dataset-name"
```

### Adding an HDF5 Dataset

To add a new HDF5 dataset from ann-benchmarks.com:

1. Find the dataset URL
2. Add it to `DATASET_INFO` in `src/benchmark/dataset.py`:

```python
DATASET_INFO = {
    "your-dataset-name": {
        "url": "http://ann-benchmarks.com/your-dataset.hdf5",
        "num_vectors": 1000000,
        "dimensions": 128,
        "distance": "Cosine",
    },
}
```

## Future Enhancements

Possible improvements to dataset loading:
1. Support for custom dataset formats (CSV, numpy arrays)
2. Dataset preprocessing (normalization, dimensionality reduction)
3. Partial dataset loading for very large datasets
4. Dataset validation and integrity checks
5. Dataset statistics and visualization
6. Support for more Hugging Face datasets
7. Automatic dataset discovery from Hugging Face Hub
