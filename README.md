# distributed-curator

Partition-aware MinHash LSH deduplication library for large-scale text data curation on Apache Spark.

## Headline

2.53 billion documents deduplicated on a 63-node EMR cluster for ~$750 using partition-aware MinHash LSH — zero shuffle during local deduplication.

## Quick Start

```bash
pip install distributed-curator
```

```python
from distributed_curator import partition_aware_deduplicate, get_jar_path
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .config("spark.jars", get_jar_path()) \
    .getOrCreate()

# input_df must have a "doc_id" column and a text column
result = partition_aware_deduplicate(
    spark=spark,
    input_df=df,
    text_column="text",
    similarity_threshold=0.8,
    num_partitions=1000,
)

# result has original columns + "representative_id" and "is_duplicate"
unique_docs = result.filter(~result.is_duplicate)
```

## How It Works

The pipeline assigns documents to partitions based on LSH band hashes so that similar documents are co-located. All comparisons happen locally within partitions with no shuffle, then a two-phase union-find merges components across partition boundaries.

| Step | What happens | Shuffle? |
|------|-------------|----------|
| 1. MinHash | Cython/NumPy SIMD computes signatures via pandas_udf | No |
| 2. Partition assignment | Scala UDF maps LSH bands to partition IDs | No |
| 3. Identity repartition | Documents move to assigned partitions | Yes (once) |
| 4. Local dedup | Scala mapPartitions finds similar pairs within each partition | No |
| 5a. Local union-find | Scala partition-local connected components | No |
| 5b. Global union-find | Driver-side union-find with Eclipse Collections LongLongHashMap | No |
| 6. Mark duplicates | Join results back to input | Yes (once) |

For a detailed architecture walkthrough, see [docs/architecture.md](docs/architecture.md).

## Benchmark Results

All benchmarks run on Common Crawl WET files (CC-MAIN-2024-22) with `similarity_threshold=0.9`, `num_hashes=64`, `num_bands=8`.

| Scale | Documents | Duplicates | Rate | Cluster | Time | Cost |
|-------|-----------|------------|------|---------|------|------|
| 9K WET | 253M | 55M | 21.82% | 9× r5ad.8xlarge | ~1.5 hr | ~$50 |
| 90K WET | 2.53B | 827.7M | 32.76% | 63× r6gd.8xlarge | ~4.5 hr | ~$750 |

## Configuration

```python
partition_aware_deduplicate(
    spark,                          # SparkSession
    input_df,                       # DataFrame with doc_id + text columns
    text_column="text",             # name of the text column
    similarity_threshold=0.8,       # Jaccard similarity threshold (0.0–1.0)
    num_hashes=64,                  # MinHash signature length
    num_bands=8,                    # LSH bands (num_hashes must be divisible by num_bands)
    num_partitions=1000,            # number of partitions for dedup
    ngram=9,                        # character n-gram size for shingling
    checkpoint_path=None,           # [optional] S3/HDFS path to cache intermediate results
    enable_diagnostics=False,       # [optional] enable driver memory logging and heap capture
)
```

**When to tune:**

- `similarity_threshold`: lower catches more near-duplicates, higher is stricter. 0.8–0.9 is typical for web text.
- `num_partitions`: set to `spark.sql.shuffle.partitions`. More partitions = less memory per partition but more overhead.
- `num_bands` / `num_hashes`: controls the LSH probability curve. More bands = higher recall but more comparisons. `num_hashes` must be divisible by `num_bands`.
- `checkpoint_path`: recommended for runs over 1K WET files. Saves MinHash signatures so you don't recompute on retry.
- `ngram`: 9 works well for English web text. Shorter n-grams increase recall but reduce precision.

## Version Compatibility

| Component | Supported | Planned |
|-----------|-----------|---------|
| PySpark | 3.5.x | 4.0+ |
| Python | 3.9, 3.10, 3.11, 3.12 | |
| Scala | 2.12 | |
| AWS EMR | 7.5–7.12 (Spark 3.5) | |
| Java | 8, 11, 17 (runtime) | |

The Scala UDF JAR is compiled against Spark 3.5. Spark 4.0 support requires recompilation due to breaking API changes in `StructType`.

## Documentation

- [Architecture](docs/architecture.md) — detailed pipeline walkthrough, design decisions, and performance characteristics
- [EMR Deployment](docs/emr-deployment.md) — Terraform setup, bootstrap configuration, and spark-submit examples
- [Contributing](CONTRIBUTING.md) — development setup, running tests, and how to submit changes

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.