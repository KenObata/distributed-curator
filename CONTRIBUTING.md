# Contributing to distributed-curator

Thanks for your interest in contributing! This guide covers how to set up your development environment, run tests, and submit changes.

## Getting Started

### Prerequisites

- Python 3.9+
- Java 17 (for Spark and Scala compilation)
- sbt (for building the Scala UDF JAR)
- A C compiler (gcc or clang, for Cython extension)

### Setup

1. Clone the repo and install in editable mode:

```bash
git clone https://github.com/KenObata/distributed-curator.git
cd distributed-curator
pip install -e .
```

2. Build the Scala UDF JAR:

```bash
sbt assembly
cp target/scala-2.12/minhash-udf-assembly-0.1.jar distributed_curator/jars/
```

3. Install the pre-commit hooks:

```bash
./scripts/install-hooks.sh
```

This sets up automated checks that run on every commit: pytest, ruff (lint + format), scalafmt, and a custom Spark persist safety check.

### Verify your setup

```bash
python -c "from distributed_curator import partition_aware_deduplicate; print('OK')"
pytest test/unit_test
sbt test
```

The unit tests require the Scala JAR. If you skip step 2, tests that depend on Scala UDFs will be skipped automatically.

## Running Tests

### Python unit tests

```bash
pytest test/unit_test
```

### Python integration tests (requires AWS credentials and EMR)

Integration tests run on EMR against Common Crawl data. Set your S3 bucket environment variables:

```bash
export S3_BUCKET_TEST_INPUT=your-benchmark-bucket
export S3_RESULTS_BUCKET=your-benchmark-bucket
```

Example spark-submit for 100 WET files:

```bash
spark-submit \
  --master yarn \
  --jars $(python3 -c "from distributed_curator import get_jar_path; print(get_jar_path())") \
  --py-files s3://your-scripts-bucket/scripts/wet_file_utils.py \
  --conf spark.yarn.appMasterEnv.S3_BUCKET_TEST_INPUT=$S3_BUCKET_TEST_INPUT \
  --conf spark.yarn.appMasterEnv.S3_RESULTS_BUCKET=$S3_RESULTS_BUCKET \
  --conf spark.sql.execution.arrow.maxRecordsPerBatch=10000 \
  --num-executors 4 \
  --executor-cores 4 \
  --executor-memory 24g \
  --driver-memory 12g \
  --conf spark.sql.shuffle.partitions=50 \
  --conf spark.hadoop.fs.s3a.aws.credentials.provider=com.amazonaws.auth.DefaultAWSCredentialsProviderChain \
  --deploy-mode client \
  s3://your-scripts-bucket/scripts/spark_deduplication_test.py validation
```

For sample terraform setup, see [EMR Deployment](docs/emr-deployment.md)

### Scala tests

```bash
sbt test
```

### All checks (same as pre-commit)

```bash
pytest test/unit_test
ruff check distributed_curator/ test/
ruff format --check distributed_curator/ test/
sbt scalafmtCheckAll
python scripts/check_spark_persist.py
```

## Code Style

- **Python**: ruff handles lint and formatting. Config is in `pyproject.toml`. Line length is 120 characters.
- **Scala**: scalafmt handles formatting. Config is in `.scalafmt.conf`.
- **Test files**: use `_test.py` suffix (e.g., `spark_utils_test.py`), not `test_` prefix.

The pre-commit hook runs all style checks automatically. To bypass in rare cases: `git commit --no-verify`.

## Project Structure

```
distributed_curator/           # Python package
├── __init__.py                # public API: partition_aware_deduplicate, get_jar_path
├── spark_partition_aware_deduplication.py  # main pipeline
├── spark_utils.py             # Spark session helpers, S3 utilities
├── two_phase_partition_aware_union_find.py # Phase 2 global union-find
├── shingle_hash_wrapper.py    # NumPy SIMD MinHash wrapper
├── driver_memory_diagnostics.py # opt-in JVM diagnostics
├── udf.py                     # Python UDF (which calls shingle_hash_wrapper.py)
├── cython_minhash/            # Cython C extension
│   ├── shingle_hash.pyx
│   ├── murmurhash3.c
│   └── murmurhash3.h
└── jars/                      # bundled Scala assembly JAR (gitignored)

src/main/scala/                # Scala UDF source (built by sbt)
test/
├── unit_test/                 # fast, no cluster needed
├── integration_test/          # requires AWS/EMR
└── conftest.py                # shared Spark session fixture

docs/                          # detailed documentation
scripts/                       # pre-commit hooks, dev tooling
```

## How to Contribute

1. **Open an issue** describing the bug or feature you'd like to work on.
2. **Fork the repo** and create a branch from `main`.
3. **Make your changes** with tests.
4. **Run the full test suite** to make sure nothing breaks.
5. **Open a pull request** referencing the issue.

For larger changes (new pipeline steps, architecture modifications), please open an issue first to discuss the approach before writing code.

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).