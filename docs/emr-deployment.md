# EMR Deployment Guide

distributed-curator can run any spark environment. 
This guide introduces an example infra setup, how to deploy,
 and run distributed-curator using AWS EMR as example.

## Prerequisites

- AWS account with EMR permissions
- Terraform installed
- AWS CLI configured with appropriate credentials
- `sbt` installed (for building the Scala JAR)

## Infrastructure Setup

The project provides Terraform configuration for provisioning EMR clusters with pre-configured Spark, YARN, and S3 settings.

### Cluster Scales and meanings

| Scale | Instance Type | Nodes | Use Case |
|-------|--------------|-------|----------|
| 1 WET | m5a.xlarge | 1 | Smoke test — validate bootstrap and imports, no spark run expected |
| 100 WET | r5.2xlarge | 4 | Development — test pipeline end-to-end at small scale |
| 1K WET | r5ad.8xlarge | 2 | Medium scale — ~28M documents |
| 9K WET | r5ad.8xlarge | 9 | Large scale — ~253M documents |
| 90K WET | r6gd.8xlarge | 63 | Full Common Crawl snapshot — ~2.53B documents |

Scales 1K and above use NVMe-backed instances for shuffle performance. YARN directories are automatically configured based on instance type.

### Provisioning

```bash
cd terraform

# Smoke test cluster
terraform apply -var="wet_file_scale=1"

# Development cluster
terraform apply -var="wet_file_scale=100"

# Full scale (on-demand for stability)
terraform apply -var="wet_file_scale=90k"
```

Terraform outputs the SSH command and Spark History Server tunnel command after provisioning.

### Teardown

```bash
terraform destroy
```

Always destroy clusters after use to avoid costs.

## Bootstrap

The bootstrap action runs on every node during cluster creation. It installs the library and its dependencies:

```bash
sudo yum install -y java-17-amazon-corretto-devel python3-devel gcc
pip install distributed-curator
```

The Cython extension compiles from source on each node.
`python3-devel` and `gcc` are required for this build step.

### Verifying bootstrap

SSH into the master node and run:

```bash
python3 -c "from distributed_curator import partition_aware_deduplicate; print('OK')"
python3 -c "from distributed_curator.cython_minhash.shingle_hash import hash_shingles; print('Cython OK')"
python3 -c "from distributed_curator import get_jar_path; print(get_jar_path())"
```

### Verifying Scala UDFs

```bash
export JAVA_HOME=/usr/lib/jvm/java-17-amazon-corretto.x86_64

pyspark --jars $(python3 -c "from distributed_curator import get_jar_path; print(get_jar_path())")
```

In the pyspark shell:

```python
spark.conf.set("spark.sql.legacy.allowUntypedScalaUDF", "true")
spark._jvm.com.partitionAssignment.ComputePartitionAssignmentsUDF.registerUdf(spark._jsparkSession)
print("Scala UDFs OK")
```

### Verifying cluster health

```bash
yarn node -list
yarn application -list
```

## Running the Pipeline

### Environment variables

Set S3 bucket names for intermediate data and results:

```bash
export S3_BUCKET_TEST_INPUT=your-benchmark-bucket
export S3_RESULTS_BUCKET=your-benchmark-bucket
```

### spark-submit

```bash
spark-submit \
  --master yarn \
  --jars $(python3 -c "from distributed_curator import get_jar_path; print(get_jar_path())") \
  --py-files s3://your-scripts-bucket/scripts/wet_file_utils.py \
  --conf spark.yarn.appMasterEnv.S3_BUCKET_TEST_INPUT=$S3_BUCKET_TEST_INPUT \
  --conf spark.yarn.appMasterEnv.S3_RESULTS_BUCKET=$S3_RESULTS_BUCKET \
  --conf spark.sql.execution.arrow.maxRecordsPerBatch=10000 \
  --conf spark.hadoop.fs.s3a.aws.credentials.provider=com.amazonaws.auth.DefaultAWSCredentialsProviderChain \
  --deploy-mode client \
  s3://your-scripts-bucket/scripts/spark_deduplication_test.py <benchmark_level>
```

Benchmark levels: `development` (1 WET), `validation` (100), `production_proof` (1K), `scale_proof` (9K), `full_corpus` (90K).


--conf spark.yarn.appMasterEnv.S3_BUCKET_TEST_INPUT can be written in .env file instead:
ex)
```bash
S3_BUCKET_TEST_INPUT=your-input-bucket-name
S3_RESULTS_BUCKET=your-output-bucket-name
```

```bash
source .env && spark-submit \
```

### Scale-specific spark-submit settings

These are just sharing working settings based on common crawl datast.
If dataset is different, you need to find your own settings.

| Scale | Executors | Executor Memory | Driver Memory| Shuffle Partitions |
|-------|-----------|----------------|-------------------|-------------------|
| 1 WET | 1 | 12g | 12g | 4 |
| 100 WET | 4 | 24g | 12g | 50 |
| 1K WET | 14 | 27g | 16g | 1000 |
| 9K WET | 63 | 27g | 48g | 9000 |
| 90K WET | 504 | 27g | 58g | 27000 |

Add these as `--num-executors`, `--executor-memory`, and `--conf spark.sql.shuffle.partitions` flags.

## Monitoring

### Driver diagnostics
Because this library runs global union find on the driver,
this library provides a way to track driver memory and
enables diagnostics to capture JVM heap and memory state:

To turn-on diagnostics, set enable_diagnostics flag to be True.
```python
result = partition_aware_deduplicate(
    spark=spark,
    input_df=df,
    ...
    enable_diagnostics=True,
)
```

Diagnostic output includes periodic memory logs (every 30s), a heap histogram at completion, and a Native Memory Tracking summary.
If you use this diagnostics tool, you should define output file like this
```hcl
"spark.driver.extraJavaOptions" = join(" ", [
          "-XX:+HeapDumpOnOutOfMemoryError",
          "-XX:HeapDumpPath=/tmp/driver_heap_%p.hprof",
          "-XX:NativeMemoryTracking=summary",                           # remove if this want to remove 10% offheap overhead.
          "-Xlog:gc*:file=/tmp/driver_gc_%p.log:time,uptime,level,tags" # clean up left over jvm diagnostic log
        ]
```

## Troubleshooting

### Bootstrap failure

Check bootstrap logs:

```bash
cat /mnt/var/log/bootstrap-actions/1/stderr
cat /mnt/var/log/bootstrap-actions/1/stdout
```

Or from S3 if the cluster already terminated:

```bash
aws s3 ls s3://your-logs-bucket/logs/<cluster-id>/node/ --recursive | grep bootstrap
```

### Java version mismatch

If you see `UnsupportedClassVersionError`, set JAVA_HOME before running Spark:

```bash
export JAVA_HOME=/usr/lib/jvm/java-17-amazon-corretto.x86_64
```

For context, EMR installs multiple JDKs on every node (we saw Java 8, 11, 17, 21, 23, 25 in /usr/lib/jvm/). The system default JAVA_HOME points to Java 8.
When EMR launches Spark via YARN (through spark-submit), it sets JAVA_HOME to Java 17 internally. But when we SSH in and run pyspark manually, our shell uses the system default — Java 8. Java 8 (class file version 52.0) can't load Spark 3.5 classes that were compiled for Java 17 (class file version 61.0).

### awscli broken after pip install

If `aws` commands fail with `ModuleNotFoundError: No module named 'dateutil'`, a dependency upgrade overwrote the system dateutil. Fix:

```bash
sudo pip3 install python-dateutil==2.9.0
```

To prevent this, ensure `pandas` is not listed in the library's `pyproject.toml` dependencies — PySpark already provides it.

## Cost Estimates

| Scale | Cluster Config | Runtime | Approximate Cost |
|-------|---------------|---------|-----------------|
| 1 WET | 1× m5a.xlarge | ~2 min | <$0.01 |
| 100 WET | 4× r5.2xlarge | ~8 min | ~$2 |
| 1K WET | 2× r5ad.8xlarge | ~25 min | ~$15 |
| 9K WET | 9× r5ad.8xlarge | ~1.5 hr | ~$50 |
| 90K WET | 63× r6gd.8xlarge | ~3 hr | ~$500 |

Costs are for on-demand instances in us-east-1. Spot instances can reduce costs by 60–80% at the risk of interruption.
