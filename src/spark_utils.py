import logging
from typing import Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructType

builtin_max = max

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_spark_session_partition_aware(
    app_name: str = "PartitionAwareDedup", graphframes_jar_path: Optional[str] = None
) -> SparkSession:
    """Create optimized Spark session for large-scale deduplication"""

    import os

    # JARs are now installed in PySpark's jars directory - no need to specify paths

    # these are default config, so they can be overriden
    builder = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        .config("spark.sql.shuffle.partitions", "1000")
        .config("spark.default.parallelism", "1000")
        .config("spark.memory.offHeap.enabled", "true")
        .config("spark.memory.offHeap.size", "2g")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.ui.enabled", "true")
        .config("spark.ui.port", "4040")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.auth.IAMInstanceCredentialsProvider,com.amazonaws.auth.DefaultAWSCredentialsProviderChain",
        )
        .config(
            "spark.hadoop.fs.s3.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.auth.IAMInstanceCredentialsProvider,com.amazonaws.auth.DefaultAWSCredentialsProviderChain",
        )
        .config("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com")
        .config("spark.hadoop.fs.s3a.path.style.access", "false")
        .config("spark.hadoop.fs.s3a.connection.timeout", "600000")
        .config("spark.hadoop.fs.s3a.connection.establish.timeout", "60000")
        .config("spark.hadoop.fs.s3a.attempts.maximum", "3")
        .config("spark.hadoop.fs.s3a.retry.interval", "1000")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config(
            "spark.driver.extraJavaOptions",
            "--add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/sun.misc=ALL-UNNAMED",
        )
        .config(
            "spark.executor.extraJavaOptions",
            "--add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/sun.misc=ALL-UNNAMED",
        )
    )

    # Add GraphFrames JAR if provided
    if graphframes_jar_path and os.path.exists(graphframes_jar_path):
        builder = (
            builder.config("spark.jars", graphframes_jar_path)
            .config("spark.driver.extraClassPath", graphframes_jar_path)
            .config("spark.executor.extraClassPath", graphframes_jar_path)
        )
        print(f"✅ GraphFrames JAR configured: {graphframes_jar_path}")

    spark = builder.getOrCreate()

    # Set log level to reduce verbosity
    spark.sparkContext.setLogLevel("WARN")
    return spark


def create_spark_session_partition_aware_emr(app_name: str = "PartitionAwareDedup") -> SparkSession:
    """Create optimized Spark session for large-scale deduplication"""

    # these are default config, so they can be overriden
    # do not set hadoop jar here because EMR already defines hadoop.
    spark = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "134217728")
        .config("spark.default.parallelism", "1000")
        .config("spark.memory.offHeap.enabled", "true")
        .config("spark.memory.offHeap.size", "2g")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.ui.enabled", "true")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config(
            "spark.driver.extraJavaOptions",
            "--add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/sun.misc=ALL-UNNAMED",
        )
        .config(
            "spark.executor.extraJavaOptions",
            "--add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/sun.misc=ALL-UNNAMED",
        )
        .getOrCreate()
    )

    # Set log level to reduce verbosity
    spark.sparkContext.setLogLevel("WARN")
    return spark


def log_dataframe(df: DataFrame, is_debug_mode: bool = False):
    if is_debug_mode:
        docs_df = df.limit(10).collect()
        for row in docs_df:
            logger.info(f"  {row}")
    else:
        logger.info(f"Documents involved in duplicates: {df.count()}")


def _split_bucket_name_n_s3_prefix(full_s3_path: str) -> tuple[str, str]:
    """
    From full s3 path, return bucket name and prefix parts separately
    Args:
        full_s3_path: Full S3 path like "s3://bucket-name/development/df_name/" or "bucket-name/development/df_name/"
    Returns:
        s3_path without bucket name
        ex) "development/df_name/"
    Examples:
        "s3://my-bucket/development/data/" -> ["my-bucket", "development/data/"]
        "my-bucket/development/data/" -> ["my-bucket", "development/data/"]
        "my-bucket/development/data" -> ["my-bucket", "development/data/"]
    """
    # Remove s3:// prefix if present
    if full_s3_path.startswith("s3://"):
        full_s3_path = full_s3_path[5:]

    # Split into parts
    parts = full_s3_path.split("/", 1)

    if len(parts) == 1:
        raise ValueError("Only bucket name provided")

    s3_bucket_name = parts[0]
    prefix = parts[1]

    return s3_bucket_name, prefix


def does_file_exists(s3_path: str) -> bool:
    """
    Check if cached input files exist in personal S3 bucket for the target benchmark level
    Args:
        s3_path: ex) {s3_bucket_name}/development/{df_name}/
    Returns:
        True if cached input files exist for this benchmark level, False otherwise
    """
    try:
        import boto3

        s3 = boto3.client("s3")

        s3_bucket_name, prefix = _split_bucket_name_n_s3_prefix(full_s3_path=s3_path)
        print(f"Checking for cached input files at s3://{s3_bucket_name}/{prefix}")

        response = s3.list_objects_v2(Bucket=s3_bucket_name, Prefix=prefix)

        if "Contents" not in response:
            print(f"✗ No cached files found at s3://{s3_bucket_name}/{prefix}")
            return False

        # Check for _SUCCESS marker (indicates complete Spark write)
        success_files = [obj for obj in response["Contents"] if obj["Key"].endswith("_SUCCESS")]
        if not success_files:
            print(f"✗ No _SUCCESS marker found - incomplete write at s3://{s3_bucket_name}/{prefix}")
            return False

        # Check if we have actual parquet files
        parquet_files = [
            obj for obj in response["Contents"] if obj["Key"].endswith(".parquet") and "_temporary" not in obj["Key"]
        ]
        if not parquet_files:
            print(f"✗ No valid parquet files found at s3://{s3_bucket_name}/{prefix}")
            return False

        print(f"✅ Found {len(parquet_files)} parquet files and _SUCCESS marker in cache")
        return True

    except Exception as e:
        print(f"Error checking cached input files: {e!s}")
        return False


def get_dataframe_size_mb_estimate(row_count: int) -> float:
    """Fast approximation using only row count for Common Crawl text data"""
    # Conservative estimate for web text data based on Common Crawl characteristics
    avg_bytes_per_row = 2000  # Typical web page text size

    total_size_bytes = row_count * avg_bytes_per_row
    return total_size_bytes / (1024 * 1024)


def calculate_optimal_partitions(df: DataFrame, row_count: int, target_file_size_mb: int = 256) -> int:
    """Calculate partition count for target file size using lightweight row count estimation"""

    # Use lightweight row count approach for good speed/accuracy balance
    estimated_total_mb = get_dataframe_size_mb_estimate(row_count)

    optimal_partitions = builtin_max(1, int(estimated_total_mb / target_file_size_mb))
    print(f"Row count estimation: {estimated_total_mb:.0f}MB -> {optimal_partitions} partitions")
    return optimal_partitions


def upload_df_to_s3(df: DataFrame, s3_path: str, row_count: int) -> None:
    """
    Upload DataFrame to S3
    Args:
        df: DataFrame to upload
        s3_path: S3 path to upload to
        file_name: File name to upload
    """
    try:
        # Ensure proper path formatting
        if not s3_path.endswith("/"):
            s3_path += "/"

        coalesce_count = calculate_optimal_partitions(df=df, row_count=row_count, target_file_size_mb=256)

        # Upload with error handling
        df.coalesce(coalesce_count).write.mode("overwrite").parquet(s3_path)
        print(f"✅ Uploaded DataFrame to S3: {s3_path}")

    except Exception as e:
        print(f"❌ Failed to upload DataFrame to S3: {e!s}")
        raise e


def read_parquet_from_s3(s3_path: str, spark: SparkSession, schema: StructType = None) -> DataFrame:
    """
    Read Parquet file/directory from S3
    Args:
        s3_path: S3 path to read from (should end with /)
        spark: SparkSession
        schema: Optional schema to avoid inference issues
    """
    try:
        # Ensure proper path formatting
        if not s3_path.endswith("/"):
            s3_path += "/"

        if schema:
            df_filtered = spark.read.schema(schema).parquet(s3_path)
        else:
            df_filtered = spark.read.parquet(s3_path)
        print(f"✅ Loaded cached data from S3: {s3_path}")
        return df_filtered

    except Exception as e:
        print(f"❌ Failed to load cached data from {s3_path}: {e!s}")
        print("Falling back to downloading from Common Crawl...")
        # Fall back to Common Crawl download
        raise e


def set_spark_context(spark, short_desc, long_desc=None):
    spark.sparkContext.setJobDescription(short_desc)
    spark.sparkContext.setLocalProperty("callSite.short", short_desc)
    if long_desc:
        spark.sparkContext.setLocalProperty("callSite.long", long_desc)
