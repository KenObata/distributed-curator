import os
import time

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col

try:
    from spark_partition_aware_deduplicattion_v2 import partition_aware_deduplicate
except ImportError:
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
    from spark_partition_aware_deduplicattion_v2 import partition_aware_deduplicate

import boto3
from wet_file_utils import get_wet_file_paths, parse_wet_record_v2

from spark_utils import (
    create_spark_session_partition_aware,
    create_spark_session_partition_aware_emr,
    does_file_exists,
    read_parquet_from_s3,
    upload_df_to_s3,
)

s3 = boto3.client("s3")

S3_BUCKET_TEST_INPUT = "text-dedupe-benchmark"

BENCHMARK_CONFIGS = {
    "development": {"wet_files": 1},
    "validation": {"wet_files": 100},
    "production_proof": {"wet_files": 1000},
    "scale_proof": {"wet_files": 9000},
    "full_corpus": {"wet_files": 90000},
}


def init() -> None:
    print("\n" + "=" * 80)
    print("COMMON CRAWL STRESS TEST - PARTITION-AWARE DEDUPLICATION")
    print("=" * 80)


def does_cralw_file_exists(benchmark_level: str) -> bool:
    """
    Check if cached input files exist in personal S3 bucket for the target benchmark level
    Args:
        benchmark_level: One of 'development', 'validation', 'production_proof', 'scale_proof'
    Returns:
        True if cached input files exist for this benchmark level, False otherwise
    """
    if benchmark_level not in BENCHMARK_CONFIGS:
        raise ValueError(f"Invalid benchmark_level: {benchmark_level}")

    try:
        # Check for cached input files in personal bucket
        # Cache structure: s3://text-dedupe-benchmark/{benchmark_level}/
        print(f"Checking for cached input files for {benchmark_level} in {S3_BUCKET_TEST_INPUT}...")

        response = s3.list_objects_v2(Bucket=S3_BUCKET_TEST_INPUT, Prefix=benchmark_level)

        if "Contents" not in response:
            print(f"✗ No cached files found at s3://{S3_BUCKET_TEST_INPUT}/{benchmark_level}")
            return False

        return True

    except Exception as e:
        print(f"Error checking cached input files: {e!s}")
        print("This might be due to:")
        print("- AWS credentials not configured")
        print("- S3 bucket access issues")
        print("- Network connectivity issues")
        return False


def read_wet_files_from_s3(spark: SparkSession, max_files: int, cc_main_id: str) -> DataFrame:
    """
    spark.read.text() splits the file into one row per line:
    """
    file_paths = get_wet_file_paths(max_files, cc_main_id)
    print(f"Reading {len(file_paths)} WET files...")
    wet_df = spark.read.text(file_paths)
    return wet_df


def test_integration_commoncrawl_sample(benchmark_level: str = "development", cc_main_id: str = "CC-MAIN-2024-22"):
    """
    Stress test with Common Crawl data from AWS S3
    Tests deduplication performance on real-world web crawl data

    Args:
    - benchmark_level: One of 'development', 'validation', 'production_proof', 'scale_proof', 'full_corpus'
    - cc_main_id: snapshot ID for that month.
      ex) https://data.commoncrawl.org/crawl-data/CC-MAIN-2024-22/index.html
    """
    init()
    # Get benchmark configuration
    if benchmark_level not in BENCHMARK_CONFIGS:
        raise ValueError(f"Invalid benchmark_level. Choose from: {list(BENCHMARK_CONFIGS.keys())}")

    config = BENCHMARK_CONFIGS[benchmark_level]
    max_files = config["wet_files"]

    s3_path = f"s3://{S3_BUCKET_TEST_INPUT}/{benchmark_level}/common_crawl_df_filtered"

    print(f"Benchmark Level: {benchmark_level}")
    # Detect environment and choose appropriate Spark session
    is_emr = os.path.exists("/emr") or "EMR" in os.environ.get("SPARK_HOME", "") or os.environ.get("AWS_EMR_CLUSTER_ID")

    if is_emr:
        print("Running on EMR - using EMR Spark session")
        spark = create_spark_session_partition_aware_emr(f"CommonCrawl_{max_files}_WET_files")
    else:
        print("Running locally - using local Spark session with S3 support")
        # os.environ["_JAVA_OPTIONS"] = "--add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/sun.misc=ALL-UNNAMED"
        os.environ["PYSPARK_SUBMIT_ARGS"] = (
            "--driver-java-options '--add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/sun.misc=ALL-UNNAMED' pyspark-shell"
        )

        spark = create_spark_session_partition_aware("CommonCrawlStressTest")

    has_common_crawl_df_filtered = does_file_exists(s3_path=s3_path)
    if has_common_crawl_df_filtered:
        df_filtered = read_parquet_from_s3(s3_path=s3_path, spark=spark)
    else:
        print(f"No cached input files found for {benchmark_level}")

        try:
            # For WET files, we need to read them as text files and parse the format
            # WET format contains:
            # - WARC-Type: conversion
            # - WARC-Target-URI: <url>
            # - Content-Length: <length>
            # - <blank line>
            # - <extracted text content>

            wet_df = read_wet_files_from_s3(spark, max_files, cc_main_id)
            wet_df.show()

            # Process in partitions and parse WET format
            # we do lazy eval insteaed of glom - process row by row to avoid stoing all data in list.
            print("Parsing WET format...")
            parsed_rdd = wet_df.rdd.mapPartitions(lambda rows: parse_wet_record_v2(row.value for row in rows))

            # Convert to DataFrame and take sample
            from pyspark.sql.types import StringType, StructField, StructType

            schema = StructType([StructField("doc_id", StringType(), True), StructField("text", StringType(), True)])

            print("Creating DataFrame from parsed records...")
            df_parsed = spark.createDataFrame(parsed_rdd, schema)
            df_parsed.show()

            print("Applying filters...")
            df_filtered = df_parsed.filter(col("text").isNotNull())

            df_filtered = df_filtered.cache()
            test_count = df_filtered.count()
            print(f"Test dataset size: {test_count:,} documents")
            upload_df_to_s3(df=df_filtered, s3_path=s3_path, row_count=test_count)
            df_filtered.unpersist()

            # to avoid re-computing of df_filtered, read from s3.
            df_filtered = read_parquet_from_s3(s3_path=s3_path, spark=spark)
        except Exception as e:
            print("Common Crawl access requires AWS credentials or has connectivity issues.")
            raise Exception(f"Error reading WET files: {e!s}") from e

    # Performance monitoring
    start_time = time.time()
    print(f"Starting deduplication at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Calculate executor count for partition sizing
    # Get total cores from executors (excluding driver)
    shuffle_partition_count = int(spark.conf.get("spark.sql.shuffle.partitions"))
    print(f"Using {shuffle_partition_count} partitions")

    df_with_partitions_s3_path = f"s3://{S3_BUCKET_TEST_INPUT}/{benchmark_level}/df_with_partitions"
    # Run partition-aware deduplication with optimized parameters for large dataset
    result = partition_aware_deduplicate(
        spark=spark,
        input_df=df_filtered,
        text_column="text",
        similarity_threshold=0.9,  # Higher threshold for URL-based content
        num_hashes=64,  # Fewer hashes for speed
        num_bands=8,  # Fewer bands for speed
        num_partitions=shuffle_partition_count,  # Use executor-based partition count
        ngram=9,
        checkpoint_path=df_with_partitions_s3_path,
    )

    # Collect results
    end_time = time.time()
    elapsed = end_time - start_time

    # Performance metrics - collect efficiently to avoid OOM
    print("Collecting performance metrics...")

    total_docs = result.count()
    print(f"Total docs counted: {total_docs:,}")

    duplicate_docs = result.filter(col("is_duplicate")).count()
    print(f"Duplicate docs counted: {duplicate_docs:,}")

    unique_docs = total_docs - duplicate_docs
    print(f"Unique docs calculated: {unique_docs:,}")

    print("\n" + "=" * 80)
    print("COMMON CRAWL STRESS TEST RESULTS")
    print("=" * 80)
    print(f"Processing time: {elapsed:.2f} seconds")
    print(f"Documents processed: {total_docs:,}")
    print(f"Duplicates found: {duplicate_docs:,}")
    print(f"Unique documents: {unique_docs:,}")
    print(f"Deduplication rate: {(duplicate_docs / total_docs * 100):.2f}%")
    print("=" * 80)

    # Write results to S3 for cluster mode compatibility
    deploy_mode = spark.conf.get("spark.submit.deployMode", "cluster")
    print(f"Deploy mode: {deploy_mode}")

    if deploy_mode == "cluster":
        print("Cluster mode detected - saving results to S3...")

        result_summary = {
            "benchmark_level": benchmark_level,
            "processing_time_seconds": elapsed,
            "total_documents": total_docs,
            "duplicate_documents": duplicate_docs,
            "unique_documents": unique_docs,
            "deduplication_rate_percent": round(duplicate_docs / total_docs * 100, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "similarity_threshold": 0.9,
                "num_hashes": 64,
                "num_bands": 8,
                "num_partitions": shuffle_partition_count,
            },
        }

        import json

        result_json = json.dumps(result_summary, indent=2)

        # Save to S3 for retrieval after job completes
        result_df = spark.createDataFrame([(result_json,)], ["result"])
        result_df.write.mode("overwrite").text("s3://text-deduplication-740959772378/results/benchmark_results/")
        print("Results saved to S3: s3://text-deduplication-740959772378/results/benchmark_results/")

    else:
        print("Client mode - results displayed above")

    print("=" * 80)
    spark.stop()


if __name__ == "__main__":
    import sys

    # Default benchmark level
    benchmark_level = "development"

    # Parse command line arguments
    if len(sys.argv) > 1:
        benchmark_level = sys.argv[1]
        print(f"Using benchmark level from command line: {benchmark_level}")
    else:
        print(f"No benchmark level specified, using default: {benchmark_level}")
        print("Available levels: development, validation, production_proof, scale_proof, full_corpus")
        print("Usage: python script.py <benchmark_level>")
        print("   or: spark-submit script.py <benchmark_level>")

    test_integration_commoncrawl_sample(benchmark_level)
