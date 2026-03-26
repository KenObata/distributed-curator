import os
import time
from collections.abc import Iterable, Iterator

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, length

try:
    from spark_partition_aware_deduplicattion_v2 import partition_aware_deduplicate
except ImportError:
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
    from spark_partition_aware_deduplicattion_v2 import partition_aware_deduplicate
import os

import boto3

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


def parse_wet_record_v2(lines: Iterable[str]) -> Iterator[tuple[str, str]]:
    """
    Args:
    - lines: Iterable. Caller's each row.value is a str. But lines receives the generator of many lines.

    Retuns:
    - Iterator[tuple[str, str]

    Parsing based on WET file format logic:
    - From WARC-Target-URI:, fetch URL. lines without URL is skipped.
    - While URL is set, when blank line is found, flag=True for IN_CONTENT.
    - While URL is set and IN_CONTENT is True, then accumulate contents.
    - when WARC-Type: keyword is found, reset

    ex) aws s3 cp s3://commoncrawl/crawl-data/CC-MAIN-2024-22/segments/1715971057216.39/wet
        /CC-MAIN-20240517233122-20240518023122-00001.warc.wet.gz - | gunzip | head -100

    WARC/1.0
    WARC-Type: warcinfo                                                    <- record 1 (metadata, no URL)
    WARC-Date: 2024-05-31T01:24:21Z
    WARC-Filename: CC-MAIN-20240517233122-20240518023122-00001.warc.wet.gz
    WARC-Record-ID: <urn:uuid:be47515c-d86b-4ce6-82f6-8c3c11086d3c>
    Content-Type: application/warc-fields
    Content-Length: 368

    Software-Info: ia-web-commons.1.1.10-SNAPSHOT-20240513074037
    Extracted-Date: Fri, 31 May 2024 01:24:21 GMT
    robots: checked via crawler-commons 1.5-SNAPSHOT (https://github.com/crawler-commons/crawler-commons)
    isPartOf: CC-MAIN-2024-22
    operator: Common Crawl Admin (info@commoncrawl.org)
    description: Wide crawl of the web for May 2024
    publisher: Common Crawl



    WARC/1.0
    WARC-Type: conversion
    WARC-Target-URI: http://0-50.ru/news/line/2013-03-26/id_30926.html    <- record 2 (get URL)
    WARC-Date: 2024-05-18T01:05:37Z
    WARC-Record-ID: <urn:uuid:6e69bc67-a141-4cc8-b949-a3a9b647d87c>
    WARC-Refers-To: <urn:uuid:a25e1f4e-42d7-4e2f-8a77-0f8645af7b2c>
    WARC-Block-Digest: sha1:FCFS6CEWGSGDH4JZJLWRSXOKZ4JUFI52
    WARC-Identified-Content-Language: rus
    Content-Type: text/plain
    Content-Length: 17217
                                                                          <- blank line = content starts
    (document contents start)
    """
    import io

    current_url = None
    content_buffer = io.StringIO()
    in_content = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("WARC-Type:"):
            # WARC-Type is the start of new doc and finalize previous record
            if current_url and content_buffer.tell() > 0:
                content_buffer.seek(0)
                text = content_buffer.read().strip()
                if len(text) > 50:  # Filter short content
                    # With yield, this function becomes a generator (each record
                    # flows downstream immediately, then gets garbage collected).
                    # We use mapPartition as lazy eval.
                    yield (current_url, text)

            # Reset state
            current_url = None
            content_buffer.seek(0)
            content_buffer.truncate(0)
            in_content = False

        elif stripped.startswith("WARC-Target-URI:"):
            # Extract URL
            current_url = stripped[17:].strip()  # Skip "WARC-Target-URI: "

        elif stripped == "" and current_url:
            # Blank line after headers means content starts next
            in_content = True

        elif in_content and stripped:
            # it means we are at contents lines - use StringIO for efficient text accumulation
            content_buffer.write(stripped)
            content_buffer.write(" ")

    # Handle last record
    if current_url and content_buffer.tell() > 0:
        content_buffer.seek(0)
        text = content_buffer.read().strip()
        if len(text) > 50:
            yield (current_url, text)


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


def get_wet_file_paths(max_files: int, cc_main_id: str) -> list[str]:
    """Get WET file paths from multiple segments
    S3 directory structure:
    - Each month contains 100 segments.
    - Each segment contains 900 WET files
    - total 900 * 100 = 90k WET files.
    one segment contains 900 .gz WET files.
    ex) s3://commoncrawl/crawl-data/CC-MAIN-2024-22/segments/
        - 1715971057216.39/
            - warc/
            - wet/
                -   CC-MAIN-20240517233122-20240518023122-00899.warc.wet.gz
                - ...
        - other segments ID/

    """
    import boto3

    s3_client = boto3.client("s3")

    bucket = "commoncrawl"
    # List ALL segments, not just one
    prefix = f"crawl-data/{cc_main_id}/segments/"

    file_paths = []
    paginator = s3_client.get_paginator("list_objects_v2")

    # Iterate through segments
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        if "CommonPrefixes" not in page:
            continue

        for segment in page["CommonPrefixes"]:
            segment_prefix = segment["Prefix"] + "wet/"

            # List WET files in this segment. default MaxKeys=1000 > 900 WET files per segment.
            response = s3_client.list_objects_v2(
                Bucket=bucket,
                Prefix=segment_prefix,
            )

            if "Contents" in response:
                for obj in response["Contents"]:
                    if obj["Key"].endswith(".warc.wet.gz"):
                        file_paths.append(f"s3://{bucket}/{obj['Key']}")

                        if len(file_paths) >= max_files:
                            print(f"Collected {len(file_paths)} WET file paths")
                            return file_paths

    print(f"Collected {len(file_paths)} WET file paths")
    return file_paths


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
        spark = create_spark_session_partition_aware_emr("CommonCrawlStressTest")
    else:
        print("Running locally - using local Spark session with S3 support")
        # os.environ["_JAVA_OPTIONS"] = "--add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/sun.misc=ALL-UNNAMED"
        os.environ["PYSPARK_SUBMIT_ARGS"] = (
            "--driver-java-options '--add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/sun.misc=ALL-UNNAMED' pyspark-shell"
        )

        # Try to find GraphFrames JAR locally
        jar_path = os.path.join(os.getcwd(), "graphframes-0.8.3-spark3.5-s_2.12.jar")
        if not os.path.exists(jar_path):
            # Try relative path from test directory
            jar_path = os.path.join(os.path.dirname(__file__), "../../terraform/graphframes-0.8.3-spark3.5-s_2.12.jar")

        if os.path.exists(jar_path):
            print(f"Found GraphFrames JAR at: {jar_path}")
            spark = create_spark_session_partition_aware("CommonCrawlStressTest", graphframes_jar_path=jar_path)
        else:
            print("GraphFrames JAR not found - using basic Spark session")
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
            df_filtered = df_parsed.filter(col("text").isNotNull() & (length(col("text")) > 100))

        except Exception as e:
            print("Common Crawl access requires AWS credentials or has connectivity issues.")
            raise Exception(f"Error reading WET files: {e!s}") from e

    # Cache for performance - runs regardless of branch above
    df_filtered = df_filtered.cache()
    test_count = df_filtered.count()
    print(f"Test dataset size: {test_count:,} documents")
    if not has_common_crawl_df_filtered:
        upload_df_to_s3(df=df_filtered, s3_path=s3_path, row_count=test_count)

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
        is_debug_mode=True,
        df_with_partitions_s3_path=df_with_partitions_s3_path,
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
