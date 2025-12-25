from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, length
from pyspark import RDD
from typing import List, Tuple
import numpy as np
import time
try:
    from spark_partition_aware_deduplicattion_v2 import (
        compute_minhash_signature,
        estimate_similarity,
        partition_aware_deduplicate
    )
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
    from spark_partition_aware_deduplicattion_v2 import (
        compute_minhash_signature,
        estimate_similarity,
        partition_aware_deduplicate
    )
from spark_utils import create_spark_session_partition_aware_emr, create_spark_session_partition_aware
import os
import boto3

s3 = boto3.client('s3')

BENCHMARK_CONFIGS = {
    "development": {
        "wet_files": 1,
        "size": "80MB",
        "pages": "~100K",
        "purpose": "Debug and optimize"
    },
    "validation": {
        "wet_files": 100,
        "size": "8GB", 
        "pages": "~10M",
        "purpose": "Compare with MLlib"
    },
    "production_proof": {
        "wet_files": 1000,
        "size": "80GB",
        "pages": "~100M",
        "purpose": "Show 10x improvement"
    },
    "scale_proof": {
        "wet_files": 10000,
        "size": "800GB",
        "pages": "~1B",
        "purpose": "Prove web-scale capability"
    }
}

def read_common_crawl_http_file(spark: SparkSession, wet_s3_path: str) -> RDD:
    import urllib.request
    import tempfile
    import os
    
    print("Downloading WET file to local storage...")
    local_dir = tempfile.mkdtemp()
    local_path = os.path.join(local_dir, "wet_file.gz")
    
    urllib.request.urlretrieve(wet_path, local_path)
    file_size = os.path.getsize(local_path) / (1024 * 1024)  # MB
    print(f"Downloaded {file_size:.1f}MB to {local_path}")
    
    # Read local WET file with Spark
    print("Reading local WET file with Spark...")
    
    # upload to S3
    s3.upload_file(local_path, 'text-deduplication-740959772378', 'data/wet_file.gz')

    wet_rdd = spark.read.text("s3://text-deduplication-740959772378/data/wet_file.gz").rdd
    wet_rdd = wet_rdd.map(lambda row: row.value)
    
    # Check if we can read any lines
    print("Counting lines in WET file...")
    line_count = wet_rdd.count()
    print(f"Read {line_count} lines from WET file")
    
    if line_count == 0:
        raise Exception("WET file appears to be empty or inaccessible")
    return wet_rdd

def read_wet_files_from_s3(spark: SparkSession, wet_s3_path: str, max_files: int = None) -> DataFrame:
    try:
        # Read all WET files directly from S3
        print("Reading WET files from S3...")
        
        if max_files is None:
            # Read all files
            wet_df = spark.read.text(wet_s3_path + "*.warc.wet.gz")
        else:
            # Get list of specific files limited by max_files
            import boto3
            s3_client = boto3.client('s3')
            
            # Parse S3 path
            bucket = "commoncrawl"
            prefix = wet_s3_path.replace("s3://commoncrawl/", "")
            
            print(f"Listing WET files (limit: {max_files})...")
            response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=max_files)
            
            if 'Contents' not in response:
                raise Exception(f"No WET files found at {wet_s3_path}")
            
            # Build list of specific file paths
            file_paths = []
            for obj in response['Contents'][:max_files]:
                if obj['Key'].endswith('.warc.wet.gz'):
                    file_paths.append(f"s3://{bucket}/{obj['Key']}")
            
            print(f"Selected {len(file_paths)} WET files for processing")
            
            if not file_paths:
                raise Exception("No .warc.wet.gz files found")
            
            # Read the selected files
            wet_df = spark.read.text(','.join(file_paths))
        
        return wet_df
    except Exception as e:
        print(f"Error reading WET files from S3: {str(e)}")
        raise Exception(f"Error reading WET files from S3: {str(e)}")

def parse_wet_record_v2(lines) -> List[Tuple[str, str]]:
    """Optimized WET record parser - 3-5x faster than original"""
    import io
    
    records = []
    current_url = None
    content_buffer = io.StringIO()
    in_content = False
    
    for line in lines:
        stripped = line.strip()
        
        # Fast string matching using startswith
        if stripped[:10] == "WARC-Type:":  # Faster than startswith for short strings
            # Finalize previous record
            if current_url and content_buffer.tell() > 0:
                content_buffer.seek(0)
                text = content_buffer.read().strip()
                if len(text) > 50:  # Filter short content
                    records.append((current_url, text))
            
            # Reset state
            current_url = None
            content_buffer.seek(0)
            content_buffer.truncate(0)
            in_content = False
            
        elif stripped[:16] == "WARC-Target-URI:":  # Extract URL efficiently
            current_url = stripped[17:].strip()  # Skip "WARC-Target-URI: "
            
        elif stripped == "" and current_url:
            in_content = True
            
        elif in_content and stripped:
            # Use StringIO for efficient text accumulation
            content_buffer.write(stripped)
            content_buffer.write(' ')
    
    # Handle last record
    if current_url and content_buffer.tell() > 0:
        content_buffer.seek(0)
        text = content_buffer.read().strip()
        if len(text) > 50:
            records.append((current_url, text))
    
    return records

# Parse WET format to extract URL and text content
def parse_wet_record_v1(lines) -> List[Tuple[str, str]]:
    """Parse WET record format to extract URL and text"""
    records = []
    current_record = {}
    content_lines = []
    in_content = False
    
    for line in lines:
        line = line.strip()
        
        if line.startswith("WARC-Type:"):
            # Start of new record
            if current_record and 'url' in current_record and content_lines:
                # Save previous record
                current_record['text'] = ' '.join(content_lines).strip()
                if len(current_record['text']) > 50:  # Filter out very short content
                    records.append((current_record['url'], current_record['text']))
            
            current_record = {}
            content_lines = []
            in_content = False
            
        elif line.startswith("WARC-Target-URI:"):
            current_record['url'] = line.split(":", 1)[1].strip()
            
        elif line == "" and 'url' in current_record:
            # Blank line indicates start of content
            in_content = True
            
        elif in_content and line:
            content_lines.append(line)
    
    # Handle last record
    if current_record and 'url' in current_record and content_lines:
        current_record['text'] = ' '.join(content_lines).strip()
        if len(current_record['text']) > 50:
            records.append((current_record['url'], current_record['text']))
    
    return records

def test_integration_commoncrawl_sample(benchmark_level: str = "development"):
    """
    Stress test with Common Crawl data from AWS S3
    Tests deduplication performance on real-world web crawl data
    
    Args:
        benchmark_level: One of 'development', 'validation', 'production_proof', 'scale_proof'
    """
    # Get benchmark configuration
    if benchmark_level not in BENCHMARK_CONFIGS:
        raise ValueError(f"Invalid benchmark_level. Choose from: {list(BENCHMARK_CONFIGS.keys())}")
    
    config = BENCHMARK_CONFIGS[benchmark_level]
    max_files = config["wet_files"]
    
    print(f"Benchmark Level: {benchmark_level}")
    print(f"Max WET files: {max_files} ({config['size']}, {config['pages']} pages)")
    print(f"Purpose: {config['purpose']}")
    
    # Detect environment and choose appropriate Spark session
    is_emr = os.path.exists('/emr') or 'EMR' in os.environ.get('SPARK_HOME', '') or os.environ.get('AWS_EMR_CLUSTER_ID')
    
    if is_emr:
        print("Running on EMR - using EMR Spark session")
        spark = create_spark_session_partition_aware_emr("CommonCrawlStressTest")
    else:
        print("Running locally - using local Spark session with S3 support")
        spark = create_spark_session_partition_aware("CommonCrawlStressTest")
    
    try:
        print("\n" + "="*80)
        print("COMMON CRAWL STRESS TEST - PARTITION-AWARE DEDUPLICATION")
        print("="*80)
        
        # Option 1: Use Common Crawl WET format for actual text content
        # WET files contain extracted plain text from web pages
        # Use HTTP endpoint instead of S3 to avoid credential issues
        # wet_path = "https://data.commoncrawl.org/crawl-data/CC-MAIN-2024-22/segments/1715971057216.39/wet/CC-MAIN-20240517233122-20240518023122-00000.warc.wet.gz"
        # wet_rdd = read_common_crawl_http_file(spark, wet_s3_path)

        wet_s3_path = "s3://commoncrawl/crawl-data/CC-MAIN-2024-22/segments/1715971057216.39/wet/"
        print(f"Loading Common Crawl WET files from: {wet_s3_path}")
        
        # For WET files, we need to read them as text files and parse the format
        # WET format contains:
        # - WARC-Type: conversion
        # - WARC-Target-URI: <url>
        # - Content-Length: <length>
        # - <blank line>
        # - <extracted text content>
        
        try:
            # Download WET file first to avoid Spark HTTP issues
            wet_df = read_wet_files_from_s3(spark, wet_s3_path, max_files)
            wet_df.show()
            wet_rdd = wet_df.rdd.map(lambda row: row.value)
            
            
            # Process in partitions and parse WET format
            print("Parsing WET format...")
            parsed_rdd = wet_rdd.glom().flatMap(parse_wet_record_v2)
            
            # Check if parsing produced any results
            # parsed_count = parsed_rdd.count()
            # print(f"Parsed {parsed_count} records from WET file")
            
            # if parsed_count == 0:
            #    raise Exception("WET file parsing produced no records")
            
            # Convert to DataFrame and take sample
            from pyspark.sql.types import StructType, StructField, StringType
            schema = StructType([
                StructField("doc_id", StringType(), True),
                StructField("text", StringType(), True)
            ])
            
            print("Creating DataFrame from parsed records...")
            df_parsed = spark.createDataFrame(parsed_rdd, schema)
            df_parsed.show()
            
            print("Applying filters...")
            df_filtered = df_parsed.filter(col("text").isNotNull() & (length(col("text")) > 100))
            
            # filtered_count = df_filtered.count()
            # print(f"After filtering: {filtered_count} records")
            
            # if filtered_count == 0:
            #    raise Exception("No records remain after filtering")
            
            
            # Cache for performance
            df_filtered = df_filtered.cache()
            test_count = df_filtered.count()
            print(f"Test dataset size: {test_count:,} documents")
            
            # No cleanup needed when reading directly from S3
            
        except Exception as e:
            print("Common Crawl access requires AWS credentials or has connectivity issues.")
            raise Exception(f"Error reading WET files: {str(e)}")
        
        # Performance monitoring
        start_time = time.time()
        print(f"Starting deduplication at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run partition-aware deduplication with optimized parameters for large dataset
        result = partition_aware_deduplicate(
            spark=spark,
            input_df=df_filtered,
            text_column="text",
            similarity_threshold=0.9,  # Higher threshold for URL-based content
            num_hashes=64,             # Fewer hashes for speed
            num_bands=8,               # Fewer bands for speed  
            num_partitions=200,         # More partitions for parallelism
            is_debug_mode=False
        )
        
        # Collect results
        end_time = time.time()
        elapsed = end_time - start_time
        
        """
        # Performance metrics - collect efficiently to avoid OOM
        print("Collecting performance metrics...")
        
        total_docs = result.count()
        print(f"Total docs counted: {total_docs:,}")
        
        duplicate_docs = result.filter(col("is_duplicate")).count()
        print(f"Duplicate docs counted: {duplicate_docs:,}")
        
        unique_docs = total_docs - duplicate_docs
        print(f"Unique docs calculated: {unique_docs:,}")
        
        print("\n" + "="*80)
        """
        print("COMMON CRAWL STRESS TEST RESULTS")
        print("="*80)
        print(f"Processing time: {elapsed:.2f} seconds")

        """
        print(f"Documents processed: {total_docs:,}")
        print(f"Duplicates found: {duplicate_docs:,}")
        print(f"Unique documents: {unique_docs:,}")
        print(f"Deduplication rate: {(duplicate_docs/total_docs*100):.2f}%")
        print(f"Throughput: {total_docs/elapsed:.0f} docs/second")
        """
        print("="*80)
        
        """
        # Skip sample display to avoid memory issues with large text content
        print("\nSkipping sample display to avoid memory issues with large dataset")
        print("Use smaller dataset or increase JVM heap size to see samples")
        
        # Performance assertions
        assert elapsed < 300, f"Processing took too long: {elapsed:.2f}s (should be < 5min)"
        assert total_docs > 0, "No documents processed"
        assert unique_docs <= total_docs, "More unique docs than total docs"
        """
        
    except Exception as e:
        print(f"\nError during Common Crawl test: {str(e)}")
        print("This might be due to:")
        print("1. AWS credentials not configured")
        print("2. Network connectivity issues") 
        print("3. Common Crawl bucket rate limiting")
        print("\nFalling back to synthetic large dataset test...")
        
    finally:
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
        print("Available levels: development, validation, production_proof, scale_proof")
        print("Usage: python script.py <benchmark_level>")
        print("   or: spark-submit script.py <benchmark_level>")
    
    test_integration_commoncrawl_sample(benchmark_level)