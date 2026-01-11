from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import Window
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_deduplication_spark_session() -> SparkSession:
    """Create Spark session optimized for deduplication"""
    
    spark = SparkSession.builder \
        .appName("WebScaleDeduplication") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.adaptive.skewJoin.enabled", "true") \
        .config("spark.executor.memory", "8g") \
        .config("spark.executor.cores", "4") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()
    
    # Set log level to reduce verbosity
    spark.sparkContext.setLogLevel("WARN")
    
    return spark

def create_spark_session_partition_aware(app_name: str = "PartitionAwareDedup", graphframes_jar_path: str = None) -> SparkSession:
    """Create optimized Spark session for large-scale deduplication"""
    
    import os
    
    # JARs are now installed in PySpark's jars directory - no need to specify paths
    
    # these are default config, so they can be overriden
    builder = SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.adaptive.skewJoin.enabled", "true") \
        .config("spark.sql.shuffle.partitions", "1000") \
        .config("spark.default.parallelism", "1000") \
        .config("spark.memory.offHeap.enabled", "true") \
        .config("spark.memory.offHeap.size", "2g") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.ui.enabled", "true") \
        .config("spark.ui.port", "4040") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.auth.IAMInstanceCredentialsProvider,com.amazonaws.auth.DefaultAWSCredentialsProviderChain") \
        .config("spark.hadoop.fs.s3.aws.credentials.provider", "org.apache.hadoop.fs.s3a.auth.IAMInstanceCredentialsProvider,com.amazonaws.auth.DefaultAWSCredentialsProviderChain") \
        .config("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com") \
        .config("spark.hadoop.fs.s3a.path.style.access", "false") \
        .config("spark.hadoop.fs.s3a.connection.timeout", "600000") \
        .config("spark.hadoop.fs.s3a.connection.establish.timeout", "60000") \
        .config("spark.hadoop.fs.s3a.attempts.maximum", "3") \
        .config("spark.hadoop.fs.s3a.retry.interval", "1000") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    
    # Add GraphFrames JAR if provided
    if graphframes_jar_path and os.path.exists(graphframes_jar_path):
        builder = builder.config("spark.jars", graphframes_jar_path) \
                        .config("spark.driver.extraClassPath", graphframes_jar_path) \
                        .config("spark.executor.extraClassPath", graphframes_jar_path)
        print(f"✅ GraphFrames JAR configured: {graphframes_jar_path}")
    
    spark = builder.getOrCreate()

    # Set log level to reduce verbosity
    spark.sparkContext.setLogLevel("WARN")
    return spark

def create_spark_session_partition_aware_emr(app_name: str = "PartitionAwareDedup") -> SparkSession:
    """Create optimized Spark session for large-scale deduplication"""
    
    # these are default config, so they can be overriden
    # do not set hadoop jar here because EMR already defines hadoop.
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.adaptive.skewJoin.enabled", "true") \
        .config("spark.default.parallelism", "1000") \
        .config("spark.memory.offHeap.enabled", "true") \
        .config("spark.memory.offHeap.size", "2g") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.ui.enabled", "true") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()

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


def upload_df_to_s3(df: DataFrame, s3_path: str, file_name: str) -> None:
    """
    Upload DataFrame to S3
    Args:
        df: DataFrame to upload
        s3_path: S3 path to upload to
        file_name: File name to upload
    """
    try:
          # Ensure proper path formatting
          if not s3_path.endswith('/'):
              s3_path += '/'

          full_s3_path = s3_path + file_name

          # Upload with error handling
          df.write.mode("overwrite").parquet(full_s3_path)
          print(f"✅ Uploaded DataFrame to S3: {full_s3_path}")

      except Exception as e:
          print(f"❌ Failed to upload DataFrame to S3: {str(e)}")
          raise e
