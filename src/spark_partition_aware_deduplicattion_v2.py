# spark_partition_aware_deduplicattion_v2.py - Scalable partition-aware MinHash LSH implementation
from __future__ import annotations

import logging

import pandas as pd
import pyspark.sql.functions as F
from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import ArrayType, IntegerType, LongType, StringType, StructField, StructType

from two_phase_partition_aware_union_find import run_phase1_local_union_find, run_phase2_global_transitivity_closure

try:
    from .spark_utils import does_file_exists, read_parquet_from_s3, set_spark_context, upload_df_to_s3
except Exception:
    from spark_utils import does_file_exists, read_parquet_from_s3, set_spark_context, upload_df_to_s3
from shingle_hash_wrapper import compute_minhash_cython_batch
from udf import compute_minhash_vectorized_batch_only_hash_once

# Import Python's built-in functions before PySpark overwrites them
builtin_sum = sum
builtin_min = min
builtin_max = max

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def partition_aware_deduplicate(
    spark: SparkSession,
    input_df: DataFrame,
    text_column: str = "text",
    similarity_threshold: float = 0.8,
    num_hashes: int = 64,
    num_bands: int = 16,
    num_partitions: int = 1000,
    ngram: int = 9,
    is_debug_mode: bool = False,
    df_with_partitions_s3_path: str | None = None,
    remove_articles: bool = False,
    use_python_udf_min_hash: bool = False,
) -> DataFrame:
    """
    Partition-aware deduplication that scales to 1TB+

    Key innovations:
    1. Documents are assigned to specific partitions based on their LSH bands
    2. Similar documents are co-located in the same partition
    3. Comparisons happen locally within partitions (no shuffle)
    4. Linear memory scaling instead of quadratic

    Args:
        spark: SparkSession
        input_df: Input DataFrame with documents
        text_column: Name of text column
        similarity_threshold: Similarity threshold for duplicates
        num_hashes: Number of MinHash functions
        num_bands: Number of LSH bands
        num_partitions: Number of partitions for processing
        use_python_udf_min_hash: if True, call python UDF, else call Cython UDF.

    Returns:
        DataFrame with duplicates marked
    """

    logger.info("Starting PARTITION-AWARE deduplication...")
    logger.info(
        f"Parameters: threshold={similarity_threshold}, hashes={num_hashes}, "
        f"bands={num_bands}, partitions={num_partitions}"
    )
    logger.info(f"Spark UI available at: {spark.sparkContext.uiWebUrl}")
    print(f"🚀 Spark UI: {spark.sparkContext.uiWebUrl}")  # Print to console for visibility
    logger.info(f"Using {'Python UDF' if use_python_udf_min_hash else 'Cython UDF'} for MinHash")

    rows_per_band = num_hashes // num_bands

    # Step 1: Compute MinHash signatures
    logger.info("Step 1: Computing MinHash signatures...")
    set_spark_context(
        spark,
        "Step 1: MinHash Signatures",
        f"Computing MinHash signatures for {similarity_threshold} similarity threshold",
    )

    if df_with_partitions_s3_path is None or not does_file_exists(df_with_partitions_s3_path):
        # Get partition count from Spark config
        num_shuffle_partitions = int(spark.conf.get("spark.sql.shuffle.partitions", "1000"))
        input_df = input_df.repartition(num_shuffle_partitions)

        @F.pandas_udf(ArrayType(LongType()))  # LongType because Step4 expects Long
        def minhash_batch_udf(rows: pd.Series) -> pd.Series:
            """Process entire batch using highly optimized vectorized operations"""
            if use_python_udf_min_hash:
                return compute_minhash_vectorized_batch_only_hash_once(
                    texts=rows, num_hashes=num_hashes, ngram=ngram, remove_articles=remove_articles
                )
            else:
                return compute_minhash_cython_batch(rows, num_hashes, ngram=ngram)

        # If users really need scala UDF (not recommended)
        # spark._jvm.com.minhash.MinHashUDF.registerUdf(spark._jsparkSession)

        df_with_signatures = input_df.withColumn(
            "minhash_signature",
            minhash_batch_udf(F.col(text_column)),
            # F.expr(f"compute_minhash({text_column}, {num_hashes}, {ngram}, {str(remove_articles).lower()})") # scala UDF
        ).cache()

        total_docs_count = df_with_signatures.count()
        logger.info(f"Processing {total_docs_count} documents...")

        # Step 2: Compute partition assignments based on LSH bands
        logger.info("Step 2: Computing partition assignments (KEY INNOVATION)...")
        set_spark_context(
            spark,
            "Step 2: Partition Assignments",
            f"Computing partition assignments with {num_bands} LSH bands and {num_partitions} partitions",
        )

        spark._jvm.com.partitionAssignment.ComputePartitionAssignmentsUDF.registerUdf(spark._jsparkSession)

        df_with_partitions = df_with_signatures.withColumn(
            "target_partitions",
            F.expr(f"compute_partition_assignments(minhash_signature, {num_bands}, {rows_per_band}, {num_partitions})"),
        ).select(
            # Drop text column - not needed for deduplication, reduces cache size and memory
            F.col("doc_id"),
            F.col("minhash_signature"),  # 128 MinHash samples
            F.col("target_partitions"),  # Array of band hash from 8 MinHash % partition count
        )

        if is_debug_mode and df_with_partitions_s3_path:
            upload_df_to_s3(df=df_with_partitions, s3_path=df_with_partitions_s3_path, row_count=total_docs_count)
    else:
        # Define schema for df_with_partitions to avoid inference issues
        # Note: text column not included - not needed for deduplication
        set_spark_context(
            spark,
            "Loading Cached Data",
            f"Loading pre-computed signatures and partitions from {df_with_partitions_s3_path}",
        )
        df_with_partitions_schema = StructType(
            [
                StructField("doc_id", StringType(), True),
                StructField(
                    "minhash_signature", ArrayType(LongType()), True
                ),  # changed from Int to Long because scala mh3 is signed int, we need unsigned.
                StructField("target_partitions", ArrayType(IntegerType()), True),
            ]
        )

        df_with_partitions = read_parquet_from_s3(
            s3_path=df_with_partitions_s3_path, spark=spark, schema=df_with_partitions_schema
        )

        # Set total_docs_count for cached data path
        total_docs_count = df_with_partitions.count()
        logger.info(f"Loaded {total_docs_count} documents from cache...")

    # Show partition distribution for monitoring
    partition_stats = (
        df_with_partitions.select(F.size(F.col("target_partitions")).alias("num_partitions_per_doc"))
        .agg(
            F.avg("num_partitions_per_doc").alias("avg_partitions"),
            F.min("num_partitions_per_doc").alias("min_partitions"),
            F.max("num_partitions_per_doc").alias("max_partitions"),
        )
        .collect()[0]
    )

    logger.info(
        f"Partition assignment stats - Avg: {partition_stats['avg_partitions']:.2f}, "
        f"Min: {partition_stats['min_partitions']}, Max: {partition_stats['max_partitions']}"
    )

    # Step 3: Explode and repartition - documents go to their assigned partitions
    logger.info("Step 3: Smart partitioning - co-locating similar documents...")
    set_spark_context(
        spark, "Step 3: Smart Partitioning", f"Co-locating similar documents across {num_partitions} partitions"
    )

    df_exploded = df_with_partitions.select(
        F.col("doc_id"),
        # F.col(text_column),  # Removed - not needed for similarity, reduces shuffle size
        F.col("minhash_signature"),
        F.explode(F.col("target_partitions")).alias("partition_id"),
    )

    # Monitor partition skew before repartitioning
    partition_distribution = df_exploded.groupBy("partition_id").count().collect()
    partition_counts = [row["count"] for row in partition_distribution]

    max_partition_size = builtin_max(partition_counts)
    min_partition_size = builtin_min(partition_counts)
    avg_partition_size = builtin_sum(partition_counts) / len(partition_counts)
    skew_ratio = max_partition_size / avg_partition_size if avg_partition_size > 0 else 0

    logger.info("Partition skew analysis:")
    logger.info(f"  Max partition size: {max_partition_size:,}")
    logger.info(f"  Min partition size: {min_partition_size:,}")
    logger.info(f"  Avg partition size: {avg_partition_size:.0f}")
    logger.info(f"  Skew ratio (max/avg): {skew_ratio:.2f}")
    logger.info(f"  Active partitions: {len(partition_counts)} / {num_partitions}")

    # KEY INNOVATION: Repartition based on computed partition assignments
    # This ensures similar documents are in the same partition
    df_partitioned = df_exploded.repartition(num_partitions, F.col("partition_id"))

    # Step 4: Process each partition locally (no shuffle!)
    logger.info("Step 4: Local deduplication within partitions (NO SHUFFLE)...")
    set_spark_context(
        spark, "Step 4: Local Deduplication", f"Finding similar pairs within partitions using {num_bands} bands"
    )

    # Process partitions and find similar pairs
    jvm_helper = spark._jvm.com.processPartitionLocally.ProcessPartitionLocallyUDF
    similar_pairs_jdf = jvm_helper.processPartitions(
        df_partitioned._jdf,  # Pass the underlying JVM DataFrame
        num_bands,
        rows_per_band,
        similarity_threshold,
    )
    similar_pairs_df = DataFrame(similar_pairs_jdf, spark)

    # PySpark version: Convert back to DataFrame
    """
    similar_pairs_schema = StructType([
        StructField("doc1", StringType(), False),
        StructField("doc2", StringType(), False),
        StructField("similarity", FloatType(), False),
        StructField("partition_id", IntegerType(), False)
    ])
    from udf import process_partition_locally
    # mapPartitions acccepts only one function pointer, so either pass process_partition_locally
    # or pass lambda and use other params.
    similar_pairs_rdd = df_partitioned.rdd.mapPartitions(lambda iterator:
            process_partition_locally,
            num_bands,
            rows_per_band,
            similarity_threshold
         )
    similar_pairs_df = spark.createDataFrame(similar_pairs_rdd, similar_pairs_schema)
    """

    # Don't drop dups because dropDuplicates triggers a shuffle that destroys your partition layout:
    # similar_pairs_df = similar_pairs_df.dropDuplicates(["doc1", "doc2"]).persist(StorageLevel.MEMORY_AND_DISK)
    similar_pairs_df = similar_pairs_df.persist(StorageLevel.MEMORY_AND_DISK)

    similar_count = similar_pairs_df.count()
    logger.info(f"Found {similar_count} similar document pairs")

    # Step 5: Build connected components for duplicate groups
    logger.info("Step 5: Build connected components. For each distinct doc_id, it has representative doc_id")
    set_spark_context(spark, "Step 5 Phase 1", "Partition-local Union-Find (no shuffle)")

    vertices = input_df.select(F.col("doc_id").alias("id")).distinct().persist(StorageLevel.MEMORY_ONLY)
    # doc_id_and_representative_doc_id_df_deduped = get_deduplicate_df_graphframes(
    #     spark=spark, similar_pairs_df=similar_pairs_df, vertices=vertices
    # ).persist(StorageLevel.MEMORY_AND_DISK)

    # if use_scala_phase1:

    # else:
    local_results = run_phase1_local_union_find(similar_pairs_df=similar_pairs_df).persist(StorageLevel.MEMORY_AND_DISK)
    local_count = local_results.count()
    logger.info(f"Phase 1 complete: {local_count} doc -> representative mappings")

    set_spark_context(spark, "Step 5 Phase 2", "Cross-partition component merge")
    doc_id_and_representative_doc_id_df_deduped = run_phase2_global_transitivity_closure(
        spark=spark, local_results=local_results, vertices=vertices, max_iterations=50
    ).persist(StorageLevel.MEMORY_AND_DISK)
    doc_id_and_representative_doc_id_df_count = doc_id_and_representative_doc_id_df_deduped.count()
    logger.info(
        f"Phase 2 complete: {doc_id_and_representative_doc_id_df_count} doc→representative mappings (includes singletons)"
    )
    logger.info("vertices cached. doc_id_and_representative_doc_id_df_deduped cached.")
    local_results.unpersist()

    # Step 6: Join back with original data
    logger.info("Step 6: Marking duplicates...")
    set_spark_context(
        spark, "Step 6: Mark Duplicates", f"Joining deduplication results with original {total_docs_count} documents"
    )

    result = (
        input_df.join(doc_id_and_representative_doc_id_df_deduped, on="doc_id", how="left")
        .withColumn(
            "representative_id",
            F.when(F.col("representative_id").isNull(), F.col("doc_id")).otherwise(F.col("representative_id")),
        )
        .withColumn("is_duplicate", F.col("representative_id") != F.col("doc_id"))
    ).persist(StorageLevel.MEMORY_AND_DISK)

    # Compute statistics
    deduplicate_docs = result.filter(~F.col("is_duplicate"))
    unique_docs_count = deduplicate_docs.count()
    duplicate_docs_count = total_docs_count - unique_docs_count

    logger.info("\n" + "=" * 60)
    logger.info("PARTITION-AWARE DEDUPLICATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total documents: {total_docs_count:,}")
    logger.info(f"Duplicate documents: {duplicate_docs_count:,}")
    logger.info(f"Unique documents: {unique_docs_count:,}")
    logger.info(f"Deduplication rate: {duplicate_docs_count / total_docs_count * 100:.2f}%")
    logger.info("=" * 60)

    # Clean up cached DataFrames to free memory
    logger.info("Cleaning up cached DataFrames...")
    set_spark_context(spark, "Cleanup", "Unpersisting cached DataFrames to free memory")
    # Only unpersist df_with_signatures if it was created (not from cache)
    if "df_with_signatures" in locals():
        df_with_signatures.unpersist()
    similar_pairs_df.unpersist()
    vertices.unpersist()
    doc_id_and_representative_doc_id_df_deduped.unpersist()

    # Don't unpersist result here because downstream caller function can trigger re-compute.

    return result
