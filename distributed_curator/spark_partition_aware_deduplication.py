# spark_partition_aware_deduplication.py - Scalable partition-aware MinHash LSH implementation
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
import logging

import pyspark.sql.functions as F
from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import ArrayType, IntegerType, LongType, StringType, StructField, StructType

from .two_phase_partition_aware_union_find import run_phase2_global_transitivity_closure

try:
    from .spark_utils import (
        does_file_exists,
        get_checkpoint_dir,
        read_parquet_from_s3,
        set_spark_context,
        upload_df_to_s3,
    )
except Exception:
    from spark_utils import (
        does_file_exists,
        get_checkpoint_dir,
        read_parquet_from_s3,
        set_spark_context,
        upload_df_to_s3,
    )
from .driver_memory_diagnostics import capture_heap_histogram, capture_nmt_summary, start_memory_logger
from .shingle_hash_wrapper import compute_minhash_cython_batch

# Import Python's built-in functions before PySpark overwrites them
builtin_sum = sum
builtin_min = min
builtin_max = max

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def apply_deterministic_salting(
    df_exploded: DataFrame,
    hot_partition_ids: list[int],
    num_splits: int,
) -> DataFrame:
    """Apply deterministic salting to hot partitions.

    Same band_hash always maps to the same sub-partition,
    preserving co-location of similar documents while
    splitting hot partitions across multiple physical partitions.

    Args:
    - df_exploded: dataframe that we will apply deterministic salting
    - hot_partition_ids: list of partition_id that has skewed data volume
    - num_splits: number of partitions to split for each hot_partition_ids

    Return:
    - DataFrame
    """
    if not hot_partition_ids:
        logger.info("hot_partition_ids is empty.")
        return df_exploded
    logger.info(f"Applying deterministic salting with {num_splits} splits on {len(hot_partition_ids)} hot partitions")
    df_exploded = df_exploded.withColumn(
        "partition_id",
        F.when(
            F.col("partition_id").isin(hot_partition_ids),
            F.col("partition_id") + F.abs(F.col("band_hash")) % num_splits,  # deterministic salting
        ).otherwise(F.col("partition_id")),
    )
    return df_exploded


def partition_aware_deduplicate(
    spark: SparkSession,
    input_df: DataFrame,
    text_column: str = "text",
    similarity_threshold: float = 0.8,
    num_hashes: int = 64,
    num_bands: int = 16,
    num_partitions: int = 1000,
    ngram: int = 9,
    checkpoint_path: str | None = None,
    enable_diagnostics: bool = False,
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
        checkpoint_path: When checkpoint_path is provided,
          intermediate results get saved there and reused on subsequent runs.
        enable_diagnostics: eabled periodic memory logging.
          (Currently only support driver mem, appears in yarn logs -am 1 stdout)

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

    if enable_diagnostics:
        # Start periodic memory logging (appears in yarn logs -am 1 stdout)
        start_memory_logger(spark.sparkContext, interval_seconds=30)

    rows_per_band = num_hashes // num_bands

    # Step 1: Compute MinHash signatures
    logger.info("Step 1: Computing MinHash signatures...")
    set_spark_context(
        spark,
        "Step 1: MinHash Signatures",
        f"Computing MinHash signatures for {similarity_threshold} similarity threshold",
    )

    df_with_partitions_schema = StructType(
        [
            StructField("doc_id", StringType(), True),
            StructField(
                "minhash_signature", ArrayType(LongType()), True
            ),  # changed from Int to Long because scala mh3 is signed int, we need unsigned.
            StructField("target_partitions", ArrayType(IntegerType()), True),
            StructField("band_hashes", ArrayType(IntegerType()), True),
        ]
    )
    if checkpoint_path is None or not does_file_exists(checkpoint_path):
        # Get partition count from Spark config
        num_shuffle_partitions = int(spark.conf.get("spark.sql.shuffle.partitions", "1000"))
        # input_df = input_df.repartition(num_shuffle_partitions)
        current_partitions = input_df.rdd.getNumPartitions()
        logger.info(f"Input has {current_partitions} partitions (shuffle_partitions config: {num_shuffle_partitions})")

        @F.pandas_udf(ArrayType(LongType()))  # LongType because Step4 expects Long
        def minhash_batch_udf(rows: pd.Series) -> pd.Series:
            """Process entire batch using highly optimized vectorized operations"""
            return compute_minhash_cython_batch(rows, num_hashes, ngram=ngram)

        # If users really need scala UDF (not recommended)
        # spark._jvm.com.minhash.MinHashUDF.registerUdf(spark._jsparkSession)

        df_with_signatures = (
            input_df.withColumn(
                "minhash_signature",
                minhash_batch_udf(F.col(text_column)),
            )
            .drop(text_column)
            .cache()
        )

        total_docs_count = df_with_signatures.count()
        logger.info(f"Processing {total_docs_count} documents...")

        # Step 2: Compute partition assignments based on LSH bands
        logger.info("Step 2: Computing partition assignments (KEY INNOVATION)...")
        set_spark_context(
            spark,
            "Step 2: Partition Assignments",
            f"Computing partition assignments with {num_bands} LSH bands and {num_partitions} partitions",
        )

        # Spark may blindly pass null to the Scala closure with primitive-type argument,
        # and the closure will see the default value of the Java type for the null argument,
        # e.g. `udf((x: Int) => x, IntegerType)`, the result is 0 for null input
        spark.conf.set("spark.sql.legacy.allowUntypedScalaUDF", "true")
        spark._jvm.com.partitionAssignment.ComputePartitionAssignmentsUDF.registerUdf(spark._jsparkSession)

        df_with_partitions = df_with_signatures.withColumn(
            "partition_struct",  # (target_partitions: List[Int], band_hashes: List[Int])
            F.expr(f"compute_partition_assignments(minhash_signature, {num_bands}, {rows_per_band}, {num_partitions})"),
        ).select(
            # Drop text column - not needed for deduplication, reduces cache size and memory
            F.col("doc_id"),
            F.col("minhash_signature"),  # 128 MinHash samples
            F.col("partition_struct.target_partitions").alias(
                "target_partitions"
            ),  # Array of band hash from 8 MinHash % partition count
            F.col("partition_struct.band_hashes").alias("band_hashes"),  # hash from 8 MinHash
        )

        if checkpoint_path:
            upload_df_to_s3(df=df_with_partitions, s3_path=checkpoint_path, row_count=total_docs_count)
            df_with_signatures.unpersist()
            # Break lineage: re-read from S3 instead of recomputing MinHash
            df_with_partitions = read_parquet_from_s3(
                s3_path=checkpoint_path, spark=spark, schema=df_with_partitions_schema
            )
        else:
            checkpoint_dir = get_checkpoint_dir(spark=spark, name="checkpoints_dir")
            spark.sparkContext.setCheckpointDir(checkpoint_dir)
            df_with_partitions = df_with_partitions.checkpoint()
            total_docs_count = df_with_partitions.count()

            df_with_signatures.unpersist()
    else:
        set_spark_context(
            spark,
            "Loading Cached Data",
            f"Loading pre-computed signatures and partitions from {checkpoint_path}",
        )

        df_with_partitions = read_parquet_from_s3(
            s3_path=checkpoint_path, spark=spark, schema=df_with_partitions_schema
        )

        # Set total_docs_count for cached data path
        total_docs_count = df_with_partitions.count()
        logger.info(f"Loaded {total_docs_count} documents from cache...")

    # Step 3: Explode and repartition - documents go to their assigned partitions
    logger.info("Step 3: Smart partitioning - co-locating similar documents...")
    set_spark_context(
        spark, "Step 3: Smart Partitioning", f"Co-locating similar documents across {num_partitions} partitions"
    )

    num_splits = 100

    hot_partition_ids = []
    partition_counts = (
        df_with_partitions.sample(0.01)
        .select(F.explode(F.col("target_partitions")).alias("partition_id"))
        .groupBy("partition_id")
        .count()
    )

    avg_partition_size = partition_counts.agg(F.avg("count")).collect()[0][0]
    if avg_partition_size is not None:
        hot_threshold = int(avg_partition_size * 4)  # 4x average = outlier
        logger.info(f"hot_threshold: {hot_threshold}")

        hot_partitions_iter = partition_counts.filter(F.col("count") > hot_threshold).collect()
        for row in hot_partitions_iter:
            hot_partition_ids.append(row.partition_id)
        logger.info(f"Hot partitions detected: {len(hot_partition_ids)}")

    # Note: we can't persist df_exploded due to limited memory
    df_exploded = df_with_partitions.select(
        F.col("doc_id"),
        # F.col(text_column),  # Removed - not needed for similarity, reduces shuffle size
        F.col("minhash_signature"),
        F.explode(
            # arrays_zip because Spark doesn't allow multiple generators in one query.
            F.arrays_zip(F.col("target_partitions"), F.col("band_hashes"))
        ).alias("zip"),
    ).select(
        F.col("doc_id"),
        F.col("minhash_signature"),
        F.col("zip.target_partitions").alias("partition_id"),
        F.col("zip.band_hashes").alias("band_hash"),
    )

    """
    deterministic salting:
    same band_hashes should get assigned the same partition_id but
    different band_id can have the same band_hashes which we don't want to compare
    That is why partition_id + salting will distinguish even if band_hashes is the same.

    With deterministic salting, partition_id value can be greater than num_partition and that's fine.
    Only step 2 contract is that partition_id value should be withing the num_partition range.
    """
    set_spark_context(spark, "Step 3: apply_deterministic_salting", "Apply deterministic salting to hot partitions")
    df_exploded = apply_deterministic_salting(df_exploded, hot_partition_ids, num_splits)

    # KEY INNOVATION: Repartition based on computed partition assignments
    # This ensures similar documents are in the same partition
    df_exploded = df_exploded.drop(F.col("band_hash"))
    set_spark_context(
        spark, "Step 3: identity_repartition", "Apply identity_repartition for DataFrame by partition_id col"
    )

    jvm_helper = spark._jvm.com.partitionAssignment.IdentityRepartition
    df_partitioned = DataFrame(
        jvm_helper.repartitionFromPython(df_exploded._jdf, "partition_id", num_partitions), spark
    )

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

    # Don't drop dups because dropDuplicates triggers a shuffle that destroys your partition layout:
    # e.g. similar_pairs_df = similar_pairs_df.dropDuplicates(["doc1", "doc2"]).persist(StorageLevel.MEMORY_AND_DISK)
    # Instead, dropDuplicates after Step5 phase 1 end.
    # similar_pairs_df = similar_pairs_df.persist(StorageLevel.MEMORY_AND_DISK)

    # similar_count = similar_pairs_df.count()
    # logger.info(f"Found {similar_count} similar document pairs")

    # Step 5: Build connected components for duplicate groups
    logger.info("Step 5: Build connected components. For each distinct doc_id, it has representative doc_id")
    set_spark_context(spark, "Step 5 Phase 1", "Partition-aware local Union-Find (no shuffle). Dedupe after phase1.")
    vertices = input_df.select(F.col("doc_id").alias("id")).distinct().persist(StorageLevel.DISK_ONLY)
    vertices_count = vertices.count()
    logger.info(f"vertices cached. vertices_count: {vertices_count}")

    local_results_s3_path = None
    if checkpoint_path is not None:
        local_results_s3_path = checkpoint_path.rsplit("/", 1)[0] + "/local_results"
    if local_results_s3_path is not None and does_file_exists(local_results_s3_path):
        set_spark_context(
            spark,
            "Loading Cached Data",
            f"Loading pre-computed signatures and partitions from {local_results_s3_path}",
        )
        local_results_schema = StructType(
            [
                StructField("doc_id", StringType(), True),
                StructField("local_representative", StringType(), True),
            ]
        )

        local_results = read_parquet_from_s3(
            s3_path=local_results_s3_path, spark=spark, schema=local_results_schema
        ).persist(StorageLevel.DISK_ONLY)
        local_count = local_results.count()
    else:
        jvm_helper = spark._jvm.com.unionFind.PartitionAwareUnionFindUDF
        local_results_jdf_and_accumulator_tuple = jvm_helper.runPhase1LocalUnionFind(
            similar_pairs_df._jdf,  # Pass the underlying JVM DataFrame
        )
        local_results_jdf = local_results_jdf_and_accumulator_tuple._1()
        pair_count_accumulator = local_results_jdf_and_accumulator_tuple._2()
        local_results = (
            DataFrame(local_results_jdf, spark)
            .dropDuplicates(["doc_id", "local_representative"])
            .persist(StorageLevel.DISK_ONLY)
        )
        local_count = local_results.count()
        # Accumulator value available after action triggers mapPartitions
        pair_count = pair_count_accumulator.value()
        logger.info(f"Similar pairs processed: {pair_count}")

        if local_results_s3_path:
            try:
                upload_df_to_s3(df=local_results, s3_path=local_results_s3_path, row_count=local_count)
            except Exception as e:
                logger.warning(f"Failed to upload local_results to S3: {e}. Continuing...")

    logger.info(f"Phase 1 complete: {local_count} doc -> representative mappings")
    set_spark_context(spark, "Step 5 Phase 2", "Cross-partition component merge")
    doc_id_and_representative_doc_id_df_deduped = run_phase2_global_transitivity_closure(
        spark=spark, local_results=local_results, vertices=vertices, max_iterations=50
    ).persist(StorageLevel.DISK_ONLY)
    doc_id_and_representative_doc_id_df_count = doc_id_and_representative_doc_id_df_deduped.count()
    logger.info(
        f"Phase 2 complete: {doc_id_and_representative_doc_id_df_count} doc→representative mappings (includes singletons)"
    )
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
    ).persist(StorageLevel.DISK_ONLY)

    # Compute statistics
    deduplicate_docs = result.filter(~F.col("is_duplicate"))
    unique_docs_count = deduplicate_docs.count()
    duplicate_docs_count = total_docs_count - unique_docs_count

    if checkpoint_path:
        results_s3_path = checkpoint_path.rsplit("/", 1)[0] + "/result"
        try:
            upload_df_to_s3(df=result, s3_path=results_s3_path, row_count=total_docs_count)
        except Exception as e:
            logger.warning(f"Failed to upload dataframe result to S3: {e}. Continuing...")

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
    vertices.unpersist()
    doc_id_and_representative_doc_id_df_deduped.unpersist()

    # Don't unpersist result here because downstream caller function can trigger re-compute.

    if enable_diagnostics:
        # capture heap state for driver diagnosis script
        capture_heap_histogram(spark.sparkContext)
        capture_nmt_summary(spark.sparkContext)  # only works if NMT flag is set and application completed successfully

    return result
