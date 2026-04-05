import os

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, spark_partition_id

from src.spark_partition_aware_deduplicattion_v2 import (
    apply_deterministic_salting,
    identity_repartition,
    partition_aware_deduplicate,
)

# Resolve JAR paths relative to repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GRAPHFRAMES_JAR = os.path.join(REPO_ROOT, "lib", "graphframes-0.8.3-spark3.5-s_2.12.jar")
SCALA_UDF_JAR = os.path.join(
    REPO_ROOT,
    "target",
    "scala-2.12",
    "minhash-udf_2.12-0.1.jar",  # adjust to your actual JAR name
)

# Skip entire module if JARs are missing
pytestmark = pytest.mark.skipif(
    not os.path.exists(GRAPHFRAMES_JAR) or not os.path.exists(SCALA_UDF_JAR),
    reason=f"Required JARs not found. Run 'sbt package' first.\n"
    f"  GraphFrames: {GRAPHFRAMES_JAR} (exists={os.path.exists(GRAPHFRAMES_JAR)})\n"
    f"  Scala UDF:   {SCALA_UDF_JAR} (exists={os.path.exists(SCALA_UDF_JAR)})",
)


@pytest.fixture(scope="session")
def spark():
    """Create SparkSession with required JARs for integration testing"""
    jars = f"{GRAPHFRAMES_JAR},{SCALA_UDF_JAR}"
    spark = (
        SparkSession.builder.appName("TestPartitionAwareDedup")
        .master("local[2]")
        .config("spark.jars", jars)
        .config("spark.driver.extraClassPath", jars)
        .config("spark.sql.shuffle.partitions", "10")
        .getOrCreate()
    )
    yield spark
    spark.stop()


class TestPartitionAwareDeduplication:
    """Integration tests for the full deduplication pipeline"""

    def test_exact_duplicates_detected(self, spark):
        """Exact duplicate documents should be identified"""
        data = [
            ("doc1", "The quick brown fox jumps over the lazy dog and runs away fast."),
            ("doc2", "The quick brown fox jumps over the lazy dog and runs away fast."),
            ("doc3", "A completely different document about machine learning and data science."),
        ]
        df = spark.createDataFrame(data, ["doc_id", "text"])

        result = partition_aware_deduplicate(
            spark,
            df,
            similarity_threshold=0.9,
            num_hashes=64,
            num_bands=8,
            num_partitions=5,
            use_python_udf_min_hash=True,
        )

        duplicates = result.filter(col("is_duplicate")).count()
        assert duplicates == 1

        # The duplicate should share a representative with its match
        reps = result.filter(col("doc_id").isin(["doc1", "doc2"])).select("representative_id").distinct().count()
        assert reps == 1, "Exact duplicates should share the same representative"

    def test_different_documents_not_marked_duplicate(self, spark):
        """Completely different documents should not be marked as duplicates"""
        data = [
            ("doc1", "The quick brown fox jumps over the lazy dog and runs away fast."),
            ("doc2", "Machine learning models require large datasets for training purposes."),
            ("doc3", "The weather forecast predicts sunny skies and warm temperatures today."),
        ]
        df = spark.createDataFrame(data, ["doc_id", "text"])

        result = partition_aware_deduplicate(
            spark,
            df,
            similarity_threshold=0.8,
            num_hashes=64,
            num_bands=8,
            num_partitions=5,
            use_python_udf_min_hash=True,
        )

        duplicates = result.filter(col("is_duplicate")).count()
        assert duplicates == 0, "Completely different documents should not be duplicates"

    def test_groups_multiple_near_duplicates(self, spark):
        """Multiple near-duplicate documents should share the same representative"""
        data = [
            ("doc1", "The quick brown fox jumps over the lazy dog and runs away fast."),
            ("doc2", "The quick brown fox jumps over the lazy dog and runs away fast!"),
            ("doc3", "The quick brown fox jumps over the lazy dog and runs away fast"),
            ("doc4", "A completely different document about machine learning and data science."),
        ]
        df = spark.createDataFrame(data, ["doc_id", "text"])

        result = partition_aware_deduplicate(
            spark,
            df,
            similarity_threshold=0.8,
            num_hashes=128,
            num_bands=16,
            num_partitions=5,
            use_python_udf_min_hash=True,
        )

        # doc1, doc2, doc3 should form one group; doc4 standalone
        unique = result.filter(~col("is_duplicate")).count()
        assert unique == 2, f"Expected 2 unique docs (1 group of 3 + 1 standalone), got {unique}"

        # Verify the near-duplicates share a representative
        reps = (
            result.filter(col("doc_id").isin(["doc1", "doc2", "doc3"])).select("representative_id").distinct().count()
        )
        assert reps == 1, "Near-duplicate group should have exactly one representative"

    def test_result_schema(self, spark):
        """Result should have expected columns"""
        data = [
            ("doc1", "The quick brown fox jumps over the lazy dog and runs away fast."),
            ("doc2", "A completely different document about machine learning and data science."),
        ]
        df = spark.createDataFrame(data, ["doc_id", "text"])

        result = partition_aware_deduplicate(
            spark,
            df,
            similarity_threshold=0.9,
            num_hashes=64,
            num_bands=8,
            num_partitions=5,
            use_python_udf_min_hash=True,
        )

        result_columns = set(result.columns)
        assert "doc_id" in result_columns
        assert "representative_id" in result_columns
        assert "is_duplicate" in result_columns
        assert "text" in result_columns


class TestDeterministicSalting:
    # ===============================================================
    # Test for deterministic_salting() in Step3
    # ===============================================================
    def test_cold_partitions_should_be_unchanged(self, spark):
        data = [
            ("hot_partition_doc1", 42, 54042),
            ("hot_partition_doc2", 42, 54042),
            ("cold_partition_doc1", 100, 99999),  # cold_partitions
        ]
        df_exploded = spark.createDataFrame(data, ["doc_id", "partition_id", "band_hash"])

        df_exploded = apply_deterministic_salting(df_exploded, hot_partition_ids=[42], num_splits=100)
        rows = {row.doc_id: row.partition_id for row in df_exploded.collect()}

        # Cold partition unchanged
        assert rows["cold_partition_doc1"] == 100

    def test_same_band_hash_should_have_same_partition(self, spark):
        data = [
            ("doc1", 42, 54042),
            ("doc2", 42, 54042),
        ]
        df_exploded = spark.createDataFrame(data, ["doc_id", "partition_id", "band_hash"])

        df_exploded = apply_deterministic_salting(df_exploded, hot_partition_ids=[42], num_splits=100)
        rows = {row.doc_id: row.partition_id for row in df_exploded.collect()}

        # Same band_hash should fall under the same salted partition
        assert rows["doc1"] == rows["doc2"]
        # But different from original
        assert rows["doc1"] != 42 or abs(54042) % 100 == 0  # unless salt happens to be 0

    def test_different_band_hash_should_have_different_partition(self, spark):
        data = [
            ("doc1", 42, 42),
            ("doc2", 42, 99),
        ]
        df_exploded = spark.createDataFrame(data, ["doc_id", "partition_id", "band_hash"])

        df_exploded = apply_deterministic_salting(df_exploded, hot_partition_ids=[42], num_splits=100)
        rows = {row.doc_id: row.partition_id for row in df_exploded.collect()}

        # Different band_hash → different salt (with high probability)
        salt1 = abs(42) % 100
        salt2 = abs(99) % 100
        if salt1 != salt2:
            assert rows["doc1"] != rows["doc2"]

    def test_no_hot_partitions_returns_no_change(self, spark):
        data = [
            ("doc1", 42, 54042),
            ("doc2", 100, 99999),
        ]
        df_exploded = spark.createDataFrame(data, ["doc_id", "partition_id", "band_hash"])

        df_exploded = apply_deterministic_salting(df_exploded, hot_partition_ids=[], num_splits=100)
        rows = {row.doc_id: row.partition_id for row in df_exploded.collect()}

        assert rows["doc1"] == 42
        assert rows["doc2"] == 100

    def test_deterministic_salting_is_deterministic(self, spark):
        data = [
            ("doc1", 42, 54042),
            ("doc2", 42, 54042),
        ]
        df_exploded = spark.createDataFrame(data, ["doc_id", "partition_id", "band_hash"])

        result1 = {
            row.doc_id: row.partition_id for row in apply_deterministic_salting(df_exploded, [42], 100).collect()
        }
        result2 = {
            row.doc_id: row.partition_id for row in apply_deterministic_salting(df_exploded, [42], 100).collect()
        }

        assert result1 == result2


class TestIdentityRepartition:
    def test_identity_mapping(self, spark):
        data = []
        num_partition = 100
        for i in range(num_partition):
            data.append([f"doc_{i}", i, [1, 2, 3]])
        df_exploded = spark.createDataFrame(data, ["doc_id", "partition_id", "minhash_signature"])
        df_partitioned = identity_repartition(
            df=df_exploded, repartition_col="partition_id", num_partitions=num_partition
        )

        row_iterator = df_partitioned.withColumn("physical_partition_id", spark_partition_id()).collect()
        for row in row_iterator:
            assert row.physical_partition_id == row.partition_id, (
                "physical_partition_id should be equal to logical partition_id"
            )

    def test_no_collision(self, spark):
        data = []
        num_partition = 100
        for i in range(num_partition):
            data.append([f"doc_{i}", i, [1, 2, 3]])
        df_exploded = spark.createDataFrame(data, ["doc_id", "partition_id", "minhash_signature"])
        df_partitioned = identity_repartition(
            df=df_exploded, repartition_col="partition_id", num_partitions=num_partition
        )

        physical_partition_count = (
            df_partitioned.withColumn("physical_partition", spark_partition_id())
            .select("physical_partition")
            .distinct()
            .count()
        )
        assert physical_partition_count == num_partition

    def test_schema(self, spark):
        """Output schema should match input schema exactly."""
        data = [("doc_1", 0, [1, 2, 3])]
        df_exploded = spark.createDataFrame(data, ["doc_id", "partition_id", "minhash_signature"])
        df_partitioned = identity_repartition(df=df_exploded, repartition_col="partition_id", num_partitions=10)
        assert df_exploded.schema == df_partitioned.schema

    def test_same_partition_id_should_be_colocated(self, spark):
        """
        same partition_id should be assigned the same physical partition_id
        """
        data = [("doc_1", 0, [1, 2, 3]), ("doc_2", 0, [1, 2, 3]), ("doc_3", 1, [1, 2, 3])]
        df_exploded = spark.createDataFrame(data, ["doc_id", "partition_id", "minhash_signature"])
        df_partitioned = identity_repartition(df=df_exploded, repartition_col="partition_id", num_partitions=10)
        row_iterator = df_partitioned.withColumn("physical_partition_id", spark_partition_id()).collect()
        doc_id_to_physical_partition_id = {}
        for row in row_iterator:
            doc_id_to_physical_partition_id[row.doc_id] = row.physical_partition_id
        assert doc_id_to_physical_partition_id["doc_1"] == doc_id_to_physical_partition_id["doc_2"]
        assert doc_id_to_physical_partition_id["doc_1"] != doc_id_to_physical_partition_id["doc_3"]

    def test_salted_partition_ids_beyond_num_partitions(self, spark):
        """
        salting can make logical partition_id > num_parittions as follows:
        F.col("partition_id") + F.abs(F.col("band_hash")) % num_splits

        In identity_repartition, partitionBy is based on
        Utils.nonNegativeMod(key.hashCode, numPartitions) so this shoud wrap via modulo without any errors.
        """
        num_partitions = 10
        data = [
            ("doc_1", 5, [1, 2, 3]),  # normal
            ("doc_2", 15, [1, 2, 3]),  # 15 % 10 = 5, same physical as doc_1
            ("doc_3", 25, [1, 2, 3]),  # 25 % 10 = 5, same physical
            ("doc_4", 8, [1, 2, 3]),  # normal
            ("doc_5", 18, [1, 2, 3]),  # 18 % 10 = 8, same physical as doc_4
        ]
        df_exploded = spark.createDataFrame(data, ["doc_id", "partition_id", "minhash_signature"])
        df_partitioned = identity_repartition(
            df=df_exploded, repartition_col="partition_id", num_partitions=num_partitions
        )
        row_iterator = df_partitioned.withColumn("physical_partition_id", spark_partition_id()).collect()

        for row in row_iterator:
            expected_physical_partition_id = row.partition_id % num_partitions
            assert row.physical_partition_id == expected_physical_partition_id, (
                f"partition_id={row.partition_id} expected physical={expected_physical_partition_id}",
                f"got {row.physical_partition}",
            )
