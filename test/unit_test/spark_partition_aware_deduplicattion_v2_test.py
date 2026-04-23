import os

import pytest
from pyspark.sql.functions import col

from src.spark_partition_aware_deduplicattion_v2 import (
    apply_deterministic_salting,
    partition_aware_deduplicate,
)

# Resolve JAR paths relative to repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCALA_UDF_JAR = os.path.join(
    REPO_ROOT,
    "target",
    "scala-2.12",
    "minhash-udf_2.12-0.1.jar",  # adjust to your actual JAR name
)

# Skip entire module if JARs are missing
pytestmark = pytest.mark.skipif(
    os.path.exists(SCALA_UDF_JAR),
    reason=f"Required JARs not found. Run 'sbt package' first.\n"
    f"  Scala UDF:   {SCALA_UDF_JAR} (exists={os.path.exists(SCALA_UDF_JAR)})",
)


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
