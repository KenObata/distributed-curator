import os

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from src.spark_partition_aware_deduplicattion_v2 import partition_aware_deduplicate

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
