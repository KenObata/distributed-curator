# test_partition_aware_deduplication.py

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import numpy as np
from src.spark_partition_aware_deduplicattion_v2 import (
    compute_minhash_signature,
    estimate_similarity,
    partition_aware_deduplicate
)

@pytest.fixture(scope="session")
def spark():
    """Create a SparkSession for testing"""
    spark = SparkSession.builder \
        .appName("TestDeduplication") \
        .master("local[2]") \
        .config("spark.sql.shuffle.partitions", "10") \
        .getOrCreate()
    yield spark

    spark.stop()

class TestMinHashSignature:
    """Test MinHash signature generation"""
    
    def test_identical_documents_have_same_signature(self):
        """Identical documents should have identical signatures"""
        text = "The quick brown fox jumps over the lazy dog."
        sig1 = compute_minhash_signature(text, num_hashes=128, k=9)
        sig2 = compute_minhash_signature(text, num_hashes=128, k=9)
        assert sig1 == sig2
    
    def test_empty_text_returns_zero_signature(self):
        """Empty text should return all zeros"""
        sig = compute_minhash_signature("", num_hashes=128, k=9)
        assert all(s == 0 for s in sig)
        assert len(sig) == 128
    
    def test_short_text_returns_zero_signature(self):
        """Text shorter than k should return all zeros"""
        sig = compute_minhash_signature("short", num_hashes=128, k=9)
        assert all(s == 0 for s in sig)

class TestSimilarityEstimation:
    """Test Jaccard similarity estimation"""
    
    def test_identical_signatures_have_similarity_one(self):
        """Identical signatures should have similarity 1.0"""
        sig = [1, 2, 3, 4, 5] * 20  # 100 values
        similarity = estimate_similarity(sig, sig)
        assert similarity == 1.0
    
    def test_completely_different_signatures_have_low_similarity(self):
        """Completely different signatures should have similarity near 0"""
        sig1 = list(range(100))
        sig2 = list(range(100, 200))
        similarity = estimate_similarity(sig1, sig2)
        assert similarity == 0.0
    
    def test_partially_similar_signatures(self):
        """Partially similar signatures should have intermediate similarity"""
        sig1 = [1] * 50 + [2] * 50
        sig2 = [1] * 50 + [3] * 50
        similarity = estimate_similarity(sig1, sig2)
        assert 0.45 < similarity < 0.55  # Should be ~0.5

class TestDocumentSimilarity:
    """Test document similarity with various text variations"""
    
    @pytest.mark.parametrize("text1, text2, k, expected_min, expected_max", [
        # Identical documents
        ("The quick brown fox jumps over the lazy dog.",
         "The quick brown fox jumps over the lazy dog.",
         9, 0.99, 1.0),
        
        # Minor punctuation difference
        ("The quick brown fox jumps over the lazy dog.",
         "The quick brown fox jumps over the lazy dog!",
         9, 0.85, 1.0),
        
        # One word difference ("the" vs "a") - should be high similarity with normalization
        ("The quick brown fox jumps over the lazy dog.",
         "The quick brown fox jumps over a lazy dog.",
         9, 0.85, 1.0),
        
        
        # One word difference ("jumps" vs "leaps")
        ("The quick brown fox jumps over the lazy dog.",
         "The quick brown fox leaps over the lazy dog.",
         9, 0.55, 0.99),
        
        # Completely different documents
        ("The quick brown fox jumps over the lazy dog.",
         "A completely different document about cats.",
         9, 0.0, 0.15),
    ])
    def test_document_similarity_ranges(self, text1, text2, k, expected_min, expected_max):
        """Test that document similarities fall within expected ranges"""
        sig1 = compute_minhash_signature(text1, num_hashes=128, k=k)
        sig2 = compute_minhash_signature(text2, num_hashes=128, k=k)
        similarity = estimate_similarity(sig1, sig2)
        
        assert expected_min <= similarity <= expected_max, \
            f"Similarity {similarity:.3f} not in range [{expected_min}, {expected_max}] for:\n" \
            f"  Text1: {text1}\n" \
            f"  Text2: {text2}\n" \
            f"  k={k}"

class TestPartitionAwareDeduplication:
    """Test the full partition-aware deduplication pipeline"""
    
    def test_deduplication_finds_exact_duplicates(self, spark):
        """Should find exact duplicates"""
        data = [
            ("doc1", "The quick brown fox jumps over the lazy dog."),
            ("doc2", "The quick brown fox jumps over the lazy dog."),
            ("doc3", "A completely different document."),
        ]
        df = spark.createDataFrame(data, ["doc_id", "text"])
        
        result = partition_aware_deduplicate(
            spark, df, similarity_threshold=0.9,
            num_hashes=64, num_bands=8, num_partitions=5
        )
        
        duplicates = result.filter(col("is_duplicate")).count()
        assert duplicates == 1
    
    def test_deduplication_threshold_sensitivity(self, spark):
        """Test that threshold controls sensitivity"""
        data = [
            ("doc1", "The quick brown fox jumps over the lazy dog."),
            ("doc2", "The quick brown fox jumps over a lazy dog."),  # Small difference
            ("doc3", "A completely different document."),
        ]
        df = spark.createDataFrame(data, ["doc_id", "text"])
        
        # High threshold - should not find doc2 as duplicate
        result_high = partition_aware_deduplicate(
            spark, df, similarity_threshold=0.7,
            num_hashes=128, num_bands=16, num_partitions=5
        )
        duplicates_high = result_high.filter(col("is_duplicate")).count()
        assert duplicates_high == 0
        
        # Lower threshold - should find doc2 as duplicate
        result_low = partition_aware_deduplicate(
            spark, df, similarity_threshold=0.6,
            num_hashes=128, num_bands=16, num_partitions=5
        )
        duplicates_low = result_low.filter(col("is_duplicate")).count()
        assert duplicates_low == 1
    
    def test_deduplication_groups_multiple_duplicates(self, spark):
        """Should group multiple similar documents together"""
        data = [
            ("doc1", "The quick brown fox jumps over the lazy dog."),
            ("doc2", "The quick brown fox jumps over the lazy dog!"),
            ("doc3", "The quick brown fox jumps over the lazy dog"),
            ("doc4", "A completely different document."),
        ]
        df = spark.createDataFrame(data, ["doc_id", "text"])
        
        result = partition_aware_deduplicate(
            spark, df, similarity_threshold=0.8,
            num_hashes=128, num_bands=16, num_partitions=5
        )
        
        # Should have 2 unique documents (one group of 3, one standalone)
        unique = result.filter(~col("is_duplicate")).count()
        assert unique == 2
        
        # Check that doc1, doc2, doc3 share the same representative
        representatives = result.filter(
            col("doc_id").isin(["doc1", "doc2", "doc3"])
        ).select("representative_id").distinct().count()
        assert representatives == 1

# Run with: pytest test_partition_aware_deduplication.py -v