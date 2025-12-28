# spark_partition_aware_deduplicattion_v2.py - Scalable partition-aware MinHash LSH implementation

# Import Python's built-in functions before PySpark overwrites them
import builtins
builtin_hash = hash
builtin_sum = sum
builtin_min = min
builtin_max = max
builtin_abs = builtins.abs

from graphframes import GraphFrame
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark import StorageLevel
import mmh3
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Iterator, Set
from collections import defaultdict
import hashlib
import time
import json
import logging
import os
try:
    from .spark_utils import log_dataframe
except:
    from spark_utils import log_dataframe

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Environment detection for optimization strategy
def is_emr() -> bool:
    """Detect if running on EMR vs local environment"""
    # Check multiple EMR indicators
    return (
        os.path.exists('/emr') or 
        'EMR' in os.environ.get('SPARK_HOME', '') or 
        os.environ.get('AWS_EMR_CLUSTER_ID') or
        os.path.exists('/usr/share/aws/emr') or
        'hadoop' in os.environ.get('JAVA_HOME', '').lower()
    )

def generate_shingles_local(text: str, ngram: int) -> List[str]:
    """Local-optimized shingle generation using built-in string operations"""
    if len(text) < ngram:
        return []
    # Built-in string slicing is fastest on local/ARM processors
    return list(set(text[i:i+ngram] for i in range(len(text) - ngram + 1)))

def generate_shingles_emr(text: str, ngram: int) -> List[str]:
    """EMR-optimized shingle generation using NumPy vectorization"""
    if len(text) < ngram:
        return []
    
    try:
        # NumPy vectorized approach for Intel/x86 with more cores
        text_len = len(text)
        text_array = np.array(list(text))
        
        # Create sliding window view
        shingle_indices = np.lib.stride_tricks.sliding_window_view(
            np.arange(text_len), window_shape=ngram
        )
        
        # Vectorized shingle generation
        shingles = [''.join(text_array[indices]) for indices in shingle_indices]
        return list(set(shingles))
    except Exception:
        # Fallback to local method if NumPy fails
        return generate_shingles_local(text, ngram)

# Function pointer based on environment
generate_shingles = generate_shingles_emr if is_emr() else generate_shingles_local

def normalize_text(text: str) -> str:
    """
    Normalize text to reduce impact of minor differences like articles
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text
    """
    import re
    
    # Convert to lowercase
    text_lower = text.lower()
    
    # Remove common articles and determiners that don't affect semantic meaning
    articles = {'the', 'a', 'an', 'this', 'that', 'these', 'those'}
    
    # Split into words, remove articles, rejoin
    filtered_words = []
    words = text_lower.split()
    for word in words:
        # Strip punctuation inline
        clean_word = word.strip('.,!?;:"()[]{}')
        if clean_word and clean_word not in articles:
            filtered_words.append(clean_word)
    
    # If we removed too many words, keep the original to avoid empty text
    if len(filtered_words) < len(words) * 0.3:  # Keep at least 30% of words
        return text
    
    return ' '.join(filtered_words)

@pandas_udf(ArrayType(IntegerType()))
def normalize_text_pandas_udf(texts: pd.Series) -> pd.Series:
    """
    Optimize by separating normalization from hashing.

    Difference from normalize_text is that this function returns
     pandas series unlike normalize_text()'s row by row string.

    Note: currently, we can't use normalize function as udf, because it needs row by row process.
    """
    # Step 1: Normalize all texts first (vectorizable!)
    normalized_texts = texts.str.lower()  # Pandas string methods are optimized
    
    # Remove articles using pandas vectorized operations
    articles_pattern = r'\b(the|a|an|this|that|these|those)\b'
    normalized_texts = normalized_texts.str.replace(articles_pattern, '', regex=True)
    normalized_texts = normalized_texts.str.replace(r'\s+', ' ', regex=True)  # Clean spaces
    
    # Step 2: Now compute MinHash on normalized texts
    results = []
    for text in normalized_texts:
        if text and len(text) >= 9:
            sig = compute_minhash_signature(text, 64, ngram=9, normalize=False)  # Already normalized
        else:
            sig = [0] * 64
        results.append(sig)
    
    return pd.Series(results)

# NOTE: compute_minhash_vectorized_batch() moved to spark_utils_deprecated.py (2025-12-28)
# Current implementation uses compute_minhash_vectorized_batch_only_hash_once() which is more optimized

def compute_minhash_vectorized_batch_only_hash_once(texts: pd.Series, num_hashes: int = 128, ngram: int = 9) -> pd.Series:
    """
    Highly optimized vectorized MinHash computation using pandas and numpy
    Difference from compute_minhash_vectorized_batch is that this function
    hashes the shingle only once and then for different seed i, use different permutation to generate different hash values.
    """
    results = []
    
    # Step 1: Vectorized text normalization using pandas string operations
    normalized_texts = texts.str.lower()
    
    # Remove articles using vectorized regex
    articles_pattern = r'\b(the|a|an|this|that|these|those)\b'
    normalized_texts = normalized_texts.str.replace(articles_pattern, '', regex=True)
    normalized_texts = normalized_texts.str.replace(r'\s+', ' ', regex=True)
    normalized_texts = normalized_texts.str.strip()
    
    # Step 2: Process each text individually (fixed the Series slicing issue)
    for text in normalized_texts:
        if not text or len(text) < ngram:
            results.append([0] * num_hashes)
            continue
        
        # Generate shingles for this specific text string
        shingles = [text[i:i+ngram] for i in range(len(text) - ngram + 1)]
        if not shingles:
            results.append([0] * num_hashes)
            continue
        
        unique_shingles = list(set(shingles))

        # HASH MIXING OPTIMIZATION: Hash each shingle ONCE, then mix with seeds
        base_hashes = np.array([builtin_hash(s) & 0xFFFFFFFF for s in unique_shingles], dtype=np.uint32)
        
        # Use same seeds as the main function for consistency
        if not hasattr(compute_minhash_vectorized_batch_only_hash_once, '_seeds_cache'):
            import random
            random.seed(42)  # Same seed for reproducibility
            compute_minhash_vectorized_batch_only_hash_once._seeds_cache = [
                random.randint(1, 0xFFFFFFFF) for _ in range(1024)
            ]
        
        hash_seeds = np.array(compute_minhash_vectorized_batch_only_hash_once._seeds_cache[:num_hashes], dtype=np.uint32)
        
        # Vectorized hash mixing: broadcast (num_shingles, 1) x (num_hashes,)
        base_hashes_expanded = base_hashes[:, np.newaxis]  # Shape: (num_shingles, 1)
        
        # Apply hash mixing: (base_hash XOR seed) for each combination
        mixed_hashes = (base_hashes_expanded ^ hash_seeds) & 0xFFFFFFFF
        
        # Get minimum across all shingles for each hash function
        signature = np.min(mixed_hashes, axis=0)
        results.append(signature.tolist())
    
    return pd.Series(results)

def compute_minhash_signature(text: str, num_hashes: int = 128, ngram: int = 9, normalize: bool = True) -> List[int]:
    """
    Compute MinHash signature for text with hash mixing optimization
    
    Args:
        text: Input text
        num_hashes: Number of hash functions
        ngram: Shingle size
        normalize: Whether to normalize text first (removes articles, etc.)
    
    Returns:
        MinHash signature
    """
    if not text:
        return [0] * num_hashes
        
    # Optionally normalize text to handle article differences
    if normalize:
        text = normalize_text(text)
    
    if len(text) < ngram:
        return [0] * num_hashes
    
    # PRE-COMPUTE SEEDS ONCE (cached as function attribute for performance)
    if not hasattr(compute_minhash_signature, '_seeds_cache'):
        import random
        random.seed(42)  # Deterministic for reproducibility
        # Generate enough seeds for the maximum expected num_hashes
        compute_minhash_signature._seeds_cache = [
            random.randint(1, 0xFFFFFFFF) for _ in range(1024)  # Pre-generate 1024 seeds
        ]
    
    hash_seeds = compute_minhash_signature._seeds_cache[:num_hashes]
    
    # Create k-shingles using efficient string slicing
    text_lower = text.lower() if not normalize else text  # Already lowercased in normalize
    
    # Use environment-specific shingle generation optimization
    shingles = set(generate_shingles(text_lower, ngram))
    
    if not shingles:
        return [0] * num_hashes
    
    # Compute MinHash signature with hash mixing optimization
    signature = np.full(num_hashes, np.iinfo(np.uint32).max, dtype=np.uint32)
    
    for shingle in shingles:
        # OPTIMIZATION: Hash once, then mix with seeds (eliminates string concatenation)
        base_hash = builtin_hash(shingle) & 0xFFFFFFFF
        for i in range(num_hashes):
            # Hash mixing instead of string concatenation: 2-3x faster
            hash_val = (base_hash ^ hash_seeds[i]) & 0xFFFFFFFF
            signature[i] = builtin_min(signature[i], hash_val)
    
    return signature.tolist()

def estimate_similarity(sig1: List[int], sig2: List[int]) -> float:
    """Estimate Jaccard similarity from MinHash signatures"""
    
    if not sig1 or not sig2 or len(sig1) != len(sig2):
        return 0.0
    
    # Count matching MinHash values
    matches = builtin_sum(1 for h1, h2 in zip(sig1, sig2) if h1 == h2 and h1 != 0)
    
    # Avoid division by zero
    if all(h == 0 for h in sig1) or all(h == 0 for h in sig2):
        return 0.0
    
    return float(matches) / len(sig1)


def get_deduplicate_df_graphframes(spark: SparkSession, similar_pairs_df:DataFrame, vertices:DataFrame) -> DataFrame:
    """
    Use GraphFrames connected components to find duplicate groups.
    
    Args:
        similar_pairs_df: DataFrame with columns (doc1, doc2) representing similar pairs
        all_doc_ids_df: DataFrame with column (doc_id) for all documents
    
    Returns:
        DataFrame with (doc_id, component) where component is the representative doc
    """
    
    # Create edges - bidirectional for undirected graph
    edges_forward = similar_pairs_df.select(
        col("doc1").alias("src"),
        col("doc2").alias("dst")
    )
    edges_backward = similar_pairs_df.select(
        col("doc2").alias("src"),
        col("doc1").alias("dst")
    )
    edges = edges_forward.union(edges_backward).distinct()
    
    # Create graph and run connected components
    g = GraphFrame(vertices, edges)
    
    # Connected components requires checkpoint directory
    # Use HDFS path on EMR for better performance, local path for local development
    if spark.conf.get("spark.master").startswith("yarn"):
        # Running on EMR/YARN - use HDFS (faster for temporary checkpoints)
        checkpoint_dir = "hdfs:///tmp/graphframes-checkpoints"
    else:
        # Running locally - use local filesystem
        checkpoint_dir = "/tmp/graphframes-checkpoints"
    logger.info(f"checkpoint_dir for GraphFrame: {checkpoint_dir}")
    
    spark.sparkContext.setCheckpointDir(checkpoint_dir)
    
    components = g.connectedComponents()
    # Result of components: (id, component) where component is a Long

    # Create temporary view for SQL query
    components.createOrReplaceTempView("components")
    
    sql_command = """
    SELECT component, MIN(id) as representative_id
    FROM components
    GROUP BY component
    """
    component_and_representative_id_df_deduped = spark.sql(sql_command)
    doc_id_and_representative_doc_id_df_deduped = components.join(component_and_representative_id_df_deduped, on="component").select(
        col("id").alias("doc_id"),
        col("representative_id")
    )
    
    # Result has columns: doc_id, representative_id
    return doc_id_and_representative_doc_id_df_deduped

def partition_aware_deduplicate(
    spark: SparkSession,
    input_df: DataFrame,
    text_column: str = "text",
    similarity_threshold: float = 0.8,
    num_hashes: int = 128,
    num_bands: int = 16,
    num_partitions: int = 1000,
    is_debug_mode = False
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
    
    Returns:
        DataFrame with duplicates marked
    """
    
    logger.info(f"Starting PARTITION-AWARE deduplication...")
    logger.info(f"Parameters: threshold={similarity_threshold}, hashes={num_hashes}, "
          f"bands={num_bands}, partitions={num_partitions}")
    logger.info(f"Spark UI available at: {spark.sparkContext.uiWebUrl}")
    print(f"🚀 Spark UI: {spark.sparkContext.uiWebUrl}")  # Print to console for visibility
    logger.info(f"Using {'EMR NumPy' if is_emr() else 'Local built-in'} shingle generation")

    rows_per_band = num_hashes // num_bands
    
    # Step 1: Compute MinHash signatures
    logger.info("Step 1: Computing MinHash signatures...")
    
    # Get partition count from Spark config
    num_shuffle_partitions = int(spark.conf.get("spark.sql.shuffle.partitions", "1000"))
    input_df = input_df.repartition(num_shuffle_partitions)
    
    # Create MinHash UDF
    #minhash_udf = udf(
    #    lambda text: compute_minhash_signature(text=text, num_hashes=num_hashes, ngram=9, normalize=True),
    #    ArrayType(IntegerType())
    #)

    @pandas_udf(ArrayType(IntegerType()))
    def minhash_batch_udf(rows: pd.Series) -> pd.Series:
        """Process entire batch using highly optimized vectorized operations"""
        return compute_minhash_vectorized_batch_only_hash_once(rows, num_hashes, ngram=9)
    
    # df_with_signatures = input_df.withColumn(
    #    "minhash_signature",
    #    minhash_udf(col(text_column))
    # ).cache()  # Cache as we'll use multiple times

    # df_with_signatures = spark.createDataFrame(
    #    input_df.rdd.mapPartitions(
    #        lambda rows: minhash_batch_udf(rows)
    #    ),
    #    schema
    #).cache()

    df_with_signatures = input_df.withColumn(
        "minhash_signature",
        minhash_batch_udf(col(text_column))
    ).cache()
    
    total_docs_count = df_with_signatures.count()
    logger.info(f"Processing {total_docs_count} documents...")
    
    # Step 2: Compute partition assignments based on LSH bands
    logger.info("Step 2: Computing partition assignments (KEY INNOVATION)...")
    
    def compute_partition_assignments(signature: List[int]) -> List[int]:
        """
        Determine which partitions this document needs to be sent to
        based on its LSH bands. This ensures similar docs end up in same partition.
        """
        if not signature or all(s == 0 for s in signature):
            return [0]  # Default partition
        
        partitions = set()
        
        for band_id in range(num_bands):
            start = band_id * rows_per_band
            end = builtin_min(start + rows_per_band, len(signature))
            
            if start >= len(signature):
                break
            
            # Hash the band to get partition assignment
            band_values = tuple(signature[start:end])
            band_hash = builtin_hash(band_values)
            
            # Map to partition - documents with same band hash go to same partition
            partition_id = builtin_abs(band_hash) % num_partitions
            partitions.add(partition_id)
        
        return list(partitions)
    
    partition_assignment_udf = udf(
        compute_partition_assignments,
        ArrayType(IntegerType())
    )
    
    df_with_partitions = df_with_signatures.withColumn(
        "target_partitions",
        partition_assignment_udf(col("minhash_signature"))
    ).cache()
    
    # Show partition distribution for monitoring
    partition_stats = df_with_partitions.select(
        size(col("target_partitions")).alias("num_partitions_per_doc")
    ).agg(
        avg("num_partitions_per_doc").alias("avg_partitions"),
        min("num_partitions_per_doc").alias("min_partitions"),
        max("num_partitions_per_doc").alias("max_partitions")
    ).collect()[0]
    
    logger.info(f"Partition assignment stats - Avg: {partition_stats['avg_partitions']:.2f}, "
          f"Min: {partition_stats['min_partitions']}, Max: {partition_stats['max_partitions']}")
    
    # Step 3: Explode and repartition - documents go to their assigned partitions
    logger.info("Step 3: Smart partitioning - co-locating similar documents...")
    
    df_exploded = df_with_partitions.select(
        col("doc_id"),
        col(text_column),
        col("minhash_signature"),
        explode(col("target_partitions")).alias("partition_id")
    ).cache()

    # Monitor partition skew before repartitioning
    partition_distribution = df_exploded.groupBy("partition_id").count().collect()
    partition_counts = [row['count'] for row in partition_distribution]

    max_partition_size = builtin_max(partition_counts)
    min_partition_size = builtin_min(partition_counts) 
    avg_partition_size = builtin_sum(partition_counts) / len(partition_counts)
    skew_ratio = max_partition_size / avg_partition_size if avg_partition_size > 0 else 0

    logger.info(f"Partition skew analysis:")
    logger.info(f"  Max partition size: {max_partition_size:,}")
    logger.info(f"  Min partition size: {min_partition_size:,}")
    logger.info(f"  Avg partition size: {avg_partition_size:.0f}")
    logger.info(f"  Skew ratio (max/avg): {skew_ratio:.2f}")
    logger.info(f"  Active partitions: {len(partition_counts)} / {num_partitions}")
    
    # KEY INNOVATION: Repartition based on computed partition assignments
    # This ensures similar documents are in the same partition
    df_partitioned = df_exploded.repartition(num_partitions, col("partition_id"))
    
    # Step 4: Process each partition locally (no shuffle!)
    logger.info("Step 4: Local deduplication within partitions (NO SHUFFLE)...")
    
    def process_partition_locally(iterator: Iterator) -> Iterator:
        """
        Process all documents within a single partition locally.
        This is where the magic happens - no network I/O needed!
        """
        # Collect documents in this partition
        local_docs = []
        for row in iterator:
            local_docs.append({
                'doc_id': row['doc_id'],
                'text': row[text_column],
                'signature': row['minhash_signature'],
                'partition_id': row['partition_id']
            })
        
        if not local_docs:
            return iter([])
        
        # Build local LSH index for this partition
        band_index = defaultdict(list)
        
        for doc in local_docs:
            sig = doc['signature']
            if not sig or all(s == 0 for s in sig):
                continue
            
            # Generate bands
            for band_id in range(num_bands):
                start = band_id * rows_per_band
                end = builtin_min(start + rows_per_band, len(sig))
                
                if start >= len(sig):
                    break
                
                band_values = tuple(sig[start:end])
                band_hash = builtin_hash(band_values)
                
                # Add to local index
                band_key = f"{band_id}_{band_hash}"
                band_index[band_key].append(doc)
        
        # Find similar pairs within this partition
        seen_pairs = set()
        similar_pairs = []
        
        for band_key, docs_in_band in band_index.items():
            if len(docs_in_band) < 2:
                continue
            
            # Compare all pairs in this band
            for i, doc1 in enumerate(docs_in_band):
                for doc2 in docs_in_band[i+1:]:
                    # Create canonical pair ID
                    pair_id = tuple(sorted([doc1['doc_id'], doc2['doc_id']]))
                    
                    if pair_id in seen_pairs:
                        continue
                    
                    seen_pairs.add(pair_id)
                    
                    # Compute similarity
                    similarity = estimate_similarity(doc1['signature'], doc2['signature'])
                    
                    if similarity >= similarity_threshold:
                        similar_pairs.append({
                            'doc1': pair_id[0],
                            'doc2': pair_id[1],
                            'similarity': similarity,
                            'partition_id': doc1['partition_id']
                        })
        
        return iter(similar_pairs)
    
    # Process partitions and find similar pairs
    similar_pairs_rdd = df_partitioned.rdd.mapPartitions(process_partition_locally)
    
    # Convert back to DataFrame
    similar_pairs_schema = StructType([
        StructField("doc1", StringType(), False),
        StructField("doc2", StringType(), False),
        StructField("similarity", FloatType(), False),
        StructField("partition_id", IntegerType(), False)
    ])
    
    similar_pairs_df = spark.createDataFrame(similar_pairs_rdd, similar_pairs_schema)
    
    # Remove duplicate pairs that might appear in multiple partitions
    similar_pairs_df = similar_pairs_df.dropDuplicates(["doc1", "doc2"]).persist(StorageLevel.MEMORY_AND_DISK)
    
    similar_count = similar_pairs_df.count()
    logger.info(f"Found {similar_count} similar document pairs")
    
    # Step 5: Build connected components for duplicate groups
    logger.info("Step 5: Build connected components. "
                "For each distinct doc_id, it has representative doc_id")

    vertices = input_df.select(col("doc_id").alias("id")).distinct().persist(StorageLevel.MEMORY_ONLY)
    doc_id_and_representative_doc_id_df_deduped = get_deduplicate_df_graphframes(
        spark=spark,
        similar_pairs_df=similar_pairs_df,
        vertices=vertices
    ).persist(StorageLevel.MEMORY_AND_DISK)
    logger.info("vertices cached. doc_id_and_representative_doc_id_df_deduped cached.")


    # Step 6: Join back with original data
    logger.info("Step 6: Marking duplicates...")
    
    result = input_df.join(
        doc_id_and_representative_doc_id_df_deduped,
        on="doc_id",
        how="left"
    ).withColumn(
        "representative_id",
        when(col("representative_id").isNull(), col("doc_id"))
        .otherwise(col("representative_id"))
    ).withColumn(
        "is_duplicate",
        col("representative_id") != col("doc_id")
    )

    # Compute statistics
    deduplicate_docs = result.filter(~col("is_duplicate"))
    unique_docs_count = deduplicate_docs.count()
    duplicate_docs_count = total_docs_count - unique_docs_count
    
    logger.info("\n" + "="*60)
    logger.info("PARTITION-AWARE DEDUPLICATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Total documents: {total_docs_count:,}")
    logger.info(f"Duplicate documents: {duplicate_docs_count:,}")
    logger.info(f"Unique documents: {unique_docs_count:,}")
    logger.info(f"Deduplication rate: {duplicate_docs_count/total_docs_count*100:.2f}%")
    logger.info("="*60)

    # Clean up cached DataFrames to free memory
    logger.info("Cleaning up cached DataFrames...")
    df_with_signatures.unpersist()
    df_with_partitions.unpersist()
    df_exploded.unpersist()
    similar_pairs_df.unpersist()
    vertices.unpersist()
    doc_id_and_representative_doc_id_df_deduped.unpersist()
    
    return result
