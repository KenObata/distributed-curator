# Import Python's built-in functions before PySpark overwrites them
import builtins
builtin_hash = builtins.hash

from pyspark.sql.functions import *
from pyspark.sql.types import *
import numpy as np
import pandas as pd

def compute_minhash_vectorized_batch(texts: pd.Series, num_hashes: int = 128, ngram: int = 9) -> pd.Series:
    """
    DEPRECATED: Highly optimized vectorized MinHash computation using pandas and numpy
    
    This function has been deprecated in favor of compute_minhash_vectorized_batch_only_hash_once()
    which provides better performance through improved hash mixing optimization.
    
    Moved from spark_partition_aware_deduplicattion_v2.py on 2025-12-28.
    """
    results = []
    
    # Step 1: Vectorized text normalization using pandas string operations
    normalized_texts = texts.str.lower()
    
    # Remove articles using vectorized regex
    articles_pattern = r'\b(the|a|an|this|that|these|those)\b'
    normalized_texts = normalized_texts.str.replace(articles_pattern, '', regex=True)
    normalized_texts = normalized_texts.str.replace(r'\s+', ' ', regex=True)
    normalized_texts = normalized_texts.str.strip()
    
    # Step 2: Batch process texts for MinHash computation
    for text in normalized_texts:
        if not text or len(text) < ngram:
            results.append([0] * num_hashes)
            continue
        
        # Generate shingles (basic implementation since generate_shingles is not available here)
        shingles = [text[i:i+ngram] for i in range(len(text) - ngram + 1)]
        unique_shingles = list(set(shingles))
        
        if not unique_shingles:
            results.append([0] * num_hashes)
            continue
        
        # Vectorized MinHash computation using numpy
        # Pre-allocate hash array for all shingles and hash functions
        signature_matrix = np.zeros((len(unique_shingles), num_hashes), dtype=np.uint32)
        
        # Hash mixing optimization: compute all hashes efficiently
        if not hasattr(compute_minhash_vectorized_batch, '_seeds_cache'):
            import random
            random.seed(42)  # Same seed for reproducibility
            compute_minhash_vectorized_batch._seeds_cache = [
                random.randint(1, 0xFFFFFFFF) for _ in range(1024)
            ]
        
        hash_seeds = compute_minhash_vectorized_batch._seeds_cache[:num_hashes]
        
        for j, shingle in enumerate(unique_shingles):
            base_hash = builtin_hash(shingle) & 0xFFFFFFFF
            for i in range(num_hashes):
                # Hash mixing instead of string concatenation
                signature_matrix[j, i] = (base_hash ^ hash_seeds[i]) & 0xFFFFFFFF
        
        # Vectorized minimum computation across all shingles
        signature = np.min(signature_matrix, axis=0)
        results.append(signature.tolist())
    
    return pd.Series(results)

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

def get_doc_id_and_representative_doc_id_df_deduped(
    spark: SparkSession,
    similar_pairs_df: DataFrame, 
    all_doc_ids_df: DataFrame, 
    is_debug_mode: bool) -> DataFrame:
    """
    This function returns this dataframe:
    Arg: 
    - similar_pairs_df: only contains doc1 - doc2 pair with similarity >= input threshold.
    - all_doc_ids_df: input_df.select(col("doc_id")).distinct()

    doc_id_and_representative_doc_id_df_deduped records:
    +------+---------------------+
    |doc_id|representative_doc_id|
    +------+---------------------+
    |doc1  |doc1                 |
    |doc4  |doc1                 |
    |doc2  |doc1                 |
    +------+---------------------+
    """
    # Get all edges
    edges = similar_pairs_df.select(
        col("doc1").alias("src"),
        col("doc2").alias("dst")
    )
    logger.info("edges records:")
    log_dataframe(edges, is_debug_mode)
    
    
    # Get documents involved in duplicates
    docs_with_duplicates = edges.select("src").union(edges.select("dst")).distinct()
    logger.info("docs_with_duplicates:")
    log_dataframe(docs_with_duplicates, is_debug_mode)

    # Build groups
    edges_group_by_src_df = edges.groupBy("src").agg(
        collect_set("dst").alias("connected_docs")
    )
    logger.info("edges_group_by_src_df records:")
    log_dataframe(edges_group_by_src_df, is_debug_mode)
    
    combine_src_and_connected_docs_df = edges_group_by_src_df.select(
        col("src").alias("doc_id"),
        array_union(array(col("src")), col("connected_docs")).alias("all_connected")
    )
    logger.info("combine_src_and_connected_docs_df records:")
    log_dataframe(combine_src_and_connected_docs_df, is_debug_mode)


    # Find representative (minimum doc_id in group)
    doc_id_and_representative_doc_id_df = combine_src_and_connected_docs_df.select(
        explode(col("all_connected")).alias("doc_id"),
        array_min(col("all_connected")).alias("representative_id")
    )

    logger.info("doc_id_and_representative_doc_id_df records:")
    log_dataframe(doc_id_and_representative_doc_id_df, is_debug_mode)

    # doc_id_and_representative_doc_id_df still contains duplicates.
    """
    ex) 
    combine_src_and_connected_docs_df records:
    +------+------------------+
    |doc_id|all_connected     |
    +------+------------------+
    |doc1  |[doc1, doc4, doc2]|
    |doc2  |[doc2, doc4]      |
    +------+------------------+

    doc_id_and_representative_doc_id_df records:
    +------+---------------------+
    |doc_id|representative_doc_id|
    +------+---------------------+
    |doc1  |doc1                 |
    |doc4  |doc1                 |
    |doc2  |doc1                 |
    |doc2  |doc2                 |
    |doc4  |doc2                 |
    +------+---------------------+

    doc_id_and_representative_doc_id_df_deduped records:
    +------+---------------------+
    |doc_id|representative_doc_id|
    +------+---------------------+
    |doc1  |doc1                 |
    |doc4  |doc1                 |
    |doc2  |doc1                 |
    +------+---------------------+

    This is because we explode the all_connected array and then select the representative document id.
    So we are missing deduping transient, ex) doc1-doc2-doc4 as one group.
    So now we need to do this:
    """
    # Create temporary view for SQL query
    doc_id_and_representative_doc_id_df.createOrReplaceTempView("doc_id_and_representative_doc_id_df")
    
    sql_command = """
    SELECT doc_id, MIN(representative_id) as representative_id
    FROM doc_id_and_representative_doc_id_df
    GROUP BY doc_id
    """
    doc_id_and_representative_doc_id_df_deduped = spark.sql(sql_command)
    logger.info("doc_id_and_representative_doc_id_df_deduped:")
    log_dataframe(doc_id_and_representative_doc_id_df_deduped, is_debug_mode)

    return doc_id_and_representative_doc_id_df_deduped

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