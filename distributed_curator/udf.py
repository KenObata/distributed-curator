from __future__ import annotations  # lazy import of pandas

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
import logging
from collections import defaultdict
from collections.abc import Iterator

import numpy as np

builtin_hash = hash
builtin_min = min
builtin_sum = sum
builtin_max = max
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Safety cap: skip bands with too many docs to prevent O(n²) explosion
# Bands with >1000 docs are likely hash collisions on common patterns
MAX_BAND_SIZE = 1000


def compute_minhash_vectorized_batch_only_hash_once(
    texts: pd.Series, num_hashes: int = 128, ngram: int = 9, remove_articles: bool = False
) -> pd.Series:
    """
    This is for Step1: compute MinHash. This function is pure python UDF which is no longer used.
    But if in case users cannot use src/cython_min_hash/ code, we keep this python UDF.
    Highly optimized vectorized MinHash computation using pandas and numpy
    Difference from compute_minhash_vectorized_batch is that this function
    hashes the shingle only once and then for different seed i, use different permutation to generate different hash values.
    """
    import pandas as pd

    results = []

    # Step 1: Vectorized text normalization using pandas string operations
    normalized_texts = texts.str.lower()

    # Remove articles using vectorized regex
    if remove_articles:
        articles_pattern = r"\b(the|a|an|this|that|these|those)\b"
        normalized_texts = normalized_texts.str.replace(articles_pattern, "", regex=True)
        normalized_texts = normalized_texts.str.replace(r"\s+", " ", regex=True)
        normalized_texts = normalized_texts.str.strip()

    # Step 2: Process each text individually (fixed the Series slicing issue)
    for text in normalized_texts:
        if not text or len(text) < ngram:
            results.append([0] * num_hashes)
            continue

        # Generate unique shingles for this specific text string (memory-optimized)
        unique_shingles = list({text[i : i + ngram] for i in range(len(text) - ngram + 1)})
        if not unique_shingles:
            results.append([0] * num_hashes)
            continue

        # HASH MIXING OPTIMIZATION: Hash each shingle ONCE, then mix with seeds
        base_hashes = np.array([builtin_hash(s) & 0xFFFFFFFF for s in unique_shingles], dtype=np.uint32)

        # Use same seeds as the main function for consistency
        if not hasattr(compute_minhash_vectorized_batch_only_hash_once, "_seeds_cache"):
            import random

            random.seed(42)  # Same seed for reproducibility
            compute_minhash_vectorized_batch_only_hash_once._seeds_cache = [
                random.randint(1, 0xFFFFFFFF) for _ in range(1024)
            ]

        hash_seeds = np.array(
            compute_minhash_vectorized_batch_only_hash_once._seeds_cache[:num_hashes], dtype=np.uint32
        )

        # Vectorized hash mixing: broadcast (num_shingles, 1) x (num_hashes,)
        base_hashes_expanded = base_hashes[:, np.newaxis]  # Shape: (num_shingles, 1)

        # Apply hash mixing: (base_hash XOR seed) for each combination
        """
        Reminder:
        MinHash needs num_hashes (e.g., 64) independent hash functions.
        But computing 64 different hashes for each shingle is expensive:

        for seed in range(64):
            hash_val = mmh3.hash(shingle, seed=seed)  # 64 hash computations is slow.

        Use XOR because:
        - AND,OR biase toward 0,1 which is less random
        - Addition has "carry" - bits affect each other and affects other bits.
        - loses reversibility.
        XOR is the only bitwise operation that preserves randomness
        """
        mixed_hashes = (base_hashes_expanded ^ hash_seeds) & 0xFFFFFFFF

        # Get minimum across all shingles for each hash function
        signature = np.min(mixed_hashes, axis=0)
        results.append(signature.tolist())

    return pd.Series(results)


def estimate_similarity(sig1: list[int], sig2: list[int]) -> float:
    """Estimate Jaccard similarity from MinHash signatures"""

    if not sig1 or not sig2 or len(sig1) != len(sig2):
        return 0.0

    # Count matching MinHash values
    matches = builtin_sum(1 for h1, h2 in zip(sig1, sig2) if h1 == h2 and h1 != 0)

    # Avoid division by zero
    if all(h == 0 for h in sig1) or all(h == 0 for h in sig2):
        return 0.0

    return float(matches) / len(sig1)


def process_partition_locally(
    iterator: Iterator, num_bands: int, rows_per_band: int, similarity_threshold: float
) -> Iterator:
    """
    This function is not referenced anywhere.
    We created this function in case we want to run python vectorized UDF as baseline
    against scala UDF.
    Process all documents within a single partition locally.
    This is where the magic happens - no network I/O needed!

    Why do we do this function per partition, not per row like other UDFs?
    It's because we need to compare documents within the same partition, not transform each row independently.
    With per row UDF process, it can't compare to neighbors.
    This function needs to see ALL docs in partition.
    """
    # Collect documents in this partition
    local_docs = []
    for row in iterator:
        local_docs.append(
            {
                "doc_id": row["doc_id"],
                # 'text': row[text_column],  # Removed to reduce Python memory usage
                "signature": row["minhash_signature"],
                "partition_id": row["partition_id"],
            }
        )

    if not local_docs:
        return iter([])

    # Build local LSH index for this partition
    band_index = defaultdict(list)

    for doc in local_docs:
        sig = doc["signature"]
        if not sig or all(s == 0 for s in sig):
            continue

        # Generate bands
        for band_id in range(num_bands):
            start = band_id * rows_per_band
            end = builtin_min(start + rows_per_band, len(sig))

            if start >= len(sig):  # if start starts after 128 hash sampels break
                break

            band_values = tuple(sig[start:end])
            band_hash = builtin_hash(band_values)

            # Add to local index
            band_key = f"{band_id}_{band_hash}"
            band_index[band_key].append(doc)

    # Find similar pairs within this partition
    seen_pairs = set()
    similar_pairs = []

    skipped_bands = 0
    max_band_seen = 0

    for _band_key, docs_in_band in band_index.items():  # _band_key for B007
        band_size = len(docs_in_band)
        max_band_seen = builtin_max(max_band_seen, band_size)

        if band_size < 2:
            continue

        # Skip oversized bands to prevent memory/compute explosion
        if band_size > MAX_BAND_SIZE:
            skipped_bands += 1
            continue

        # Compare all pairs in this band
        for i, doc1 in enumerate(docs_in_band):
            for doc2 in docs_in_band[i + 1 :]:
                # Create canonical pair ID
                """
                Sort ensures there is no dups in similar_pairs with this case (doc1,doc2) and (doc2, doc1).
                This case never happens.
                """
                pair_id = tuple(sorted([doc1["doc_id"], doc2["doc_id"]]))

                if pair_id in seen_pairs:
                    continue

                seen_pairs.add(pair_id)

                # Compute similarity
                similarity = estimate_similarity(doc1["signature"], doc2["signature"])

                if similarity >= similarity_threshold:
                    similar_pairs.append(
                        {
                            "doc1": pair_id[0],
                            "doc2": pair_id[1],
                            "similarity": similarity,
                            "partition_id": doc1["partition_id"],
                        }
                    )
    logger.info(f"skipped_bands: {skipped_bands}")
    logger.info(f"max_band_seen: {max_band_seen}")

    return iter(similar_pairs)
