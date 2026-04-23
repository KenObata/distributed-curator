"""
This is a wrapper python file to call shingle_hash.pyx

Architecture:
  1. shingle_hash.pyx:
      Cython C loop: text → MurmurHash3_x86_32 per shingle → raw uint32 array
     (no dedup, no Python objects, no set)
  2. shingle_hash_wrapper.py:
    NumPy SIMD: raw hashes from shingle_hash.pyx →
    broadcast XOR with seeds → min reduction → signature
     (processes all shingles including duplicates, min() is idempotent)
"""

import logging
import random

import numpy as np
import pandas as pd

# Import compiled Cython module
try:
    from .cython_minhash.shingle_hash import hash_shingles
except Exception:
    from shingle_hash import hash_shingles

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Precompute seeds once (same as existing code for reproducibility)
_SEEDS_CACHE = None


def _get_seeds(num_hashes: int) -> np.ndarray:
    global _SEEDS_CACHE
    if _SEEDS_CACHE is None:
        rng = random.Random(42)
        _SEEDS_CACHE = np.array([rng.randint(1, 0xFFFFFFFF) for _ in range(1024)], dtype=np.uint32)
    return _SEEDS_CACHE[:num_hashes]


def compute_minhash_cython_batch(texts: pd.Series, num_hashes: int = 128, ngram: int = 9) -> pd.Series:
    """
    Batch version for pandas_udf.

    This is a replacement for udf/compute_minhash_vectorized_batch_only_hash_once
    """
    hash_seeds = _get_seeds(num_hashes)
    results = []
    # Log which module is loaded (runs once per executor batch)
    import shingle_hash

    logger.info(f"shingle_hash loaded from: {shingle_hash.__file__}")

    for text in texts.str.lower():
        if not text or not isinstance(text, str) or len(text) < ngram:
            results.append([0] * num_hashes)
            continue

        # Cython C loop — direct MurmurHash3, no Python objects
        base_hashes = hash_shingles(text, ngram, seed=0)

        if len(base_hashes) == 0:
            results.append([0] * num_hashes)
            continue

        # NumPy SIMD — broadcast XOR + min
        base_expanded = base_hashes[:, np.newaxis]
        mixed = (base_expanded ^ hash_seeds) & 0xFFFFFFFF
        signature = np.min(mixed, axis=0)
        results.append(signature.tolist())

    return pd.Series(results)
