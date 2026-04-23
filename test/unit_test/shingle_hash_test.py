"""
Test script to verify:
1. Cython MurmurHash3 produces identical hashes to Python mmh3
2. Integration with NumPy MinHash generation
3. Performance comparison: Cython C loop vs Python set comprehension
"""

import time

import mmh3
import numpy as np

# Import the compiled Cython module
from distributed_curator.cython_minhash.shingle_hash import hash_shingles


def test_hash_correctness():
    """Verify Cython mmh3 matches Python mmh3 exactly."""
    text = "hello world this is a test document for minhash"
    ngram = 9
    seed = 0

    # Cython: direct C MurmurHash3
    cython_hashes = hash_shingles(text, ngram, seed)

    # Python: mmh3 package
    text_bytes = text.encode("utf-8")
    python_hashes = []
    for i in range(len(text_bytes) - ngram + 1):
        shingle = text_bytes[i : i + ngram]
        # mmh3.hash returns signed int32, convert to unsigned
        h = mmh3.hash(shingle, seed) & 0xFFFFFFFF
        python_hashes.append(h)
    python_hashes = np.array(python_hashes, dtype=np.uint32)

    # Compare
    assert len(cython_hashes) == len(python_hashes), f"Length mismatch: {len(cython_hashes)} vs {len(python_hashes)}"

    mismatches = np.sum(cython_hashes != python_hashes)
    if mismatches == 0:
        print(f"PASS: All {len(cython_hashes)} hashes match exactly")
    else:
        print(f"FAIL: {mismatches} mismatches out of {len(cython_hashes)}")
        # Show first mismatch
        for i in range(len(cython_hashes)):
            if cython_hashes[i] != python_hashes[i]:
                print(f"  Index {i}: Cython={cython_hashes[i]:#x}, Python={python_hashes[i]:#x}")
                break


def test_minhash_integration():
    """
    Show how Cython shingle hashing integrates with NumPy SIMD MinHash.
    This replaces the existing compute_minhash_vectorized_batch_only_hash_once.
    """
    text = "hello world this is a test document for minhash deduplication"
    ngram = 9
    num_hashes = 64

    # Step 1: Cython C loop — hash shingles (no dedup)
    base_hashes = hash_shingles(text, ngram, seed=0)  # uint32 array

    # Step 2: NumPy SIMD — compute MinHash signature
    # Generate seeds (same as existing code)
    import random

    random.seed(42)
    hash_seeds = np.array([random.randint(1, 0xFFFFFFFF) for _ in range(num_hashes)], dtype=np.uint32)

    # Broadcast XOR: (num_shingles, 1) ^ (num_hashes,) → (num_shingles, num_hashes)
    # This is the SIMD-optimized operation
    base_expanded = base_hashes[:, np.newaxis]  # (N, 1)
    mixed = (base_expanded ^ hash_seeds) & 0xFFFFFFFF  # (N, num_hashes) — SIMD
    signature = np.min(mixed, axis=0)  # (num_hashes,) — SIMD

    print(f"Shingles: {len(base_hashes)}")
    print(f"Signature length: {len(signature)}")
    print(f"First 5 values: {signature[:5]}")


def test_performance():
    """Compare Cython C loop vs Python set comprehension."""
    # Generate a realistic document (~10KB)
    text = "The quick brown fox jumps over the lazy dog. " * 250
    ngram = 9
    iterations = 1000

    # Cython: C loop with direct MurmurHash3 (no dedup)
    start = time.perf_counter()
    for _ in range(iterations):
        cython_result = hash_shingles(text, ngram)
    cython_time = time.perf_counter() - start

    # Python: set comprehension with mmh3 (current approach)
    start = time.perf_counter()
    for _ in range(iterations):
        unique_shingles = list({text[i : i + ngram] for i in range(len(text) - ngram + 1)})
        python_result = np.array([hash(s) & 0xFFFFFFFF for s in unique_shingles], dtype=np.uint32)
    python_time = time.perf_counter() - start

    print(f"\nPerformance ({iterations} iterations, ~{len(text)} byte document):")
    print(f"  Cython (C loop, no dedup): {cython_time:.3f}s  ({len(cython_result)} hashes)")
    print(f"  Python (set + hash):       {python_time:.3f}s  ({len(python_result)} unique hashes)")
    print(f"  Speedup: {python_time / cython_time:.1f}x")


if __name__ == "__main__":
    print("=" * 60)
    print("Test 1: Hash correctness (Cython vs Python mmh3)")
    print("=" * 60)
    test_hash_correctness()

    print("\n" + "=" * 60)
    print("Test 2: MinHash integration with NumPy SIMD")
    print("=" * 60)
    test_minhash_integration()

    print("\n" + "=" * 60)
    print("Test 3: Performance comparison")
    print("=" * 60)
    test_performance()
