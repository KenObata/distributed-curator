# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    
def process_partition_locally(iterator: Iterator) -> Iterator:
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
        local_docs.append({
            'doc_id': row['doc_id'],
            # 'text': row[text_column],  # Removed to reduce Python memory usage
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
            
            if start >= len(sig): # if start starts after 128 hash sampels break
                break
            
            band_values = tuple(sig[start:end])
            band_hash = builtin_hash(band_values)
            
            # Add to local index
            band_key = f"{band_id}_{band_hash}"
            band_index[band_key].append(doc)
    
    # Find similar pairs within this partition
    seen_pairs = set()
    similar_pairs = []
    
    # Safety cap: skip bands with too many docs to prevent O(n²) explosion
    # Bands with >1000 docs are likely hash collisions on common patterns
    MAX_BAND_SIZE = 1000
    skipped_bands = 0
    max_band_seen = 0

    for band_key, docs_in_band in band_index.items():
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
            for doc2 in docs_in_band[i+1:]:
                # Create canonical pair ID
                """
                Sort ensures there is no dups in similar_pairs with this case (doc1,doc2) and (doc2, doc1).
                This case never happens.
                """
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
    logger.info(f"skipped_bands: {skipped_bands}")
    logger.info(f"max_band_seen: {max_band_seen}")
    
    return iter(similar_pairs)