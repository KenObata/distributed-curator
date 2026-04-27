# Architecture

## Section1: Overview

distributed-curator deduplicates near duplciate text documents at web scale using partition-aware MinHash LSH.
The core idea is to co-locate similar documents in the same Spark partition before comparing them,
so that all similarity comparisons happen locally with no shuffle. A two-phase union-find then
merges duplicate groups across partition boundaries.

The pipeline has six steps:

| Step | What happens | Output | Shuffle? |
|------|-------------|--------|----------|
| 1. MinHash | Compute MinHash signatures and LSH band hashes | (doc_id, minhash_signature[64]) | No |
| 2. Partition assignment | Map band hashes to partition IDs | (doc_id, minhash_signature[64], partition_ids[8], band_hashes[8]) — one row per band | No |
| 3. Explode Array & Identity repartition | Move documents to assigned partitions | (doc_id, minhash_signature[64], partition_id, band_hash), one row per band, physically co-located by partition_id | Yes (once) |
| 4. Local similarity | Find candidate pairs exceeding similarity threshold within each partition | (doc1, doc2, similarity, partition_id) where similarity exceeds user input threshold | No |
| 5a. Local union-find | Build connected components per partition | (doc_id, local_representative) one per unique doc per partition | No |
| 5b. Global union-find | Merge components across partitions on driver | (doc_id, global representative_id) for all docs including singletons | No |
| 6. Mark duplicates | Join results back to input | Original columns + representative_id + is_duplicate | Yes (once) |


Only two steps involve shuffles — the initial repartition and the final join.
Everything in between operates on partition-local data.

# Section2: Details
## Step 1: MinHash Signatures

MinHash estimates Jaccard similarity between two documents without comparing them directly.
The idea: if you hash all character n-grams (shingles) of a document with a hash function
and take the minimum hash value, two similar documents will share the same minimum with
probability equal to their Jaccard similarity.

### textbook reference
MinHash and LSH (Locality-Sensitive Hashing) is explained more details in the textbook:
Leskovec, Rajaraman & Ullman, "Mining of Massive Datasets" (MMDS), Chapter 3: Finding Similar Items. This is freely available at http://www.mmds.org and covers MinHash + LSH banding technique in detail.

### Why 64 or more hash samples?

A single minimum hash gives a noisy estimate — one coin flip. Using 64 independent hash
functions produces 64 minimums (a "signature"), and the fraction of matching minimums
between two documents converges to their true Jaccard similarity.

Each of the 64 "hash functions" is created by XOR-ing the base hash with a different seed:

base_hash = MurmurHash3(shingle)          # one hash per shingle
signature[i] = min(base_hash XOR seed[i])  # one min per seed, across all shingles

This is cheaper than computing 64 independent hashes — one MurmurHash3 call per shingle,
then 64 XOR operations via NumPy SIMD (all seeds processed in a single vectorized pass).

More hashes = more accurate similarity estimate but more memory per document.
64 is the sweet spot for web-scale dedup — enough accuracy for 0.8+ similarity detection
with manageable memory at 2.5B documents.

### How band hashing works

The 64-element signature is split into 8 bands of 8 rows each.
For each band, the 8 values are hashed together into a single band hash:

signature = [h0, h1, h2, ..., h63]

band 0: hash(h0,  h1,  h2,  h3,  h4,  h5,  h6,  h7)   -> band_hash_0

band 1: hash(h8,  h9,  h10, h11, h12, h13, h14, h15)   -> band_hash_1

...

band 7: hash(h56, h57, h58, h59, h60, h61, h62, h63)   -> band_hash_7

Two documents become a candidate pair if they share **any** band hash.
This is the LSH (Locality-Sensitive Hashing) amplification trick:

- Identical documents match on all 8 bands
- Similar documents (Jaccard ~0.9) likely match on at least one band
- Dissimilar documents are unlikely to match on any band

For details on the background of how MinHash mathmatically work,
see [docs/background_math.md](background_math.md).

### Implementation Details

Step1 runs on @pandas_udf:
```
JVM -> [Arrow IPC] -> Arrow RecordBatch (one contiguous string in memory) -> pd.Series[String]
-> actual function
-> pd.Series[List[Long]] -> Arrow RecordBatch -> [Arrow IPC] -> JVM
```

- Arrow RecordBatch -> pd.Series[List[String]]: 
  - pd.Series is a scattered heap allocations. Each str is a separate malloc call. 
    So converion from arrow format to pd.Series mean:
        - extract every string from Arrow into a Python str object (28K malloc calls + UTF-8 copies), and then wrap them in a pandas Series with object dtype (an array of pointers)    
    - ref: to_pandas()  // arrow/python/pyarrow/src/arrow/python/arrow_to_pandas.cc 's PyUnicode_FromStringAndSize
  - [FYI] With pandas 3.0+, pandas keeps the Arrow buffer underneath and never created Python str objects at all. So we can eliminate this python string conversion (malloc calls + UTF-8 copies).


we minimize Arrow RecordBatch -> pd.Series conversion by sending reconds in batch (=arrow.maxRecordsPerBatch).
But this Arrow RecordBatch -> pd.Series conversion is what roughly costs ~75ms per task.


#### Why pandas_udf over scala UDF?
Using 90K WET file scale with 90K spark partiiton count config, each task processes ~28K documents (~196MB of text). With arrow.maxRecordsPerBatch=30K, Observed: ~3.5 min/task, ~7,800-9,400 docs/min throughput.

Step 1 UDF rough breakdown:
```
Arrow IPC (JVM -> Python socket):     ~300ms    (local socket, ~196MB)
Arrow StringArray -> pd.Series[List[String]]:       ~50-100ms (28K Python str object allocations)
Actual computation:
    - Cython shingling + MurmurHash3:      ~70-90s   (28K × ~7000 shingles × hash each)
    - NumPy SIMD MinHash (64 perms):       ~100-120s (vectorized XOR+min across batch)
pd.Series[List[Long]] -> Arrow IPC:   ~30ms     (numeric, near zero-copy)
─────────────────────────────────────────────
Total:                                ~210s
```

    
Actual computation dominates the processing time over arrow format -> pd.Series string conversion cost. Scala UDF by default does not take advantage of SIMD operation.

### Output

Each document produces:
- `doc_id`: document identifier
- `minhash_signature[64]`: the 64-element MinHash signature

Step 2 takes these signatures and computes partition assignments.

## Step 2: Partition Assignment

Step 2 takes each document's 8 band hashes and maps them to logical partition IDs. Here, logical means we have a column partiiton_id and the physical partition_id mapping happens in Step3.
This is where co-location happens — documents sharing a band hash get
assigned to the same partition, guaranteeing they will be compared in Step 4.

### How it works

For each band hash, the partition ID is computed as:
```
partition_id = band_hash % num_partitions
```

A document with 8 bands gets up to 8 partition assignments.
Two documents sharing band hash 3 both get `band_hash_3 % num_partitions`,
so they land in the same partition.

### Why this is a Scala UDF

Partition assignment is a lightweight computation — no SIMD benefit,
just integer modulo. It's implemented as a Scala UDF (`ComputePartitionAssignmentsUDF`)
to avoid Python serialization overhead. The UDF returns a struct with two arrays:

- `target_partitions[8]`: which partitions this document belongs to
- `band_hashes[8]`: the raw band hashes (used later for deterministic salting)

### Output

Each document now knows which partitions it should be sent to.
A single document may appear in multiple partitions — this is expected and necessary.
If doc A and doc B share the same band hash X, they both appear in the same partition
via that band, even if their other 7 bands differ.

Step 3 explodes these assignments and physically moves documents
to their assigned partitions.

## Step 3: Identity Repartition

Step 3 explodes the 8 partition assignments into 8 rows per document,
then physically moves each row to its assigned partition.

### Explode

Each document goes from one row with `target_partitions[8]` to eight rows,
each with a single `partition_id` and `band_hash`:
```
Before: (doc_id, minhash_signature[64], target_partitions[8], band_hashes[8])
After:  (doc_id, minhash_signature[64], partition_id, band_hash)  × 8 rows
```

This is the point where data volume multiplies by 8.

### Deterministic salting for hot partitions

Some partitions receive disproportionately more documents — common patterns
in web text produce popular band hashes that map to the same partition.
Step 4's pairwise comparison within a partition is O(n²) in partition size,
so a single hot partition can dominate runtime.

Before repartitioning, we sample 1% of the data, count documents per partition,
and flag partitions with more than 4× the average as "hot." Hot partitions
are split using deterministic salting:
```
new_partition_id = partition_id + abs(band_hash) % num_splits
```
The same `band_hash` always maps to the same sub-partition, so documents
that need to be compared stay together. Different band hashes spread across
sub-partitions, reducing the size of each. Locality remains.

### Why identity repartition, not Spark's repartition()

Spark's built-in `repartition("partition_id")` applies MurmurHash3 internally:
```
physical_partition = murmur3(partition_id) % num_partitions
```
This means two documents with `partition_id = 5` and `partition_id = 1005`
could collide into the same physical partition, while `partition_id = 5`
might not map to physical partition 5 at all. The co-location guarantee
from Step 2 is destroyed.

Identity repartition uses Spark's `HashPartitioner` with integer keys,
which maps `partition_id` directly to the physical partition with the same index.
This is implemented as a Scala helper (`IdentityRepartition.repartitionFromPython`)
to stay in the JVM.

### Output
(doc_id, minhash_signature[64], partition_id) — physically co-located by partition_id

The `band_hash` column is dropped after salting — no longer needed.
This is the only shuffle in the pipeline until the final join in Step 6.

## Step 4: Local Similar Candidate Generation

Step 4 processes each partition independently via `mapPartitions`.
Within each partition, it finds all document pairs whose estimated
Jaccard similarity exceeds the threshold.

### How it works

For each partition, the Scala UDF (`ProcessPartitionLocallyUDF`) does:

1. **Group by band hash**: documents in the same partition may have arrived
   via different bands. Group them by band hash so only documents
   sharing the same band are compared.

2. **Pairwise comparison within each band group**: for each pair of documents
   in the same band group, estimate their Jaccard similarity by counting
   matching values in their 64-element MinHash signatures:

```
estimated_jaccard = count(sig1[i] == sig2[i]) / len(sig1)
```

3. **Threshold filter**: emit the pair only if the estimated similarity
   exceeds the threshold (e.g., 0.9).

### Why Scala, not Python

This step is branchy logic — iterating over documents, building a HashMap
of band groups, comparing signature arrays element by element.
Python would serialize every row from JVM -> Python -> JVM.
The Scala UDF keeps everything in the JVM, operating directly on
Spark's internal memory format.

The Scala implementation also uses unboxed `Array[Long]` instead of
`Seq[Long]`, `while` loops instead of `for` comprehensions, and
index-based iteration instead of `zip`

### Safety cap

A band hash (e.g. hash of 8 MinHash signatures) with more than 1,000 documents are skipped entirely.
These are almost always hash collisions on common boilerplate
(cookie notices, navigation menus), not genuine near-duplicates.
Comparing all pairs in a 1,000-doc band would produce 500K comparisons
in a single partition — an O(n²) explosion that stalls the executor.

If a document pair is truly similar, they should be caught by at least one of the other 7 band hashes where the band group is smaller.

### No shuffle

All comparisons happen within the partition. Documents were co-located
in Step 3 specifically so this step requires zero network I/O.

### Output
(doc1, doc2, similarity_score, partition_id)


One row per candidate per partition pair exceeding the threshold.
Note: the same pair can appear multiple times if the two documents
share multiple band hashes. Deduplication of pairs happens in Step 5a
after `dropDuplicates` — not here, because `dropDuplicates` would
trigger a shuffle that destroys the partition layout.

## Step 5-a: Local Union-Find

Step 5a takes the similar pairs from Step 4 and builds connected components
within each partition. If doc A is similar to doc B, and doc B is similar to doc C,
then all three belong to the same duplicate group — even if A and C were never
directly compared.

### How it works

The Scala UDF (`PartitionAwareUnionFindUDF`) runs Union-Find via `mapPartitions`:

1. For each partition, iterate over all `(doc1, doc2)` pairs
2. `union(doc1, doc2)` — merge their components with union by rank
3. `find(doc_id)` — path compression gives each document its root representative
4. Emit `(doc_id, local_representative)` for every unique document in the partition

This is a classic Union-Find with path compression and union by rank —
O(α(n)) per operation, effectively constant time.

### Why "local" representative

A document can appear in multiple partitions (it was exploded into 8 rows in Step 3).
In partition 0, doc A might get representative X. In partition 1, doc A might get
representative Y. These are "local" because they're only valid within that partition.

Step 5-b resolves these cross-partition disagreements.

### dropDuplicates happens here

The same `(doc_id, local_representative)` pair can appear from multiple partitions.
After `mapPartitions` completes, we call `dropDuplicates(["doc_id", "local_representative"])`
to remove redundant mappings. This triggers a shuffle, but it's intentional — the partition
layout from Step 3 is no longer needed after Step 5-a finishes.

### No shuffle during union-find

The union-find itself runs entirely within `mapPartitions` — zero shuffle.
Each partition independently builds its own connected components from the pairs
it received in Step 4.

### Output
(doc_id, local_representative)

ex) (URL_A, URL_B)


Only documents that appeared in at least one similar pair are included.
Singletons (documents with no duplicates) are not in this output —
they get handled in Step 5b.

## Step 5b: Global Union-Find

Step 5b merges connected components across partition boundaries.

### Why it's needed

Doc A appears in partition 0 with representative X, and in partition 1
with representative Y. X and Y are different documents that both represent A —
but they don't know about each other. Without Step 5b, the pipeline would
treat X's group and Y's group as separate duplicate clusters.

### How it works

1. **Collect cross-partition edges**: for each `doc_id` that has multiple distinct
   `local_representative` values, create edges between those representatives.
   For a doc with representatives [X, Y, Z], emit edges: (X,Y), (X,Z).

2. **Early exit**: if no document has multiple representatives, all components
   were resolved locally. Skip directly to Step 6.

3. **Single-pass Union-Find on driver**: the cross-partition edge graph is much
   smaller than the original data — only documents that appeared in multiple
   partitions with different representatives. This graph fits in driver memory.

   The implementation uses Eclipse Collections `LongLongHashMap` to eliminate
   JVM boxing overhead. Document URLs are encoded to `Long` IDs via
   `monotonically_increasing_id()` + `.checkpoint()` (to prevent re-evaluation),
   and the union-find runs on the Long-encoded edges.

4. **Decode and merge**: map Long IDs back to document URLs, then left join
   with the local results. Each `local_representative` gets its global
   `component` — the true root of the duplicate group.

5. **Map every document to its global representative**: left join all documents
   (including singletons) to produce the final `(doc_id, representative_id)` mapping.
   Singletons get `representative_id = doc_id`.

### Why Eclipse Collections for rank and parent HashMap?

Standard Scala `mutable.HashMap[Long, Long]` boxes every key and value into
`java.lang.Long` objects — ~82 bytes per node. Eclipse Collections `LongLongHashMap`
stores primitive longs directly — ~20 bytes per node. At 90K WET scale with millions
of cross-partition nodes, this saves ~12 GB of driver heap.

### Why not on a single executor ?
If we want to fit global union find on a single executor, we would need to increase executor memory config. But incresing executor memory means decreasing executor count since cluster memory is fixed, which leads to less total cores (tasks) running simultanously.

Driver has more control since its memory config is independent from executor memory. 
Also because increasing driver memory does not significantly lowers executor memory resource.

### Fallback: iterative SQL

For edge graphs exceeding 600M edges (unlikely but possible with extreme data skew),
the pipeline falls back to an iterative SQL approach. This propagates the minimum
component ID through the edge graph via repeated joins until convergence,
using `.checkpoint()` each iteration to truncate lineage and prevent driver OOM.

### Output
(doc_id, global_representative_id) — for all documents including singletons

Each document now has a single global representative.
Documents sharing the same `representative_id` are duplicates.

## Step 6: Mark Duplicates

Step 6 joins the global representative mapping back to the original input
and marks each document as duplicate or unique.

### How it works

1. **Left join**: join `input_df` with the `(doc_id, representative_id)` output
   from Step 5b on `doc_id`. Singletons already have `representative_id = doc_id`
   from Step 5b, but any document not in the mapping (e.g., documents that had
   no text or were too short for shingling) gets `representative_id = doc_id`
   via a `COALESCE` fallback.

2. **Mark duplicates**: a document is a duplicate if its `representative_id`
   differs from its `doc_id`:
   
   is_duplicate = (representative_id != doc_id)

Within each duplicate group, exactly one document has `representative_id = doc_id` —
   that's the representative (the one you keep). All others are duplicates.

### Output
Original columns + representative_id + is_duplicate

Users can filter with:

```python
unique_docs = result.filter(~result.is_duplicate)
```

This is the second and final shuffle in the pipeline — joining the
deduplication results back to the original DataFrame.


## Section3: Composable Language Architecture

distributed-curator uses multiple languages, each chosen for what it does best.
The principle: choose the language per workload layer, not per project.

### Orchestration layer: Python (PySpark)

Python controls the pipeline — reading data, calling UDFs, managing persists
and checkpoints, writing results. PySpark provides the API, shuffle management,
scheduling, and fault tolerance.

This is where users interact with the library:

```python
from distributed_curator import partition_aware_deduplicate

result = partition_aware_deduplicate(spark, input_df)
```

Python is the orchestration choice because it's where the Spark community lives.
The heavy computation happens in the layers below.

### Compute layer: Cython (C) — shingle hashing

Step 1 starts by hashing every character n-gram (shingle) in each document
using MurmurHash3. This is a tight loop over bytes — one hash per shingle,
millions of shingles per executor batch.

Cython compiles this loop to C, calling MurmurHash3 directly without
Python object overhead per shingle. The compiled `.so` runs on each
executor as if it were a native C library.

### Compute layer: NumPy (C/SIMD) — MinHash signatures

After Cython produces the raw shingle hashes, NumPy computes the 64-element
MinHash signature via broadcast XOR and min reduction:

```python
base_expanded = base_hashes[:, np.newaxis]       # (num_shingles, 1)
mixed = (base_expanded ^ hash_seeds) & 0xFFFFFFFF  # (num_shingles, 64)
signature = np.min(mixed, axis=0)                  # (64,)
```

This is a single vectorized operation — NumPy dispatches it to SIMD instructions
(ARM NEON on Graviton, SSE/AVX on x86) that process multiple hashes per CPU cycle.

We benchmarked this against a Scala UDF with equivalent logic. The Scala version
used JVM scalar loops — one hash at a time, no SIMD. NumPy was 3.4× faster
because the broadcast XOR + min pattern maps perfectly to SIMD lanes.

### The bridge: Arrow IPC

PySpark's `pandas_udf` transfers data between the JVM and Python using
Apache Arrow's IPC format. Arrow defines a columnar memory layout —
how bytes are arranged in memory — so both sides read the same buffer
without serialization or copying.

This is what makes the Cython + NumPy compute layer practical.
Without Arrow, every row would be serialized from JVM -> Python pickle -> Python objects,
and NumPy's SIMD advantage would be lost to serialization overhead.

### Compute layer: Scala (JVM UDFs) — partition-local logic

Steps 2, 3, 4, and 5a run as Scala UDFs or JVM helpers:

- **Step 2** (`ComputePartitionAssignmentsUDF`): band hash -> partition ID mapping
- **Step 3** (`IdentityRepartition`): physical repartition by integer key
- **Step 4** (`ProcessPartitionLocallyUDF`): pairwise comparison within partition
- **Step 5a** (`PartitionAwareUnionFindUDF`): partition-local union-find
- **Step 5b** (`runGlobalUnionFindFromDriver`): global union-find on driver using Eclipse Collections `LongLongHashMap`

These steps involve branchy logic — iterating over documents, HashMap lookups,
if/else per document pair, building data structures. This is where JVM excels
and Python suffers from serialization overhead.

The Scala UDFs use boxing-free primitives throughout: `Array[Long]` instead of
`Seq[Long]`, `while` loops instead of `for` comprehensions, direct index access
instead of `zip`. These micro-optimizations matter at scale — Step 4 dropped
from 2.1 minutes (Python) to 39 seconds (Scala) on 100 WET files.

### When to use which

| Pattern | Best language | Why |
|---------|--------------|-----|
| Vectorized math (XOR, min, broadcast) | NumPy (C/SIMD) | SIMD processes multiple values per cycle |
| Tight byte-level loops | Cython (C) | No Python object overhead |
| Branchy logic, HashMap, if/else | Scala (JVM) | Avoids Python ↔ JVM serialization |
| Pipeline orchestration, API | Python (PySpark) | Where users are |
| Driver-side data structures at scale | Scala + Eclipse Collections | Primitive types, no boxing overhead |
