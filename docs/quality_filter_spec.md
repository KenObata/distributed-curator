# Tech Spec

## Overview
### Phase 0 — WARC/WET text extraction layer.**
Use resiliparse (cython) .mapPartition to extract only text from WARC.

#### why WARC format is better than WET form in quality scoring?
WARC gives you the raw HTML, so you control extraction. With the DOM available you can run a proper content extractor — trafilatura is the current standard (RefinedWeb and FineWeb both use it) — which uses tag structure, link density, and DOM position to isolate the main content block and discard chrome before any filtering happens. You also gain filtering signals that don't exist in WET: link-to-text ratio (high = nav/spam page), tag density, <code> blocks you can deliberately preserve, metadata like lang attributes, and clean paragraph boundaries that make heuristics like "fraction of lines ending in punctuation" actually meaningful. FineWeb's ablations showed WARC + trafilatura produced measurably better models than the same pipeline on WET — the extraction step alone was one of their larger wins.

#### Why dedupe is okay to use WET but quality scorre should use WARC?
WARC dumps are roughly 3–4× larger than WET, and HTML parsing per document is far more compute than reading pre-extracted text — for a dedup-focused pipeline like yours, WET is a defensible choice because MinHash over slightly noisy text still finds the duplicates, and the boilerplate is itself near-identical across pages (arguably helping template-clone detection). But for quality scoring specifically, garbage-in applies: KenLM perplexity and classifier scores computed over text that's 30% nav-menu noise are measuring the extractor as much as the page

### conclusion
- "why WET in dedupe but WARC for quality score? " → dedup is robust to extraction noise; quality filtering isn't, which is why FineWeb pays the WARC tax

### resiliparse (cython) vs trafilatura (Python)
- resiliparse is faster per page but slightly more aggressive in what it drops; DCLM used that one. Usage is about as simple as it gets: trafilatura.extract(html) returns the clean text or None if the page has no real content — that None itself acts as a first quality filter.
  - resiliparse keeps about 10% more text than trafilatura (some of it useful, like section titles and dates) and runs roughly 8× faster
    - ref: https://arxiv.org/pdf/2406.11794
  - On model quality, one recent ablation comparing all four extractors found aggregate benchmark scores of 44.29% for trafilatura, 44.02% for resiliparse, and 44.74% for jusText, versus 40.57% for WET
    - ref: https://arxiv.org/pdf/2511.18054

### Phase 1 — Heuristic layer.
 Implement the approved heuristic set as score columns. Include: a config object enabling/disabling each rule; golden-file tests with hand-computed expected scores on ~20 crafted documents (clean prose, boilerplate, code, gibberish, repeated lines, non-English); a benchmark harness measuring rows/sec/core on sample WET data. Deliverable: scores match golden files; benchmark numbers reported.

#### Phase 1a 
 create 12 native q_heur_* columns. we implement this in Cython because SQL expression scan the same document 12 times, but Cython allows to scan only 2 times as for loop.

##### why can't we import from datatrove directly.
- datatrove's PUNCTUATION_SET, copied exactly from `datatrove/utils/text.py` 
- we need to ensure determinism. copying is fine.

#### Phase 1b 
 9 Gopher n-gram repetition columns via Cython kernel + pandas_udf 
 - top_ngram_char_frac_3: "how dominant is the single most repeated short phrase?"
 - dup_ngram_char_frac_5: it means "how much of the document is covered by any repeated long phrase?"

ex) `buy cheap shoes online buy cheap shoes online buy cheap shoes today and save`
total char = 76

top_ngram_char_frac_3: count every 3-word window, take the most frequent one:
buy cheap shoes appears 3× (word positions 1, 5, 9). Its char length is 15. So 15 × 3 / 76 = 0.592

dup_ngram_char_frac_5: it means "how much of the document is covered by any repeated long phrase?"
  `buy cheap shoes online buy` appeans from position 5. it consists of 26 chars
  so 26/76 = 0.342

we do this for top_ngram_char_frac_2 to top_ngram_char_frac_4, dup_ngram_char_frac_5 to dup_ngram_char_frac10

**how to interpret this document?**
this document repeats a 4-word cycle about 2.5 times, so duplication is visible up to n=7 and vanishes at n=8


#### diff between 12 heauristic vs 9 n-gram.
- 12 Phase-1a heuristics are stateless expressions
  - each one is a closed-form function of the string, expressible as a single Catalyst expression tree (length, split, regexp_count, arithmetic).
  - they compile into WholeStageCodegen: JIT'd JVM bytecode operating directly on Tungsten rows.

- 9 n-gram repetition columns are stateful algorithms.
  - require tokenizing to words, building per-document frequency maps, membership hash sets, and the skip-ahead logic that advances past an already-counted duplicated span. That's branchy, data-structure-heavy, positional computation. Catalyst has no efficient primitive for it
  - each of the 9 columns as an independent SQL expression re-tokenizes and re-hashes the document from scratch — Catalyst can't share a frequency map across expression trees.

#### why do we need to hash from n-gram in phase 1b (pass5)? 
To count repeats you must compare windows against each other. Comparing strings means building "buy cheap shoes" (15 chars) for every window, then string-comparing on every hash-table probe. Hashing collapses each window to one 64-bit integer: equal windows → equal integer, so counting becomes integer lookups in a C array.
Without hashing you'd allocate ~44× the document's characters as strings; with it, pass 5 touches no characters at all 

##### why phase 1a (pass 1 - 4) doesn't need polinomial hash ?
Because passes 1–4 never need to combine smaller pieces into bigger ones — each unit they compare is already a fixed, non-overlapping piece of text.


### Phase 2 — fastText layer.**
 `mapPartitions` scoring with per-partition model load; model file distribution mechanism (SparkFiles or S3 pull — match repo conventions); training script that reproduces the DCLM-Baseline classifier from public data (document every data source and step); tests with a tiny fixture model committed to the repo. Deliverable: scoring a sample partition matches single-machine fastText output exactly; throughput benchmark.

#### fastText - which model to use
we'll use fasttext-oh-eli5 2.39 GB

### How to load 2.4gb model to each executor/core
- (a) JVM mapPartitions — one copy/executor, but a fastText JVM impl (fastText4j or your own) + parity gate + Scala surface.
  
- (b) Python + memmap — one copy/node, pure Python, ~15 lines + parity gate, no Scala.
  - .bin contains the model and we only need the model so we need to unzip .bin
  - unzip .bin model and convert into custom model
- (c) Python naive — 28 copies/node, budget the RAM, ship it; optimize only if it OOMs.
  - this will OOM easily.

**Decision: JVM scorer fed by an offline model converter.** The model is
loaded once per *executor* on the JVM heap — task slots are threads sharing
one address space, so it's one copy per executor by construction (vs. the
Python path's one copy per *worker process* = 28 copies ≈ 67 GB/node on an
r6gd.8xlarge). The per-executor copy is also legible to YARN container
accounting.

Rather than parse fastText's `.bin` format on the JVM (fastText4j's mistake —
unmaintained since 2019, open UTF-8/subword bug, not thread-safe), we split
the work:
- **PR-6 — offline converter (Python).** The official fastText library parses
  the `.bin` and exports raw matrices + vocab + a JSON manifest. Format: flat
  little-endian float32 per matrix + `manifest.json` (shapes, dtype,
  endianness, per-file SHA-256, source-model hash, fastText args). Chosen over
  `.npy` (JVM would need a header parser) and Arrow IPC (a library dependency
  for a load-once-at-startup blob). Scoped to `oh-eli5` (flat softmax) only;
  `lid.176` uses hierarchical softmax — a different inference algorithm
  (Huffman-tree walk), deferred to Phase 2 language ID, gated by a
  `loss == "softmax"` assert in the converter.
   - the converted will output these:
     - vocab.txt
       - each line in input_matrix.f32 is a vector about a word representation.
       - vocab.txt is a list of words corresponding to each row in input_matrix.f32
       - nput_matrix.f32 is just raw numbers. a big grid of floats with no labels attached. Row 47,382 is some word's vector, but the file itself doesn't say which word. vocab.txt is the missing half: line i names the word that lives at row i. At inference time, scoring a document means: for each word in it, look up its row index (via vocab.txt), pull that row out of the matrix, and use the vector. Without vocab.txt, input_matrix.f32 is 5.7 million anonymous vectors — useless.
     - labels.txt
       - output_matrix.f32 has 2 rows. But which row is __label__hq and which is __label__cc? fastText assigns label indices internally based on training-time bookkeeping (label-encounter order, not alphabetical, not anything you can predict from outside). If row 0 happens to be __label__cc and you assumed it was __label__hq, every quality score PR-7 produces would be inverted.
    - manifest.json
      - JVM scala mapPartition() loads this file. 
      - How to read the raw bytes: shape (rows, cols) and dtype/byte-order for each .f32 file
      - How to run inference: dim, wordNgrams, bucket (needed to hash bigrams into the right range), minn/maxn (confirmed 0 for oh-eli5, so no subword hashing branch needed), label prefix.
      - Provenance/integrity: source.sha256 traces a staged S3 artifact set back to the exact .bin it came from

- **PR-7 — JVM inference (Scala).** Loads only converted artifacts
  (`ByteBuffer.order(LITTLE_ENDIAN).asFloatBuffer()`, reshaped and
  SHA-verified via the manifest). Never parses the `.bin`. Hashes over UTF-8
  `byte[]` (never Java `char`), immutable shared matrices + per-thread scratch.
  Parity gate: matches the Python oracle's committed fixtures within 1e-6,
  including CJK/emoji.
- **PR-8 — Spark wiring + S3 model distribution + throughput/RSS benchmark.**

#### Why JVM supports shared access?
One JVM executor is one OS process with one heap. Its task slots are threads, and threads share the process's address space by definition. So a model loaded into that heap is visible to all 7 threads at one memory address

#### Process vs thread (refresher)
- process A (shared memory space)
  - thread1
  - thread2

#### Then Why Spark python fork a process per task, not per executor?
Short answer: because CPython has a GIL

- python uses GIL which enforces one thread at one time per process.
- in Spark (pyspark), in order to keep parallel run of each core, one core means one thread and one thread runs at a time per process. So only way to run cores at the same time is to create python process per core. it blows up the numebr of memory if one process one fastText model.

#### Why GIL enforces one thread at a time per process?
CPython manages memory by counting references. Every object has an integer ob_refcnt; when it hits zero, the object is freed. That counter is incremented and decremented constantly — every assignment, every function argument, every list append.
The problem is that ob_refcnt++ is not one CPU instruction. It's load, add, store.

**Why not just make refcounts atomic?**
It was tried. The problem is that atomic increments are dramatically more expensive than plain ones.  "Gilectomy" (2016) removed the GIL this way and made single-threaded Python roughly 2× slower.

**How the JVM avoids all this**
The JVM never had refcounts. It uses a tracing garbage collector — the GC walks the object graph from roots and finds what's unreachable. There's no per-object counter being mutated on every access, so there's nothing to race on. It cost more upfront (write barriers, safepoints, GC pauses, a much harder VM to write), but it bought real multithreading. That's the trade CPython didn't make.

### why broadcast variables don't help here
- In JVM, broad cast variable is copying to each executor
- in Python UDF/mapPartition(), from broadcasted executor .bin model,
  Each Python worker reads those bytes and unpickles them into its own private heap, caching the result in that worker's _broadcastRegistry dict.
  => 28 copies too.

### what is the difference between __label__hq vs __label__cc
DCLM uses the __label__hq score
- Positive class = the reference data (OH-2.5 + ELI5, or Wikipedia, or whichever variant is being ablated) — this is what you'd be calling __label__hq.
- Negative class = random RefinedWeb-reproduction Common Crawl documents — this is what you'd be calling __label__cc.

### Model Converter 
To enable model load in scala JVM, we can't use .bin 

### Phase 3 — KenLM layer (won't do).
 EMR bootstrap/install docs and script; per-partition binding load; perplexity scoring column; graceful degradation (clear error, not silent nulls) when the native lib is absent; tests gated to skip cleanly where KenLM isn't installed. Deliverable: perplexity matches reference KenLM CLI output on fixtures.

 Won't do because DCLM papers already proved that good Heuristic and fastText achieves the same result against becnhmark done by KenLM layer.

### diff b/w KenLM (LLM) and fastText (classic ML)
- KenLM is a generative language model, not a classifier. A 5-gram model trained on Wikipedia learns P(word | previous 4 words), and perplexity measures how surprised the model is by your document. It needs only one corpus (the "good" one) — no negatives, no labels. What it detects is fluency: gibberish, keyword-stuffed SEO spam, scrambled boilerplate, and machine-mangled text all have high perplexity because their word sequences are statistically improbable. But it has no notion of content value — a fluent, grammatical product listing or a clickbait article scores great perplexity.
- fastText is a discriminative classifier. It needs both positives and negatives, and it learns whatever separates them — which in practice is largely topical and stylistic signal: vocabulary distribution, n-grams that look encyclopedic vs. commercial. It can catch the fluent-but-worthless content KenLM waves through. But its weakness is the mirror image: it's only as good as the positive set's definition of quality, and it can be fooled by spam that mimics the right vocabulary.

## Phase 4 — Composition + filter step.
 A `QualityPipeline` that chains any subset of layers; a filter step supporting **two threshold modes**: (a) static — user-supplied SQL expression over score columns, and (b) **dynamic percentile thresholding** — "keep top X% by score" with the cutoff computed on-corpus via `approxQuantile` (this is a first-class feature, not an example: DCLM's own tooling ships a hardcoded threshold and lists dynamic percentile computation as an open gap — this feature is a headline differentiator; document the relativeError/accuracy tradeoff and benchmark the quantile pass separately); an end-to-end example notebook/script: WET sample → all 2 score layers (heauristic + astText) → dynamic-threshold filter → summary stats (score distributions, retention rate per rule).

## Phase 5 — Benchmark + release
Comparative benchmark mirroring the dedup module's, with two layers: (a) scoring throughput and cost-per-TB on a multi-node EMR run vs. datatrove's quality filters on the same sample, and (b) **whole-pipeline benchmark — the headline number**: dedup + scoring + dynamic-threshold filter composed as a single Spark job (one scan, no intermediate materialization) vs. the same stages run as sequential datatrove-style passes; report end-to-end wall-clock and cost. Plus score-agreement analysis vs. DCLM-Baseline classifier on a common sample; README/docs to release quality; version tag. (Positioning note for docs: do not claim shuffle/coordination advantages for scoring itself — it is map-only; the claims are single-pass pipeline composition, dynamic thresholding, and cost economics.)

## Discussion
### Spark vs Ray
- Ray is good when the pipeline has a mix of GPU and CPU nodes.
- Why Spark: all of these phases don't require GPU tasks. 
  - Spark remains the entrenched standard for exactly the workloads text curation mostly is

### Phase 4: Design of drop early or not.
make the pipeline an ordered list of steps, where a step is either a scorer or a filter. 
[heur, filter(expr), fasttext, percentile_filter(p)] is DCLM-shaped cascade
[heur, fasttext, filter(expr)] is score-all. 

### Language Identifier (LID)
lid.176.bin is fastText's pretrained language-ID model, published by Meta at fasttext.cc. Breaking down the name: lid = language identification, 176 = it distinguishes 176 languages, .bin = the full-precision fastText binary (~130 MB)
we feed it text, it returns labels like __label__en, __label__ja with confidence scores.



## Example

### phase 1a, 1b
- Input row: (doc_id="doc_42", text="…") — 158 chars, 31 whitespace tokens.
```
Best Widgets Online - Home
- Home
- Shop
Our widgets are the best widgets you can buy today.
Our widgets are the best widgets you can buy today.
Read more...
```

- Intermediate state (scratch, dropped before return): 
    _q_tmp_words (array of 31 strings), _q_tmp_nonsym_words (30), _q_tmp_quality_lines (6), _q_tmp_line_stats/_q_tmp_para_stats (structs {dups, chars}). These exist only inside the projection; nothing is materialized.

- Phase 1b adds 9 more columns to the same row
- Final heuristic row: 
  2 input columns + 21 score columns, same 1 row

### fastText layer
- Input:
  - 2 input columns + 21 score columns, same 1 row
- Final output:
  - q_ft_score
  - q_lid_lang="en" / q_lid_score=0.94.

- Spark internal
  - the 2.39 GB binary is loaded once per Python worker process (= core)
  - => [update] we decided to implement our custom fastText to deal with web Scale.
 

#### Diff b/w process heap vs mmap sharing
- Process heap (what fastText does today) 
  - allocations are private anonymous memory (malloc)
- mmap sharing (what would fix it)
  -  2.4 GB, once per node

# Previous Studies
IBM Data Prep Kit (Granite models) — open source, and the orchestration pattern is the lesson. DPK's modules were tested producing pretraining datasets for the Granite models, include quality/dedup/PII filter transforms, and are built on common frameworks for Spark and Ray. But look at how: each transform is a Python class over parquet chunks, and Ray or Spark wrappers are provided to readily scale out the Python implementations, with multi-step pipelines orchestrated by Kubeflow Pipelines. Their stated design goal was that users shouldn't need deep knowledge of Kubernetes, Ray, or Spark.

DPK uses Spark as an interchangeable scale-out scheduler for opaque Python functions which is a reasonable goal for their "long tail" audience, but it means Spark's engine is inert: no Catalyst expressions, no whole-stage codegen, no fused single-job composition, no approxQuantile over score columns; each transform is a separate materialize-to-parquet pass in a KFP DAG (structurally the same multi-pass shape as Dolma). Our design is the opposite bet: Spark as the engine. Native expressions for heuristics, score columns flowing through one job graph, the quantile as a plan-level aggregation.

# Project Milestones
## Chain A: heuristic layer

PR-1 · Phase 1a: native heuristic scoring — DONE, in review
12 q_heur_* native-expression columns, HeuristicConfig, 20-doc golden
fixtures, benchmark harness. Artifact: phase1a-quality-heuristics.patch.
Gate: Ken's review verdict.

PR-2 · Phase 1a amendment: punctuation-set provenance
datatrove commit SHA in the constant's comment; scripts/regen_punctuation_set.py;
parity test vs installed datatrove (skips cleanly; datatrove in dev extras only).
Gate: parity test passes with datatrove installed, skips without.

PR-3a · Cython version of heuristic scoring
tokenize() + the 12 heuristic metrics in Cython (goldens + differential test vs the merged SQL expressions on ~10K random docs, benchmark vs the 0.56 baseline)

PR-3 · Phase 1b: n-gram repetition kernel
9 columns (q_heur_top_ngram_char_frac_{2,3,4}, q_heur_dup_ngram_char_frac_{5..10})
as Cython kernel (word murmur64 → rolling combine → hash-set skip-ahead) inside a
pandas_udf. Golden fixtures vs datatrove reference; 64-bit hash-collision
approximation documented; benchmark: pure-Python vs Cython ms/doc in harness.
Gate: goldens match; Cython ≥10× pure-Python per-doc.

## Chain B: DCLM interop

PR-4 · DCLM pool format I/O
read_dclm_pool / write_dclm_pool (JSONL gz/zst + parquet; text, url,
metadata keys; score columns ↔ appended JSONL keys; alias
q_ft_score ↔ fasttext_oh_eli5_vs_rw_v2_prob). Round-trip tests on tiny fixtures.
Gate: byte-level round-trip on fixture shards.

## Chain C: fastText layer (decision: JVM production scorer, Python oracle)

PR-5 · Phase 2a: Python oracle scorer + fixtures
Naive fastText + LID scoring via mapPartitions (module-level model cache, the
known 28-copies caveat documented — this path is test oracle, not production).
Tiny fixture model committed; parity fixtures generated incl. Japanese + emoji
docs; model download docs with sizes/licenses (oh-eli5 2.39 GB — verify HF
license tag; lid.176.bin CC-BY-SA-3.0, never vendored).
Gate: fixture scores match single-machine fastText exactly.

PR-6 · Model conversion tool (offline)
Python: official fastText parses .bin → exports raw input/output matrices +
vocab + config (dim, wordNgrams, bucket, labels). Works for oh-eli5 and lid.176.
No quantized-model support (out of scope by design).
Gate: converted artifacts reload and reproduce get_*_matrix() bitwise.

PR-7 · Phase 2b: JVM inference (Scala)
Loads converted artifacts only (never parses Facebook binary format). UTF-8
byte[] hashing (never Java chars); immutable shared matrices + per-thread
scratch (thread-safe by design); one model instance per executor.
Gate: matches PR-5 parity fixtures within 1e-6, incl. non-ASCII docs.

PR-8 · fastText Spark integration + benchmark
Column API wiring (q_ft_score, q_lid_lang, q_lid_score); S3-pull model
distribution to node-local disk; EMR memory-config notes; throughput benchmark
(rows/sec/core) + worker/executor RSS measurement on pilot.
Gate: benchmark numbers reported; memory within executor budget.

## Chain D: composition + filtering

PR-9 · QualityPipeline: ordered steps + static filter
Pipeline = ordered list of scorer|filter steps (no mode flag). Static filter =
SQL expression step; retention rate reported per filter step; scorers never
drop rows (contract test). Docs: "analysis mode" vs "DCLM-style cascade"
recipes with percentile-population caveat.
Gate: both recipe shapes run end-to-end on sample data.

PR-10 · Dynamic percentile filter
percentile_filter(p) via approxQuantile over narrow projection;
persist(DISK_ONLY) before quantile; relativeError/accuracy tradeoff
documented; quantile pass benchmarked separately.
Gate: cutoff matches exact quantile within documented relativeError on fixtures.

PR-11 · End-to-end example + summary stats
Script/notebook: sample → heuristics → n-grams → LID+fastText →
percentile filter → score distributions + retention per rule.
Gate: runs on a fresh checkout with documented commands.

## Chain E: benchmark + release

PR-12 · Comparative scoring benchmark vs datatrove
Same sample, multi-node EMR: throughput + cost-per-TB, ours vs datatrove
quality filters. Positioning note enforced (map-only; no shuffle claims).
Gate: numbers in README with cluster config + methodology.

PR-13 · Whole-pipeline benchmark + score agreement
Headline: dedup + scoring + dynamic-threshold as one composed Spark job vs
sequential datatrove-style passes (wall-clock + cost). Score-agreement analysis
vs DCLM-Baseline classifier on common sample. Wording: scores materialize once
to NVMe; quantile reads narrow projection (no "zero materialization" claim).
Gate: end-to-end numbers reproducible from scripts.

PR-14 · Release: docs polish + version tag
README to release quality; CHANGELOG; version bump + tag; PyPI publish.
Gate: fresh-install smoke test from PyPI.

Parallel track (later, own Phase 0 recon)

PR-15+ · WARC extraction module
Recon first: resiliparse vs trafilatura license check, FastWARC integration
design, 1 → 100 → 1K WARC cost pilot ($ /TB, docs/sec/core) before any
full-crawl commitment. Not a dependency of anything above.