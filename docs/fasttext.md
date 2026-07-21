# fastText models: downloads, sizes, licenses

The quality-scoring module uses two fastText models. **Neither is vendored in
this repository** — both are downloaded by the user and referenced by path
through `FastTextConfig`. Only ~85 KB toy models used by the test suite are
committed (`test/unit_test/fixtures/tiny_*_model.bin`).

## 1. Quality classifier (DCLM OH-2.5 + ELI5)

| | |
|---|---|
| Source | `mlfoundations/fasttext-oh-eli5` on Hugging Face |
| File | `openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin` |
| Size | **2.39 GB** |
| License | **MIT** (per the model card) |
| Labels | `__label__hq` / `__label__cc` |
| Reference threshold | `0.018112` (DCLM's published value; see below) |

This is the classifier used to produce DCLM-Baseline. It scores
"high quality" (OpenHermes 2.5 + Reddit ELI5) against "low quality"
(Common Crawl). We emit the `__label__hq` probability as `q_ft_score`.

DCLM's own config names the output key `fasttext_oh_eli5_vs_rw_v2_prob`, while
the published `dclm-baseline-1.0` dataset uses the longer
`fasttext_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train_prob`. Both are
just user-chosen key names for the same score; the pool I/O layer handles the
aliasing.

**The threshold is documented, not applied.** This layer emits scores; the
filter step is separate and user-configured. DCLM applies 0.018112 as a fixed
cutoff, and notes in their repo that computing a *percentile-based* threshold
over the corpus is the more principled approach — which is what
`percentile_filter` provides.

**Training data is not recoverable.** The OH-2.5 + ELI5 corpus used to train
this classifier was never released (open requests on both the HF discussion
and the dclm issue tracker). Users who want a custom classifier must supply
their own positive-class data; see the custom-classifier notes in the
training script.

## 2. Language identification (LID)

Two variants, both from fastText's official language-identification release,
covering **176 languages**, trained on Wikipedia / Tatoeba / SETimes:

| File | Size | Notes |
|---|---|---|
| `lid.176.bin` | ~126 MB (some mirrors report 131 MB) | faster, slightly more accurate |
| `lid.176.ftz` | **~917 KB** | compressed; small accuracy compromise |

License for both: **CC-BY-SA-3.0** (Creative Commons
Attribution-Share-Alike 3.0). Note this is a *share-alike* license, unlike
the MIT-licensed quality classifier — relevant if you redistribute the model
or a derivative. This is one reason neither model is vendored here.

**Prefer `lid.176.ftz` unless you have measured a need for the `.bin`.** At
~917 KB it is three orders of magnitude smaller, which removes LID from the
per-worker memory problem entirely (see below); the accuracy difference is
small for whole-document language ID, which is how this layer uses it.

Models expect UTF-8 input. Labels are ISO-639 codes prefixed with
`__label__`; we strip the prefix and emit `q_lid_lang` / `q_lid_score`.

## Memory: why this path is the oracle, not production

The Python scorer in `fasttext_scoring.py` loads the model inside a
`pandas_udf`, so it lives in the **Python worker process**. Spark starts one
worker per concurrent task slot, so the model is duplicated per slot:

| | 2.39 GB quality model | `lid.176.ftz` |
|---|---|---|
| Per executor (4 task slots) | ~9.6 GB | ~3.7 MB |
| Per r6gd.8xlarge node (28 slots) | **~67 GB** | ~26 MB |

The production scorer (Phase 2b) loads the model once per **executor** on the
JVM heap, where task slots are threads sharing one address space — one copy
by construction, and legible to YARN's container accounting. Use the Python
path for fixtures, correctness work, and small runs.

## Python environment caveat

The official `fasttext` package (0.9.3, last released 2022) is **incompatible
with NumPy 2.x**: `model.predict()` raises
`ValueError: Unable to avoid copy while creating an array as requested`.
Training, `save_model`, `labels`, and `get_input_matrix()` are unaffected —
only `predict()`, which is the call this layer depends on.

Options:

1. **Isolated `numpy<2` virtualenv** — what
   `scripts/generate_fasttext_fixtures.py` uses to produce the golden values.
   Keeps the pin out of the repo's environment.
2. **`fasttext-numpy2`** — MIT, a one-line community fork, wheels through
   CPython 3.13. Verified to return bit-identical scores to the official
   package. Convenient for development; deliberately not used to generate the
   correctness fixtures.

The test suite skips the model-dependent tests (with an explicit reason)
when neither is available, so the suite stays green in environments without
a working fasttext.

## Getting the models

```bash
# Quality classifier (2.39 GB, MIT)
huggingface-cli download mlfoundations/fasttext-oh-eli5 \
    openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin --local-dir models/

# Language ID (~917 KB compressed, CC-BY-SA-3.0)
curl -O https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz
```

Then point the config at the local paths:

```python
from distributed_curator.quality.config import FastTextConfig

config = FastTextConfig(
    quality_model_path="/mnt/models/openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin",
    lid_model_path="/mnt/models/lid.176.ftz",
)
```

Paths must resolve on **every executor**. This oracle path does not
distribute models; cluster-wide distribution (S3 pull to node-local disk) is
part of the production scorer's integration.

# fasttext convert
## Offline model conversion (JVM artifacts)
 
The JVM scorer never parses fastText's binary `.bin` format. It loads a small set
of flat artifacts produced offline by `distributed_curator.quality.fasttext_convert`.
The official `fasttext` Python package is the parser; we never reimplement it.
 
Not every fastText model can be converted — see "Confirmed production models"
below for which of the two models this pipeline currently uses actually
convert, and why one doesn't.
 
### Why a converter exists
 
The `.bin` format is only readable through fastText's own C++ loader. Rather than
port that loader to the JVM, we export the two matrices, the vocabulary, the
labels, and the config the scorer needs into formats the JVM reads directly:
row-major little-endian `float32` for matrices, UTF-8 line-oriented text for
vocab and labels, JSON for config and checksums.
 
### Artifact layout
 
```
<out_dir>/
    manifest.json        config, matrix shapes, SHA-256 of every file, source .bin hash
    input_matrix.f32     row-major little-endian float32, (n_words + bucket, dim)
    output_matrix.f32    row-major little-endian float32, (n_labels, dim)
    vocab.txt            UTF-8, one word per line, input-matrix row order
    labels.txt           UTF-8, one label per line, output-matrix row order
```
 
The matrix files carry no header. The JVM loader reads shape, dtype, and byte
order from `manifest.json`, then maps each `.f32` file as
`ByteBuffer.order(LITTLE_ENDIAN).asFloatBuffer()`.
 
Row-order is a hard contract, not a convenience: `vocab.txt` line *i* is
`input_matrix` row *i*, and `labels.txt` line *i* is `output_matrix` row *i*.
Subword and word-n-gram vectors occupy the `bucket` rows appended after the
in-vocabulary words, so `input_matrix` has `n_words + bucket` rows — the
converter asserts this against the loaded model rather than assuming it.
 
### Supported models only
 
The converter reads the model's own training args and **hard-fails** on anything
the JVM scorer cannot represent, so an unsupported model is rejected at
conversion time rather than producing silently wrong scores at scale:
 
- **`loss` must be `softmax`.** With hierarchical softmax the output-matrix rows
  are internal Huffman-tree nodes, not labels, and reconstructing labels would
  require the tree — which a flat matrix does not carry.
- **`model` must be `supervised`.** The scorer performs classification only.
- **Quantized models are rejected.** Convert the unquantized `.bin`.
- **No vocab/label entry may contain a fastText separator byte**
  (space, `\n`, `\t`, `\v`, `\r`, `\f`, `\0`). A line-oriented file cannot
  round-trip one. fastText tokenizes on exactly these bytes, so a real
  vocabulary word can never contain one; the check exists to fail loudly if that
  assumption ever breaks.
### CLI usage
 
```bash
python -m distributed_curator.quality.fasttext_convert \
    /path/to/model.bin \
    /path/to/out_dir
```
 
On success the config block is printed to stdout. Flags:
 
- `--overwrite` — replace `out_dir` if it already exists.
- `--no-verify` — skip the bytewise reload check (not recommended).
By default the CLI runs `verify_export` after writing: it reloads the exported
artifacts and compares them **bytewise** against `get_input_matrix()` /
`get_output_matrix()`, and checks that `vocab.txt` / `labels.txt` reproduce
`get_words()` / `get_labels()` order exactly. Bytewise rather than `==` so `NaN`
and `-0.0` are compared honestly.
 
### Staging to S3
 
Converted artifacts are **not** committed to the repo. They are derived
(regenerable from the `.bin`), multi-GB, and consumed by EMR at runtime from S3 —
the same place the `.bin` models already live. Committing them would create a
second copy that can drift from what the cluster actually loads.
 
The converter writes locally only; upload is a separate documented step:
 
```bash
# Convert locally
python -m distributed_curator.quality.fasttext_convert \
    ./models/oh-eli5.bin \
    ./artifacts/oh-eli5
 
# Stage to S3 alongside the model artifacts
aws s3 cp ./artifacts/oh-eli5 \
    s3://<BUCKET>/<MODEL_PREFIX>/oh-eli5/ \
    --recursive
```
 
`manifest.json` records `source.sha256` — the SHA-256 of the `.bin` it was
converted from. That is the provenance link: a converted artifact set staged in
S3 is always traceable to the exact source model that produced it. If a staged
artifact set is ever in question, compare its manifest `source.sha256` against
the `.bin`.
 
### Confirmed production models
 
Both production `.bin` files were run through the converter and `verify_export`
as the actual acceptance test for the bytewise-reload gate.
 
| model      | dim | n_labels | wordNgrams | bucket    | minn | maxn | loss     | supported |
| ---------- | --- | -------- | ---------- | --------- | ---- | ---- | -------- | --------- |
| oh-eli5    | 100 | 2        | 2          | 2,000,000 | 0    | 0    | softmax  | yes       |
| lid.176    | —   | —        | —          | —         | —    | —    | **hs**   | **no**    |
 
**oh-eli5** — `n_words = 3,777,339`. Converted and verified successfully;
`input_matrix` rows = `n_words + bucket` = 5,777,339, confirming the row-count
contract against a real (not stub) model. `minn = maxn = 0`: no character-level
subwords. `wordNgrams = 2` means it does hash word *bigrams* into the `bucket`
range — a different mechanism from character subwords — so PR-7 needs whole-word
and whole-bigram lookups, not character-level hashing.
 
**lid.176** — `read_model_config` rejected it: `loss = hs` (hierarchical
softmax), not `softmax`. This is exactly the check working as designed — see
"Supported models only" above. `lid.176` was trained with hierarchical softmax
regardless of its label count being small; label count alone doesn't predict
loss type, which is why the converter checks the model rather than assuming
from context. No dim/bucket/minn/maxn values were read since conversion stops
at the loss check before those matter.
 
> **PR-7 sizing note (resolved):** the JVM scorer, as scoped, only needs to
> support **oh-eli5** — the only model that converts today. It needs whole-word
> and whole-bigram lookups only; **no subword/character hashing is required.**
> `lid.176` is out of scope for the current quality-scoring work: language ID
> is deferred to Phase 2 per the project's own phase-gated plan, and hierarchical
> softmax support (Huffman-tree export format + JVM tree-walking inference) is
> real, separate scope that should be sized when Phase 2 actually starts —
> not added speculatively here for a model nothing currently calls.
 
### Known limitation: hierarchical-softmax models are rejected by design
 
Not every fastText model can be converted today. Models trained with
hierarchical softmax (`loss = hs`) are rejected at conversion time with a clear
`UnsupportedModelError` rather than producing a matrix that looks valid but
scores nonsense (see "Supported models only" above for why). `lid.176`, the
standard fastText language-identification model, is one such model.
 
Supporting hierarchical softmax would require:
1. Exporting the Huffman tree structure itself (parent/child pointers per
   internal node, and which leaf maps to which label) — nothing in the current
   flat-matrix format carries this.
2. JVM tree-walking inference in PR-7's scorer, replacing the single
   matvec-and-softmax path with a root-to-leaf traversal.
This is real, scoped work, not a small extension — appropriately deferred
alongside the rest of Phase 2 language-ID work rather than folded into PR-6.
 
### Memory footprint
 
`get_input_matrix()` returns a copy, so conversion transiently holds the loaded
model plus a full `float32` copy of the input matrix — budget roughly 2× the
model size. Matrix writing is chunked, so nothing beyond that copy is allocated
during export. `verify_export` loads the model a second time (sequentially, not
concurrently with the first). For the ~2.39 GB production models this is
comfortable on the conversion box; it is not intended to run on the EMR
executors.