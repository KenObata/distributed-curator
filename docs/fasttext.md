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
