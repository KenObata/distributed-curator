# fasttext_scoring.py - fastText quality + language-ID scoring (Python oracle path)
"""fastText-based scoring: DCLM quality classifier and language identification.

STATUS: this is the **oracle / reference** implementation, not the production
path. It runs the official ``fasttext`` Python bindings inside a pandas_udf,
which means the model is loaded once per *Python worker process*. Spark starts
one worker per concurrent task slot, so a 2.39 GB model becomes N copies per
node (28 on an r6gd.8xlarge with 28 task slots -> ~67 GB). The production
scorer loads the model once per *executor* on the JVM heap (Phase 2b); see
docs/fasttext.md. Use this path for correctness work, fixtures, and small
runs — not for full-crawl scoring.

Semantics are copied from DCLM's ``classify_fasttext_hq_prob``
(baselines/mappers/enrichers/quality_prediction_enrichers_calc_fasttext.py),
verified against the repo at PR-5 time:

- Text is normalized with ``" ".join(content.strip().splitlines())``. Note
  this is NOT ``replace("\\n", " ")``: ``str.splitlines()`` also breaks on
  \\v, \\f, \\r, \\x1c-\\x1e, \\x85, U+2028 and U+2029. fastText's predict()
  rejects embedded newlines, so some normalization is mandatory; we match
  theirs exactly rather than inventing our own.
- The score is read from the TOP prediction only (``k=1``) and inverted when
  that label is the negative class: ``hq = 1 - p if label == negative else p``.
  For a 2-label model this is equivalent to reading the positive label's
  probability up to float rounding — but DCLM computes the inverted form, so
  we do too, to keep score-agreement analysis exact.

Columns (see also the DCLM key aliases handled by the pool I/O layer):
- ``q_ft_score``   double  - P(high quality); DCLM key
  ``fasttext_oh_eli5_vs_rw_v2_prob``, reference threshold 0.018112
- ``q_lid_lang``   string  - top language label, ``__label__`` prefix stripped
- ``q_lid_score``  double  - that label's probability
"""

from __future__ import annotations

import logging

import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

from distributed_curator.quality.config import FastTextConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

QUALITY_SCORE_COLUMN = "q_ft_score"
LID_LANG_COLUMN = "q_lid_lang"
LID_SCORE_COLUMN = "q_lid_score"

_LID_STRUCT_COLUMN = "_q_lid_struct"
_LABEL_PREFIX = "__label__"

# Model cache, keyed by path. Populated lazily inside the UDF so it lives in
# the executor's Python worker process (one entry per process — the 28-copies
# caveat in this module's docstring).
_MODEL_CACHE: dict = {}


def _load_model(model_path: str):
    """Load (and cache) a fastText model in this worker process."""
    model = _MODEL_CACHE.get(model_path)
    if model is None:
        import fasttext

        model = fasttext.load_model(model_path)
        _MODEL_CACHE[model_path] = model
    return model


def normalize_text(content: str) -> str:
    """DCLM-verbatim input normalization (see module docstring)."""
    return " ".join(content.strip().splitlines())


def score_quality(model, content: str, negative_label: str) -> float:
    """DCLM-verbatim hq probability for one document."""
    labels, probs = model.predict(normalize_text(content))
    label, prob = labels[0], float(probs[0])
    return 1.0 - prob if label == negative_label else prob


def score_language(model, content: str) -> tuple[str, float]:
    """Top language label (prefix stripped) and its probability."""
    labels, probs = model.predict(normalize_text(content))
    label = labels[0]
    if label.startswith(_LABEL_PREFIX):
        label = label[len(_LABEL_PREFIX) :]
    return label, float(probs[0])


def _validate(df: DataFrame, text_column: str, new_columns: list[str]) -> None:
    if text_column not in df.columns:
        raise ValueError(f"text_column '{text_column}' not found in DataFrame columns: {df.columns}")
    collisions = [c for c in new_columns if c in df.columns]
    if collisions:
        raise ValueError(f"Output/scratch columns already exist on input DataFrame: {collisions}")


def compute_fasttext_quality_scores(
    df: DataFrame,
    text_column: str = "text",
    config: FastTextConfig | None = None,
) -> DataFrame:
    """Append ``q_ft_score`` (P(high quality)) using a fastText classifier.

    Args:
        df: input DataFrame.
        text_column: document text column.
        config: model path and label names. ``quality_model_path`` must be
            readable at that path on every executor (the oracle path does not
            distribute models; that is Phase 2b's job).

    Returns:
        ``df`` with ``q_ft_score`` appended. NULL text -> NULL score.
    """
    config = config or FastTextConfig()
    if not config.quality_model_path:
        raise ValueError("config.quality_model_path must be set to score quality")
    _validate(df, text_column, [QUALITY_SCORE_COLUMN])

    import pandas as pd
    from pyspark.sql.functions import pandas_udf

    model_path = config.quality_model_path
    negative_label = config.negative_label

    @pandas_udf(DoubleType())
    def quality_udf(texts):
        model = _load_model(model_path)
        return pd.Series(
            [score_quality(model, t, negative_label) if isinstance(t, str) else None for t in texts],
            dtype="float64",
        )

    logger.info(f"fastText quality scoring (oracle path) on text_column='{text_column}'")
    return df.withColumn(QUALITY_SCORE_COLUMN, quality_udf(F.col(text_column)))


def compute_language_scores(
    df: DataFrame,
    text_column: str = "text",
    config: FastTextConfig | None = None,
) -> DataFrame:
    """Append ``q_lid_lang`` and ``q_lid_score`` using a fastText LID model.

    NULL text -> NULL in both columns.
    """
    config = config or FastTextConfig()
    if not config.lid_model_path:
        raise ValueError("config.lid_model_path must be set to score language")
    _validate(df, text_column, [LID_LANG_COLUMN, LID_SCORE_COLUMN, _LID_STRUCT_COLUMN])

    import pandas as pd
    from pyspark.sql.functions import pandas_udf

    model_path = config.lid_model_path
    schema = StructType(
        [
            StructField(LID_LANG_COLUMN, StringType(), True),
            StructField(LID_SCORE_COLUMN, DoubleType(), True),
        ]
    )

    @pandas_udf(schema)
    def lid_udf(texts):
        model = _load_model(model_path)
        rows = [score_language(model, t) if isinstance(t, str) else (None, None) for t in texts]
        return pd.DataFrame(rows, columns=[LID_LANG_COLUMN, LID_SCORE_COLUMN])

    logger.info(f"fastText language ID (oracle path) on text_column='{text_column}'")
    df = df.withColumn(_LID_STRUCT_COLUMN, lid_udf(F.col(text_column)))
    df = df.withColumn(LID_LANG_COLUMN, F.col(_LID_STRUCT_COLUMN)[LID_LANG_COLUMN])
    df = df.withColumn(LID_SCORE_COLUMN, F.col(_LID_STRUCT_COLUMN)[LID_SCORE_COLUMN])
    return df.drop(_LID_STRUCT_COLUMN)
