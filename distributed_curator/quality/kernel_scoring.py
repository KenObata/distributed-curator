# kernel_scoring.py - Spark integration for the Cython heuristic kernel
"""pandas_udf wrapper around the Cython heuristic kernel.

One Arrow batch in, one struct column of 12 scores out; the struct is then
expanded into the same flat q_heur_* columns the native implementation emits.
Validation (text column presence, collision refusal) mirrors
native_heuristics so the two paths are drop-in interchangeable.
"""

from __future__ import annotations

import logging

import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import DoubleType, IntegerType, StructField, StructType

from distributed_curator.quality.config import HeuristicConfig
from distributed_curator.quality.native_heuristics import SCORE_COLUMN_GROUPS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Canonical column order — MUST match score_document()'s return tuple order.
KERNEL_COLUMN_ORDER: tuple[str, ...] = (
    "q_heur_word_count",
    "q_heur_mean_word_len",
    "q_heur_hash_word_ratio",
    "q_heur_ellipsis_word_ratio",
    "q_heur_bullet_line_frac",
    "q_heur_ellipsis_line_frac",
    "q_heur_alpha_word_frac",
    "q_heur_stopword_count",
    "q_heur_dup_line_frac",
    "q_heur_dup_line_char_frac",
    "q_heur_dup_para_frac",
    "q_heur_dup_para_char_frac",
)

_INT_COLUMNS = frozenset({"q_heur_word_count", "q_heur_stopword_count"})

_STRUCT_COLUMN = "_q_kernel_struct"

_RESULT_SCHEMA = StructType(
    [StructField(c, IntegerType() if c in _INT_COLUMNS else DoubleType(), True) for c in KERNEL_COLUMN_ORDER]
)


def _make_kernel_udf(config: HeuristicConfig):
    """Build the pandas_udf capturing config parameters by value.

    The kernel import happens inside the UDF body so it resolves on the
    executor (mirrors the shingle_hash_wrapper pattern). The decorated
    function carries NO type annotations: PySpark inspects annotations at
    runtime and `from __future__ import annotations` breaks that contract
    (documented repo lesson).
    """
    import pandas as pd
    from pyspark.sql.functions import pandas_udf

    stop_words = tuple(config.stop_words)
    bullet_prefixes = tuple(config.bullet_prefixes)
    ellipsis_suffixes = tuple(config.ellipsis_suffixes)

    @pandas_udf(_RESULT_SCHEMA)
    def kernel_scores(texts):
        try:
            from distributed_curator.quality.kernel.heuristic_kernel import score_document
        except ImportError:  # executor without the built package layout
            from heuristic_kernel import score_document  # pragma: no cover

        # Arrow delivers SQL NULL strings as NaN floats in the pandas Series;
        # normalize to None so the typed kernel signature accepts them.
        rows = []
        for t in texts:
            rows.append(
                score_document(t if isinstance(t, str) else None, stop_words, bullet_prefixes, ellipsis_suffixes)
            )
        return pd.DataFrame(rows, columns=list(KERNEL_COLUMN_ORDER))

    return kernel_scores


def compute_kernel_heuristic_scores(
    df: DataFrame,
    text_column: str = "text",
    config: HeuristicConfig | None = None,
) -> DataFrame:
    """Kernel-backed equivalent of compute_native_heuristic_scores.

    Emits the same enabled columns for the same config; disabled rule groups
    are computed inside the kernel (single pass computes all 12 regardless)
    but their columns are not attached.
    """
    # Fail fast on the driver if the extension isn't built anywhere at all.
    try:
        from distributed_curator.quality.kernel import heuristic_kernel  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "implementation='kernel' requires the compiled Cython extension. "
            "Build it with: python setup.py build_ext --inplace "
            "(or use implementation='native', which has no build requirements)."
        ) from e

    config = config or HeuristicConfig()

    if text_column not in df.columns:
        raise ValueError(f"text_column '{text_column}' not found in DataFrame columns: {df.columns}")

    enabled_columns = [col for flag, cols in SCORE_COLUMN_GROUPS.items() if getattr(config, flag) for col in cols]
    collisions = [c for c in [*enabled_columns, _STRUCT_COLUMN] if c in df.columns]
    if collisions:
        raise ValueError(f"Output/scratch columns already exist on input DataFrame: {collisions}")

    logger.info(f"Kernel heuristic scoring: {len(enabled_columns)} columns on text_column='{text_column}'")

    kernel_udf = _make_kernel_udf(config)
    df = df.withColumn(_STRUCT_COLUMN, kernel_udf(F.col(text_column)))
    for col in enabled_columns:
        df = df.withColumn(col, F.col(_STRUCT_COLUMN)[col])
    return df.drop(_STRUCT_COLUMN)
