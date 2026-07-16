# heuristics.py - public API for heuristic quality scoring (implementation dispatcher)
"""Heuristic quality-scoring layer: public entry point.

Two interchangeable implementations compute the same 12 ``q_heur_*`` columns
(schema and semantics documented in native_heuristics.py, which is the
reference implementation and parity oracle):

- ``implementation="native"`` (default): pure Spark SQL expressions. Zero
  build requirements; runs anywhere PySpark runs.
- ``implementation="kernel"``: single-pass Cython kernel via a pandas_udf.
  Faster per document; requires the compiled extension
  (``python setup.py build_ext --inplace`` or a built wheel).

Both are pure DataFrame -> DataFrame transformations appending only score
columns; rows are never dropped or reordered. Parity between the two is
enforced by golden and differential tests (quality_kernel_test.py).

The default flips to "kernel" once the n-gram columns land (Phase 1b/PR-3b)
and the full 21-column benchmark is recorded.
"""

from __future__ import annotations

import logging

from pyspark.sql import DataFrame

from distributed_curator.quality.config import HeuristicConfig
from distributed_curator.quality.native_heuristics import (
    SCORE_COLUMN_GROUPS as SCORE_COLUMN_GROUPS,  # re-export: part of the public API
)
from distributed_curator.quality.native_heuristics import (
    compute_native_heuristic_scores,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_IMPLEMENTATIONS = ("native", "kernel")


def compute_heuristic_scores(
    df: DataFrame,
    text_column: str = "text",
    config: HeuristicConfig | None = None,
    implementation: str = "native",
) -> DataFrame:
    """Append heuristic quality-score columns to ``df``.

    See native_heuristics.compute_heuristic_scores for the full column
    list, reference thresholds, and null semantics — identical across
    implementations.

    Args:
        df: input DataFrame.
        text_column: name of the document text column.
        config: per-rule enable flags and parameters. Defaults to all enabled.
        implementation: "native" (Spark SQL expressions) or "kernel"
            (compiled Cython via pandas_udf).

    Raises:
        ValueError: unknown implementation, missing text column, or output
            column collision.
        ImportError: implementation="kernel" without the compiled extension.
    """
    if implementation not in _IMPLEMENTATIONS:
        raise ValueError(f"implementation must be one of {_IMPLEMENTATIONS}, got '{implementation}'")

    logger.info(f"Heuristic scoring via implementation='{implementation}'")
    if implementation == "native":
        return compute_native_heuristic_scores(df, text_column=text_column, config=config)

    # local import: the kernel path (and its compiled-extension requirement)
    # is only touched when requested
    from distributed_curator.quality.kernel_scoring import compute_kernel_heuristic_scores

    return compute_kernel_heuristic_scores(df, text_column=text_column, config=config)
