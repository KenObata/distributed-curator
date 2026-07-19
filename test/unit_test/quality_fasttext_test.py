# quality_fasttext_test.py - parity and contract tests for the fastText scoring layer
"""Parity and contract tests for the fastText scoring layer (Phase 2a oracle).

Expected values in fixtures/fasttext_golden.json were produced by the
OFFICIAL fastText bindings via scripts/generate_fasttext_fixtures.py, run in
an isolated ``numpy<2`` virtualenv (the official package's ``predict()`` is
broken on NumPy 2.x — see that script's docstring). The fixtures are
committed, so regenerating them is never required to run these tests.

The tests do, however, need a *working* fasttext to exercise our scorer.
They skip cleanly when the import is missing or when ``predict()`` raises the
NumPy 2 incompatibility, so a modern environment without a compatible
fasttext does not fail the suite.

Fixture models (tiny_quality_model.bin, tiny_lid_model.bin; ~85 KB each) are
committed and were trained by the same script's sibling recipe documented in
docs/fasttext.md. The production models are never vendored.
"""

import json
import math
import os

import pytest
from pyspark.sql.types import StringType, StructField, StructType

from distributed_curator.quality.config import FastTextConfig
from distributed_curator.quality.fasttext_scoring import (
    compute_fasttext_quality_scores,
    compute_language_scores,
    normalize_text,
)

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
QUALITY_MODEL = os.path.join(FIXTURE_DIR, "tiny_quality_model.bin")
LID_MODEL = os.path.join(FIXTURE_DIR, "tiny_lid_model.bin")
GOLDEN = os.path.join(FIXTURE_DIR, "fasttext_golden.json")


def _fasttext_usable() -> tuple[bool, str]:
    """True when fasttext is importable AND predict() works (NumPy 2 check)."""
    try:
        import fasttext
    except ImportError:
        return False, "fasttext not installed"
    try:
        fasttext.load_model(LID_MODEL).predict("hello world")
    except Exception as exc:
        return False, f"fasttext predict() unusable in this environment: {type(exc).__name__}"
    return True, ""


_USABLE, _SKIP_REASON = _fasttext_usable()
requires_fasttext = pytest.mark.skipif(not _USABLE, reason=_SKIP_REASON)


def values_equal(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    # fastText probabilities round-trip through JSON and Arrow float64
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def load_golden() -> list[dict]:
    with open(GOLDEN, encoding="utf-8") as f:
        return json.load(f)


def fixture_config() -> FastTextConfig:
    return FastTextConfig(quality_model_path=QUALITY_MODEL, lid_model_path=LID_MODEL)


class TestNormalizationSemantics:
    """DCLM-verbatim input normalization (no model required)."""

    def test_joins_lines_with_single_space(self):
        assert normalize_text("a\nb") == "a b"
        assert normalize_text("a\r\nb") == "a b"

    def test_strips_before_splitting(self):
        assert normalize_text("  a\nb  \n") == "a b"

    def test_breaks_on_exotic_separators_like_splitlines(self):
        # The distinction from replace("\n", " "): str.splitlines() also
        # breaks on these, and DCLM uses splitlines().
        for sep in ("\v", "\f", "\x1c", "\x85", "\u2028", "\u2029"):
            assert normalize_text(f"a{sep}b") == "a b", f"separator {sep!r} not handled"

    def test_no_newlines_survive(self):
        # fastText's predict() rejects embedded newlines outright.
        assert "\n" not in normalize_text("a\nb\nc\n")


@requires_fasttext
class TestFastTextGoldenParity:
    """Spark scorer reproduces the official-fastText oracle values exactly."""

    def test_quality_scores_match_golden(self, spark):
        golden = load_golden()
        schema = StructType([StructField("doc_id", StringType(), False), StructField("text", StringType(), True)])
        df = spark.createDataFrame([(d["doc_id"], d["text"]) for d in golden], schema)
        rows = {r["doc_id"]: r for r in compute_fasttext_quality_scores(df, config=fixture_config()).collect()}
        for doc in golden:
            expected = doc["expected"]["q_ft_score"]
            actual = rows[doc["doc_id"]]["q_ft_score"]
            assert values_equal(actual, expected), f"{doc['doc_id']}.q_ft_score: got {actual}, oracle {expected}"

    def test_language_scores_match_golden(self, spark):
        golden = load_golden()
        schema = StructType([StructField("doc_id", StringType(), False), StructField("text", StringType(), True)])
        df = spark.createDataFrame([(d["doc_id"], d["text"]) for d in golden], schema)
        rows = {r["doc_id"]: r for r in compute_language_scores(df, config=fixture_config()).collect()}
        for doc in golden:
            for column in ("q_lid_lang", "q_lid_score"):
                actual = rows[doc["doc_id"]][column]
                assert values_equal(actual, doc["expected"][column]), (
                    f"{doc['doc_id']}.{column}: got {actual}, oracle {doc['expected'][column]}"
                )

    def test_newline_normalization_is_score_invariant(self, spark):
        """Same text with different line breaks must score identically.

        This is the behavioral consequence of DCLM's normalization, and the
        reason we copied it verbatim rather than inventing our own.
        """
        base = "explain the reason carefully because evidence matters"
        variants = [base, base.replace(" ", "\n", 1), base.replace(" ", "\r\n", 1), f"  {base}  "]
        df = spark.createDataFrame([(f"v{i}", t) for i, t in enumerate(variants)], ["doc_id", "text"])
        scores = [r["q_ft_score"] for r in compute_fasttext_quality_scores(df, config=fixture_config()).collect()]
        assert all(values_equal(s, scores[0]) for s in scores), scores


@requires_fasttext
class TestFastTextContract:
    """Same DataFrame contract as the other scoring layers."""

    def test_null_text_yields_null_scores(self, spark):
        schema = StructType([StructField("doc_id", StringType(), False), StructField("text", StringType(), True)])
        df = spark.createDataFrame([("d1", None)], schema)
        config = fixture_config()
        row = compute_fasttext_quality_scores(df, config=config).collect()[0]
        assert row["q_ft_score"] is None
        row = compute_language_scores(df, config=config).collect()[0]
        assert row["q_lid_lang"] is None
        assert row["q_lid_score"] is None

    def test_extra_columns_pass_through_and_no_scratch_leaks(self, spark):
        df = spark.createDataFrame([("d1", "widgets now", "keep")], ["doc_id", "text", "meta"])
        result = compute_language_scores(df, config=fixture_config())
        assert result.collect()[0]["meta"] == "keep"
        assert not any(c.startswith("_q_") for c in result.columns)

    def test_row_count_preserved(self, spark):
        df = spark.createDataFrame([(f"d{i}", "widgets now") for i in range(7)], ["doc_id", "text"])
        assert compute_fasttext_quality_scores(df, config=fixture_config()).count() == 7

    def test_custom_text_column(self, spark):
        df = spark.createDataFrame([("d1", "widgets now")], ["doc_id", "content"])
        result = compute_fasttext_quality_scores(df, text_column="content", config=fixture_config())
        assert "q_ft_score" in result.columns

    def test_missing_text_column_raises(self, spark):
        df = spark.createDataFrame([("d1", "x")], ["doc_id", "text"])
        with pytest.raises(ValueError, match="not found"):
            compute_fasttext_quality_scores(df, text_column="nope", config=fixture_config())

    def test_column_collision_raises(self, spark):
        df = spark.createDataFrame([("d1", "x", 0.5)], ["doc_id", "text", "q_ft_score"])
        with pytest.raises(ValueError, match="already exist"):
            compute_fasttext_quality_scores(df, config=fixture_config())


class TestConfigValidation:
    """Config errors are raised on the driver, before any model load."""

    def test_missing_quality_model_path_raises(self, spark):
        df = spark.createDataFrame([("d1", "x")], ["doc_id", "text"])
        with pytest.raises(ValueError, match="quality_model_path"):
            compute_fasttext_quality_scores(df, config=FastTextConfig())

    def test_missing_lid_model_path_raises(self, spark):
        df = spark.createDataFrame([("d1", "x")], ["doc_id", "text"])
        with pytest.raises(ValueError, match="lid_model_path"):
            compute_language_scores(df, config=FastTextConfig())

    def test_reference_threshold_documented(self):
        assert FastTextConfig.DCLM_REFERENCE_THRESHOLD == 0.018112
