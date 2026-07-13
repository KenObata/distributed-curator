# quality_heuristics_test.py - Golden-file and contract tests for the heuristic scoring layer
"""Golden-file and contract tests for the heuristic scoring layer.

How the expected scores in fixtures/quality_golden.json were derived
---------------------------------------------------------------------
1. Semantics were pinned against datatrove's source (gopher_quality_filter.py,
   gopher_repetition_filter.py, utils/text.py): char-fraction denominators are
   len(text); dup lines split via re.split(r"\\n+", text) UNSTRIPPED while
   paragraphs use re.split(r"\\n{2,}", text.strip()); stopword count is a
   distinct, case-sensitive set intersection; bullet/ellipsis line fractions
   use str.splitlines() semantics; the punctuation set is datatrove's
   PUNCTUATION_SET (281 chars — note "•" is NOT in it, so bullets count as
   words). One documented deviation: words are whitespace-split, not
   tokenizer-split (see heuristics.py module docstring).
2. Expected values were computed by an independent pure-Python oracle
   (~60 lines: str.split / re.split / set ops / find_duplicates logic)
   implementing those semantics. It shares no code with the Spark
   implementation, so agreement between the two is a differential test.
   JSON carries no comments, so each fixture doc instead has a
   "_derivation" field explaining its expected values.
3. A subset was verified by hand against the derivation notes:
   bullet_boilerplate word_count=21 (the "•"-is-a-word surprise),
   repeated_lines char math (3 dups x 28 chars / 164 = 0.5122...),
   clean_prose stopword_count=7, gibberish hash ratio 3/10.
4. Residual risk — a shared misreading of datatrove by both author-written
   implementations — is covered downstream: the punctuation-set parity test
   vs installed datatrove (dev extras), and Phase 5's score-agreement
   analysis against datatrove's actual outputs at corpus scale.

Regenerating: the oracle is intentionally not committed so the fixtures stay
frozen; to regenerate, reimplement the semantics above (or lift them from
the datatrove files cited) and dump docs + scores to the JSON schema used
here.
"""

import json
import math
import os

import pytest
from pyspark.sql.types import StringType, StructField, StructType

from distributed_curator.quality.native_heuristics import (
    SCORE_COLUMN_GROUPS,
    HeuristicConfig,
    compute_native_heuristic_scores,
)

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "quality_golden.json")

ALL_SCORE_COLUMNS = [c for cols in SCORE_COLUMN_GROUPS.values() for c in cols]


def load_golden():
    with open(FIXTURE_PATH, encoding="utf-8") as f:
        return json.load(f)


def assert_score_equal(doc_id: str, column: str, actual, expected):
    """NULL-aware comparison with float tolerance."""
    if expected is None:
        assert actual is None, f"{doc_id}.{column}: expected NULL, got {actual}"
    elif isinstance(expected, float):
        assert actual is not None, f"{doc_id}.{column}: expected {expected}, got NULL"
        assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12), (
            f"{doc_id}.{column}: expected {expected}, got {actual}"
        )
    else:
        assert actual == expected, f"{doc_id}.{column}: expected {expected}, got {actual}"


class TestHeuristicGoldenFiles:
    """Every score column matches hand-computed values on 20 crafted documents."""

    def test_all_columns_match_golden(self, spark):
        golden = load_golden()
        schema = StructType([StructField("doc_id", StringType(), False), StructField("text", StringType(), True)])
        df = spark.createDataFrame([(d["doc_id"], d["text"]) for d in golden], schema)

        result = {r["doc_id"]: r.asDict() for r in compute_native_heuristic_scores(df).collect()}

        assert len(result) == len(golden), "Row count changed — a layer must never drop rows"
        for doc in golden:
            actual_row = result[doc["doc_id"]]
            for column, expected in doc["expected"].items():
                assert_score_equal(doc["doc_id"], column, actual_row[column], expected)

    def test_null_text_yields_all_null_scores(self, spark):
        golden = {d["doc_id"]: d for d in load_golden()}
        assert golden["null_text"]["text"] is None, "fixture invariant"
        assert all(v is None for v in golden["null_text"]["expected"].values())


class TestHeuristicLayerContract:
    """Composition contract: pure transformation appending only its own columns."""

    def test_passes_through_arbitrary_extra_columns(self, spark):
        # Simulates composition after dedup: extra columns must survive untouched.
        data = [("d1", "The cat sat on the mat with a hat.", "rep_1", False, 42)]
        df = spark.createDataFrame(data, ["doc_id", "text", "representative_id", "is_duplicate", "shard"])

        result = compute_native_heuristic_scores(df)

        for col in ["doc_id", "representative_id", "is_duplicate", "shard"]:
            assert col in result.columns
        row = result.collect()[0]
        assert (row["representative_id"], row["is_duplicate"], row["shard"]) == ("rep_1", False, 42)
        assert not any(c.startswith("_q_tmp_") for c in result.columns), "scratch columns must be dropped"

    def test_disabled_groups_emit_no_columns(self, spark):
        df = spark.createDataFrame([("d1", "some text here")], ["doc_id", "text"])
        config = HeuristicConfig(enable_dup_lines_paragraphs=False, enable_stop_words=False)

        result = compute_native_heuristic_scores(df, config=config)

        for col in SCORE_COLUMN_GROUPS["enable_dup_lines_paragraphs"] + SCORE_COLUMN_GROUPS["enable_stop_words"]:
            assert col not in result.columns
        assert "q_heur_word_count" in result.columns

    def test_missing_text_column_raises(self, spark):
        df = spark.createDataFrame([("d1",)], ["doc_id"])
        with pytest.raises(ValueError, match="text_column"):
            compute_native_heuristic_scores(df)

    def test_output_column_collision_raises(self, spark):
        df = spark.createDataFrame([("d1", "text", 0.5)], ["doc_id", "text", "q_heur_word_count"])
        with pytest.raises(ValueError, match="already exist"):
            compute_native_heuristic_scores(df)

    def test_custom_text_column_name(self, spark):
        df = spark.createDataFrame([("d1", "the words and the content")], ["doc_id", "content"])
        result = compute_native_heuristic_scores(df, text_column="content")
        row = result.collect()[0]
        assert row["q_heur_word_count"] == 5
        assert row["q_heur_stopword_count"] == 2  # {"the", "and"}
