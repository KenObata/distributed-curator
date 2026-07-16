# quality_kernel_test.py - parity and contract tests for the Cython heuristic kernel
"""Parity and contract tests for the Cython heuristic kernel.

The kernel must be indistinguishable from the native Spark-SQL implementation
(native_heuristics.py, the parity oracle):

1. Golden parity: kernel output matches every fixture in quality_golden.json
   (same fixtures, same derivation provenance as the native tests — see
   quality_heuristics_test.py's module docstring).
2. native_heuristics.py parity: kernel == native_heuristics.py, column by column, on seeded random
   adversarial documents (unicode whitespace, CJK, \\r\\n, bullet/ellipsis,
   newline runs, empty/whitespace/None docs). A 3,001-doc run of this
   generator found two real bugs in the native implementation during PR-3a
   development (ASCII-only \\s; \\R$ double terminator strip) — both now
   pinned as golden fixtures (ideographic_space, trailing_lf_cr).
3. Contract: same config behavior, same error behavior, same pass-through
   guarantees as the native path.
"""

import json
import math
import os
import random

import pytest
from pyspark.sql.types import StringType, StructField, StructType

from distributed_curator.quality import SCORE_COLUMN_GROUPS, HeuristicConfig, compute_heuristic_scores

kernel_mod = pytest.importorskip(
    "distributed_curator.quality.kernel.heuristic_kernel",
    reason="Cython kernel not built (python setup.py build_ext --inplace)",
)
from distributed_curator.quality.kernel_scoring import KERNEL_COLUMN_ORDER  # noqa: E402

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "quality_golden.json")

N_DIFFERENTIAL_DOCS = 500  # pre-commit-friendly; 3,001-doc run recorded in PR-3a


def values_equal(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def random_adversarial_doc(rng: random.Random) -> str | None:
    words = [
        "the",
        "of",
        "and",
        "widgets",
        "Buy",
        "now.",
        "•",
        "-",
        "###",
        "...",
        "…",
        "東京",
        "日本語のテキスト",
        "naïve",
        "🎉emoji",
        "12345",
        "word.",
        "I",
    ]
    seps = [" ", "  ", "\t", "\n", "\n\n", "\n\n\n", "\r\n", " \n ", "\u3000"]
    kind = rng.random()
    if kind < 0.02:
        return None
    if kind < 0.05:
        return ""
    if kind < 0.08:
        return rng.choice(["\n", "   ", "\n\n\n", "\t\r\n"])
    parts = []
    for _ in range(rng.randint(1, 80)):
        parts.append(rng.choice(words))
        parts.append(rng.choice(seps))
    doc = "".join(parts)
    if rng.random() < 0.3:  # inject duplicate lines
        lines = doc.split("\n")
        if len(lines) > 2:
            lines.insert(rng.randrange(len(lines)), lines[0])
            doc = "\n".join(lines)
    if rng.random() < 0.2:
        doc += rng.choice(["\n", "\r\n", "...", "\n\r"])
    return doc


class TestKernelGoldenParity:
    """Kernel matches every golden fixture value (no Spark required)."""

    def test_all_golden_docs_match(self):
        with open(FIXTURE_PATH, encoding="utf-8") as f:
            golden = json.load(f)
        for doc in golden:
            got = dict(zip(KERNEL_COLUMN_ORDER, kernel_mod.score_document(doc["text"])))
            for column, expected in doc["expected"].items():
                assert values_equal(got[column], expected), (
                    f"{doc['doc_id']}.{column}: kernel={got[column]}, golden={expected}"
                )


class TestKernelNativeHeuristicParity:
    """Kernel and native implementations agree column-by-column on random docs."""

    def test_differential_parity(self, spark):
        rng = random.Random(20260713)
        docs = [random_adversarial_doc(rng) for _ in range(N_DIFFERENTIAL_DOCS)]
        schema = StructType([StructField("doc_id", StringType(), False), StructField("text", StringType(), True)])
        df = spark.createDataFrame([(f"d{i}", t) for i, t in enumerate(docs)], schema)

        native = {r["doc_id"]: r.asDict() for r in compute_heuristic_scores(df, implementation="native").collect()}
        kernel = {r["doc_id"]: r.asDict() for r in compute_heuristic_scores(df, implementation="kernel").collect()}

        for i in range(len(docs)):
            for column in KERNEL_COLUMN_ORDER:
                n, k = native[f"d{i}"][column], kernel[f"d{i}"][column]
                assert values_equal(k, n), f"d{i}.{column}: kernel={k} native={n} text={docs[i]!r:.80}"


class TestKernelContract:
    """Kernel path honors the same API contract as the native path."""

    def test_disabled_groups_and_passthrough(self, spark):
        df = spark.createDataFrame([("d1", "the cat and the hat", "extra")], ["doc_id", "text", "meta"])
        config = HeuristicConfig(enable_dup_lines_paragraphs=False)
        result = compute_heuristic_scores(df, config=config, implementation="kernel")
        for col in SCORE_COLUMN_GROUPS["enable_dup_lines_paragraphs"]:
            assert col not in result.columns
        row = result.collect()[0]
        assert row["meta"] == "extra"
        assert row["q_heur_stopword_count"] == 2  # {the, and}
        assert not any(c.startswith("_q_") for c in result.columns)

    def test_none_text_all_null(self, spark):
        schema = StructType([StructField("doc_id", StringType(), False), StructField("text", StringType(), True)])
        df = spark.createDataFrame([("d1", None)], schema)
        row = compute_heuristic_scores(df, implementation="kernel").collect()[0]
        for column in KERNEL_COLUMN_ORDER:
            assert row[column] is None, f"{column} should be NULL for NULL text"

    def test_collision_raises(self, spark):
        df = spark.createDataFrame([("d1", "x", 1)], ["doc_id", "text", "q_heur_word_count"])
        with pytest.raises(ValueError, match="already exist"):
            compute_heuristic_scores(df, implementation="kernel")

    def test_unknown_implementation_raises(self, spark):
        df = spark.createDataFrame([("d1", "x")], ["doc_id", "text"])
        with pytest.raises(ValueError, match="implementation"):
            compute_heuristic_scores(df, implementation="rust")
