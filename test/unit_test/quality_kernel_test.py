# quality_kernel_test.py - parity and contract tests for the Cython heuristic kernel
"""Parity and contract tests for the Cython heuristic kernel.

The kernel must be indistinguishable from the native Spark-SQL implementation
(native_heuristics.py, the parity oracle):

1. Golden parity: kernel output matches every fixture in quality_golden.json
   (same fixtures, same derivation provenance as the native tests — see
   quality_heuristics_test.py's module docstring).
2. Differential parity: kernel == native, column by column, on seeded random
   adversarial documents (unicode whitespace, CJK, \\r\\n, bullet/ellipsis,
   newline runs, empty/whitespace/None docs). A 3,001-doc run of this
   generator found two real bugs in the native implementation during PR-3a
   development (ASCII-only \\s; \\R$ double terminator strip) — both now
   pinned as golden fixtures (ideographic_space, trailing_lf_cr).
3. Contract: same config behavior, same error behavior, same pass-through
   guarantees as the native path.
"""

# Python 3.9 compatibility: PEP 604 unions (str | None) are evaluated at
# def time without this import, and the repo targets py39 (venv39).
from __future__ import annotations

import json
import math
import os
import random
from collections import Counter

import pytest
from pyspark.sql.types import StringType, StructField, StructType

from distributed_curator.quality import SCORE_COLUMN_GROUPS, HeuristicConfig, compute_heuristic_scores

# Skipping this module is OPT-IN, not automatic.
#
# This file used to call pytest.importorskip(), which silently skipped all of
# its tests when the Cython extension was not built. That converted real
# breakage into a green run: the kernel/Spark column mismatch (score_document
# returning 21 values while KERNEL_COLUMN_ORDER declared 12) reached main
# behind a wall of 's' characters in the pytest output.
#
# Contributors without a build toolchain can still opt out:
#     DC_ALLOW_KERNEL_SKIP=1 pytest
try:
    from distributed_curator.quality.kernel import heuristic_kernel as kernel_mod
except ImportError as exc:  # pragma: no cover - environment-dependent
    if os.environ.get("DC_ALLOW_KERNEL_SKIP") == "1":
        pytest.skip(
            f"Cython kernel not built and DC_ALLOW_KERNEL_SKIP=1: {exc}",
            allow_module_level=True,
        )
    raise RuntimeError(
        "The Cython kernel is not importable, so these tests cannot verify it.\n"
        "Build it with:  python setup.py build_ext --inplace   (or reinstall the wheel)\n"
        "To skip anyway: DC_ALLOW_KERNEL_SKIP=1 pytest"
    ) from exc
from distributed_curator.quality.kernel_scoring import KERNEL_COLUMN_ORDER

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


class TestKernelNativeDifferential:
    """Kernel and native implementations agree column-by-column on random docs."""

    def test_differential_parity(self, spark):
        rng = random.Random(20260713)
        docs = [random_adversarial_doc(rng) for _ in range(N_DIFFERENTIAL_DOCS)]
        schema = StructType([StructField("doc_id", StringType(), False), StructField("text", StringType(), True)])
        df = spark.createDataFrame([(f"d{i}", t) for i, t in enumerate(docs)], schema)

        native = {r["doc_id"]: r.asDict() for r in compute_heuristic_scores(df, implementation="native").collect()}
        kernel = {r["doc_id"]: r.asDict() for r in compute_heuristic_scores(df, implementation="kernel").collect()}

        ngram_cols = set(SCORE_COLUMN_GROUPS["enable_ngram_repetition"])
        shared_cols = [c for c in KERNEL_COLUMN_ORDER if c not in ngram_cols]
        for i in range(len(docs)):
            for column in shared_cols:  # n-gram cols are kernel-only; oracle-tested below
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

    def test_top_ngram_none_at_word_count_boundary(self):
        """out_top_valid boundary: exactly n-1 words -> column is None, not 0.0.

        Targets the ngram_kernel.pyx <-> heuristic_kernel.pyx module boundary
        specifically (the flag array), not the n-gram math itself (already
        covered by the golden and oracle-differential tests). A bug that
        left out_top_valid uninitialized would silently emit 0.0 here.
        """
        # top_ngram_char_frac_2 needs >=2 words; top_ngram_char_frac_4 needs >=4.
        got_1w = dict(zip(KERNEL_COLUMN_ORDER, kernel_mod.score_document("solo")))
        assert got_1w["q_heur_top_ngram_char_frac_2"] is None
        assert got_1w["q_heur_top_ngram_char_frac_3"] is None
        assert got_1w["q_heur_top_ngram_char_frac_4"] is None

        got_3w = dict(zip(KERNEL_COLUMN_ORDER, kernel_mod.score_document("one two three")))
        assert got_3w["q_heur_top_ngram_char_frac_2"] is not None  # 3 words >= 2
        assert got_3w["q_heur_top_ngram_char_frac_3"] is not None  # 3 words >= 3
        assert got_3w["q_heur_top_ngram_char_frac_4"] is None  # 3 words < 4

    def test_ngram_hash_table_resize_path(self):
        """Large documents exercise the open-addressing table's capacity
        doubling (cap starts small, grows to >= 2*n_words+2); golden fixtures
        are too short to reach a resize. Cross-checked against the datatrove-
        verbatim oracle so this also confirms correctness survives the
        larger capacity, not just that it runs without crashing.
        """
        words = [f"word{i}" for i in range(2000)] + ["repeat"] * 50
        text = " ".join(words)
        got = dict(zip(KERNEL_COLUMN_ORDER, kernel_mod.score_document(text)))

        oracle_words = text.split()
        oracle_grams = [" ".join(oracle_words[i : i + 2]) for i in range(len(oracle_words) - 1)]
        top = Counter(oracle_grams).most_common(1)[0]
        expected_top_2 = len(top[0]) * top[1] / len(text)

        assert values_equal(got["q_heur_top_ngram_char_frac_2"], expected_top_2)
        assert got["q_heur_dup_ngram_char_frac_5"] > 0  # the 50x "repeat" run


# ── datatrove n-gram reference, copied verbatim (gopher_repetition_filter.py) ──
def _get_n_grams(words, n):
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


def _find_top_duplicate(x):
    from collections import Counter

    counter = Counter()
    for element in x:
        counter[element] += 1
    top_n_gram = counter.most_common(1)[0]
    return len(top_n_gram[0]) * top_n_gram[1]


def _find_all_duplicate(words, n):
    unique = set()
    repeated_chars, idx = 0, 0
    while idx < len(words) - n + 1:
        n_gram = "".join(words[idx : idx + n])
        if n_gram in unique:
            repeated_chars += len(n_gram)
            idx += n
        else:
            unique.add(n_gram)
            idx += 1
    return repeated_chars


def oracle_ngram_scores(text):
    """Expected n-gram scores per the pinned semantics (whitespace words)."""
    out = {}
    if text is None or len(text) == 0:
        for n in (2, 3, 4):
            out[f"q_heur_top_ngram_char_frac_{n}"] = None
        for n in range(5, 11):
            out[f"q_heur_dup_ngram_char_frac_{n}"] = None
        return out
    words = text.split()
    for n in (2, 3, 4):
        grams = _get_n_grams(words, n)
        out[f"q_heur_top_ngram_char_frac_{n}"] = _find_top_duplicate(grams) / len(text) if grams else None
    for n in range(5, 11):
        out[f"q_heur_dup_ngram_char_frac_{n}"] = _find_all_duplicate(words, n) / len(text)
    return out


class TestKernelNgramOracleDifferential:
    """Kernel n-gram columns match the datatrove reference implementation.

    (No Spark needed: the oracle is pure Python and the kernel is called
    directly. A 4,001-doc run of this comparison — including the
    '"ab"+"c" == "a"+"bc"' concatenation-identity trap and Counter tie
    behavior — gated PR-3b; this in-suite version runs a smaller seeded
    sample per commit.)
    """

    def test_ngram_differential_parity(self):
        rng = random.Random(20260714)
        phrase_words = ["the", "buy", "widgets", "ab", "c", "a", "bc", "東京", "…", "now."]
        docs = []
        for _ in range(400):
            parts = []
            for _ in range(rng.randint(1, 90)):
                parts.append(rng.choice(phrase_words))
                parts.append(rng.choice([" ", "  ", "\n", "\u3000"]))
            doc = "".join(parts)
            if rng.random() < 0.5:
                phrase = " ".join(rng.choice(phrase_words) for _ in range(rng.randint(2, 12)))
                doc += (" " + phrase) * rng.randint(1, 4)
            docs.append(doc)
        docs += ["", None, "one", "a b c d e f g a b c d e f g"]

        for i, t in enumerate(docs):
            got = dict(zip(KERNEL_COLUMN_ORDER, kernel_mod.score_document(t)))
            for column, expected in oracle_ngram_scores(t).items():
                assert values_equal(got[column], expected), (
                    f"doc {i}.{column}: kernel={got[column]} oracle={expected} text={t!r:.70}"
                )


class TestNativeOmitsNgramColumns:
    """Native implementation omits (with a warning) the kernel-only n-gram group."""

    def test_native_omits_and_kernel_emits(self, spark):
        df = spark.createDataFrame([("d1", "a b c d e f g a b c d e f g")], ["doc_id", "text"])
        native_cols = set(compute_heuristic_scores(df, implementation="native").columns)
        kernel_cols = set(compute_heuristic_scores(df, implementation="kernel").columns)
        ngram_cols = set(SCORE_COLUMN_GROUPS["enable_ngram_repetition"])
        assert ngram_cols.isdisjoint(native_cols)
        assert ngram_cols.issubset(kernel_cols)
        assert len(kernel_cols) == len(native_cols) + 9


class TestKernelSchemaAlignment:
    """The kernel's return arity and the Spark column list must not drift.

    Regression guard: PR-3b landed the kernel side (score_document returning
    21 values) without the Spark wiring (KERNEL_COLUMN_ORDER still 12), so
    every implementation="kernel" call failed at runtime with
    '12 columns passed, passed data had 21 columns'. These assertions fail
    loudly on the driver instead, with no Spark session needed.
    """

    def test_return_arity_matches_column_order(self):
        assert len(kernel_mod.score_document("a b c")) == len(KERNEL_COLUMN_ORDER)

    def test_null_return_arity_matches_column_order(self):
        assert len(kernel_mod.score_document(None)) == len(KERNEL_COLUMN_ORDER)

    def test_every_kernel_column_is_registered_in_a_group(self):
        from distributed_curator.quality.native_heuristics import SCORE_COLUMN_GROUPS

        registered = {c for cols in SCORE_COLUMN_GROUPS.values() for c in cols}
        missing = [c for c in KERNEL_COLUMN_ORDER if c not in registered]
        assert not missing, f"kernel columns absent from SCORE_COLUMN_GROUPS: {missing}"
