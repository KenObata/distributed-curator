# native_heuristics.py - reference implementation: heuristic scoring as native Spark SQL expressions
# This module is the parity oracle for the Cython kernel (kernel_scoring.py);
# public entry point is heuristics.compute_heuristic_scores.
"""Heuristic quality-scoring layer.

Appends ``q_heur_*`` score columns to an input DataFrame. Emits raw
measurements only — never pass/fail. Filtering is a separate, user-configured
step over the score columns (see reference thresholds per column below).

Semantics follow datatrove's Gopher reimplementation
(gopher_quality_filter.py / gopher_repetition_filter.py) so that scores are
directly comparable in the Phase 5 datatrove benchmark, with one documented
deviation:

- DEVIATION (word splitting): datatrove uses a language-aware word tokenizer
  (``split_into_words``); this layer splits on whitespace runs, which is what
  a native Spark expression can do without a Python UDF. Word-based scores
  can therefore differ slightly from datatrove on punctuation-adjacent tokens
  (e.g. "word." counts as one word here, "word" + "." as two there).

Null handling: a NULL text value yields NULL in every score column. An empty
or whitespace-only text yields 0 word counts and NULL ratios (null-safe
division via try_divide). No layer ever drops rows.
"""

from __future__ import annotations

import logging

import pyspark.sql.functions as F
from pyspark.sql import Column, DataFrame

from .config import PUNCTUATION_STRING, HeuristicConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Internal scratch columns, dropped before returning. Underscore-prefixed to
# minimize collision risk; compute_heuristic_scores() raises on collision.
_TMP_WORDS = "_q_tmp_words"
_TMP_NONSYM_WORDS = "_q_tmp_nonsym_words"
_TMP_QUALITY_LINES = "_q_tmp_quality_lines"
_TMP_LINE_STATS = "_q_tmp_line_stats"
_TMP_PARA_STATS = "_q_tmp_para_stats"

# Column groups, keyed by the HeuristicConfig enable flag that controls them.
SCORE_COLUMN_GROUPS: dict[str, tuple[str, ...]] = {
    "enable_word_count": ("q_heur_word_count",),
    "enable_mean_word_length": ("q_heur_mean_word_len",),
    "enable_symbol_ratios": ("q_heur_hash_word_ratio", "q_heur_ellipsis_word_ratio"),
    "enable_line_format_fractions": ("q_heur_bullet_line_frac", "q_heur_ellipsis_line_frac"),
    "enable_alpha_word_fraction": ("q_heur_alpha_word_frac",),
    "enable_stop_words": ("q_heur_stopword_count",),
    "enable_dup_lines_paragraphs": (
        "q_heur_dup_line_frac",
        "q_heur_dup_line_char_frac",
        "q_heur_dup_para_frac",
        "q_heur_dup_para_char_frac",
    ),
    # Gopher n-gram repetition (Rae et al. Table A1; datatrove parity).
    # KERNEL-ONLY: the duplicate-n-gram skip-ahead is data-dependent control
    # flow, not expressible as SQL expressions. The native implementation
    # omits these columns and warns.
    "enable_ngram_repetition": (
        "q_heur_top_ngram_char_frac_2",
        "q_heur_top_ngram_char_frac_3",
        "q_heur_top_ngram_char_frac_4",
        "q_heur_dup_ngram_char_frac_5",
        "q_heur_dup_ngram_char_frac_6",
        "q_heur_dup_ngram_char_frac_7",
        "q_heur_dup_ngram_char_frac_8",
        "q_heur_dup_ngram_char_frac_9",
        "q_heur_dup_ngram_char_frac_10",
    ),
}

# Column groups the native (SQL-expression) implementation cannot compute.
NATIVE_UNSUPPORTED_GROUPS: frozenset[str] = frozenset({"enable_ngram_repetition"})


def _whitespace_words(text: Column) -> Column:
    """Split text into words on whitespace runs, dropping empty tokens.

    Matches Python's ``str.split()`` semantics (used as the documented
    whitespace-split deviation from datatrove's tokenizer).

    (?U) makes Java's \\s Unicode-aware (IsWhite_Space property) — without
    it, \\s is ASCII-only and e.g. U+3000 IDEOGRAPHIC SPACE does not split
    words, silently corrupting word-based scores on CJK text. Found by the
    kernel differential test. Known residual deviation from str.split():
    Java (?U)\\s excludes \\x1c-\\x1f (Python isspace() includes them).
    """
    return F.filter(F.split(text, r"(?U)\s+"), lambda w: w != F.lit(""))


def _size(arr: Column) -> Column:
    """NULL-safe array size.

    Spark's size() returns -1 for NULL input (spark.sql.legacy.sizeOfNull
    defaults to true). -1 silently corrupts count columns and makes
    try_divide(-1, -1) yield 1.0 for NULL text. This helper returns NULL for
    NULL arrays so NULL text propagates to NULL scores, independent of
    session config.
    """
    return F.when(arr.isNull(), F.lit(None).cast("int")).otherwise(F.size(arr))


def _non_symbol_words(words: Column) -> Column:
    """Words containing at least one character outside the punctuation set.

    translate() strips every punctuation character; anything left means the
    word is a non-symbol word. Direct codepoint mapping — measured 2.1x faster
    than an equivalent rlike character class.
    """
    return F.filter(words, lambda w: F.length(F.translate(w, PUNCTUATION_STRING, "")) > 0)


def _occurrence_count(text: Column, needle: str) -> Column:
    """Non-overlapping, left-to-right occurrence count of a literal substring.

    Implemented as a length delta after replace(), which removes occurrences
    with the same non-overlapping left-to-right semantics as Python's
    ``str.count``.
    """
    return (F.length(text) - F.length(F.replace(text, F.lit(needle), F.lit("")))) / len(needle)


def _duplicate_stats(arr: Column) -> Column:
    """Duplicate-element and duplicate-character counts for a string array.

    Returns struct(dups INT, chars INT) where an element's 2nd..Nth
    occurrences each count as one duplicate and contribute their length to
    ``chars`` — identical to datatrove's ``find_duplicates``.

    Implementation: sort the array, then a single aggregate() pass comparing
    each element to its predecessor (O(n log n), no quadratic nested filter).
    The initial ``prev`` is NULL and compared null-safely, so a genuine empty
    string as the first sorted element is not miscounted as a duplicate.
    """
    return F.aggregate(
        F.array_sort(arr),
        F.struct(
            F.lit(None).cast("string").alias("prev"),
            F.lit(0).alias("dups"),
            F.lit(0).alias("chars"),
        ),
        lambda acc, x: F.struct(
            x.alias("prev"),
            (acc["dups"] + F.when(x.eqNullSafe(acc["prev"]), 1).otherwise(0)).alias("dups"),
            (acc["chars"] + F.when(x.eqNullSafe(acc["prev"]), F.length(x)).otherwise(0)).alias("chars"),
        ),
        lambda acc: F.struct(acc["dups"].alias("dups"), acc["chars"].alias("chars")),
    )


def _python_strip(text: Column) -> Column:
    """Equivalent of Python str.strip() (Spark's trim() only strips spaces)."""
    return F.regexp_replace(text, r"(?U)^\s+|\s+$", "")


def _lstrip(line: Column) -> Column:
    return F.regexp_replace(line, r"(?U)^\s+", "")


def _rstrip(line: Column) -> Column:
    return F.regexp_replace(line, r"(?U)\s+$", "")


def compute_native_heuristic_scores(
    df: DataFrame,
    text_column: str = "text",
    config: HeuristicConfig | None = None,
) -> DataFrame:
    """Append heuristic quality-score columns to ``df``.

    Pure transformation: returns input rows + score columns, preserving all
    existing columns and never dropping or reordering rows. No actions are
    triggered, no shuffles introduced (narrow, per-row expressions only).

    Score columns and their published reference thresholds (for the separate
    filter step — this function applies none of them):

    - q_heur_word_count          INT     Gopher keeps 50..100_000
    - q_heur_mean_word_len       DOUBLE  Gopher keeps 3..10
    - q_heur_hash_word_ratio     DOUBLE  Gopher keeps <= 0.1
    - q_heur_ellipsis_word_ratio DOUBLE  Gopher keeps <= 0.1
    - q_heur_bullet_line_frac    DOUBLE  Gopher keeps <= 0.9
    - q_heur_ellipsis_line_frac  DOUBLE  Gopher keeps <= 0.3
    - q_heur_alpha_word_frac     DOUBLE  Gopher keeps >= 0.8
    - q_heur_stopword_count      INT     Gopher keeps >= 2 (distinct, case-sensitive)
    - q_heur_dup_line_frac       DOUBLE  Gopher keeps <= 0.30
    - q_heur_dup_line_char_frac  DOUBLE  Gopher keeps <= 0.20
    - q_heur_dup_para_frac       DOUBLE  Gopher keeps <= 0.30
    - q_heur_dup_para_char_frac  DOUBLE  Gopher keeps <= 0.20

    Args:
        df: input DataFrame.
        text_column: name of the document text column.
        config: per-rule enable flags and parameters. Defaults to all enabled.

    Returns:
        DataFrame with the enabled score columns appended.

    Raises:
        ValueError: if ``text_column`` is missing, or an output/scratch column
            name already exists on ``df`` (refuses to silently overwrite).
    """
    config = config or HeuristicConfig()

    if text_column not in df.columns:
        raise ValueError(f"text_column '{text_column}' not found in DataFrame columns: {df.columns}")

    if getattr(config, "enable_ngram_repetition", False):
        logger.warning(
            "n-gram repetition columns are not expressible as native SQL expressions "
            "and will be OMITTED by implementation='native'; use implementation='kernel' "
            "to compute them."
        )
    enabled_columns = [
        col
        for flag, cols in SCORE_COLUMN_GROUPS.items()
        if getattr(config, flag) and flag not in NATIVE_UNSUPPORTED_GROUPS
        for col in cols
    ]
    scratch_columns = [_TMP_WORDS, _TMP_NONSYM_WORDS, _TMP_QUALITY_LINES, _TMP_LINE_STATS, _TMP_PARA_STATS]
    collisions = [c for c in enabled_columns + scratch_columns if c in df.columns]
    if collisions:
        raise ValueError(f"Output/scratch columns already exist on input DataFrame: {collisions}")

    logger.info(f"Computing heuristic scores: {len(enabled_columns)} columns on text_column='{text_column}'")

    text = F.col(text_column)
    needs_words = (
        config.enable_word_count
        or config.enable_mean_word_length
        or config.enable_symbol_ratios
        or config.enable_alpha_word_fraction
        or config.enable_stop_words
    )

    # --- scratch arrays, computed once and reused by multiple score columns ---
    if needs_words:
        df = df.withColumn(_TMP_WORDS, _whitespace_words(text))
    if config.enable_word_count or config.enable_mean_word_length:
        df = df.withColumn(_TMP_NONSYM_WORDS, _non_symbol_words(F.col(_TMP_WORDS)))
    if config.enable_line_format_fractions:
        # datatrove uses str.splitlines() here. Emulation: drop ONE trailing
        # line terminator (anchored \z, NOT $: Java's $ also matches before
        # a final line terminator, so \R$ + replaceAll strips up to TWO
        # trailing terminators - found by the kernel differential test), then split on \R (Java regex: \n, \r\n, \r, \v, \f,
        # \x85, \u2028, \u2029 — splitlines() minus \x1c-\x1e, a negligible
        # difference for web text). splitlines("") == [], so empty text maps
        # to an empty array rather than [""].
        df = df.withColumn(
            _TMP_QUALITY_LINES,
            F.when(F.length(text) == 0, F.array().cast("array<string>")).otherwise(
                F.split(F.regexp_replace(text, r"\R\z", ""), r"\R")
            ),
        )

    words = F.col(_TMP_WORDS)
    nonsym_words = F.col(_TMP_NONSYM_WORDS)
    quality_lines = F.col(_TMP_QUALITY_LINES)
    n_words = _size(words)

    # --- word-based scores (Gopher quality rules) ---
    if config.enable_word_count:
        df = df.withColumn("q_heur_word_count", _size(nonsym_words))

    if config.enable_mean_word_length:
        total_len = F.aggregate(nonsym_words, F.lit(0), lambda acc, w: acc + F.length(w))
        df = df.withColumn(
            "q_heur_mean_word_len",
            F.try_divide(total_len.cast("double"), _size(nonsym_words)),
        )

    if config.enable_symbol_ratios:
        df = df.withColumn("q_heur_hash_word_ratio", F.try_divide(_occurrence_count(text, "#"), n_words))
        df = df.withColumn(
            "q_heur_ellipsis_word_ratio",
            F.try_divide(_occurrence_count(text, "...") + _occurrence_count(text, "…"), n_words),
        )

    if config.enable_line_format_fractions:
        is_bullet = lambda line: F.array_contains(  # noqa: E731 - tiny predicate, named for readability
            F.array(*[_lstrip(line).startswith(F.lit(p)) for p in config.bullet_prefixes]), True
        )
        is_ellipsis_end = lambda line: F.array_contains(  # noqa: E731
            F.array(*[_rstrip(line).endswith(F.lit(s)) for s in config.ellipsis_suffixes]), True
        )
        df = df.withColumn(
            "q_heur_bullet_line_frac",
            F.try_divide(_size(F.filter(quality_lines, is_bullet)), _size(quality_lines)),
        )
        df = df.withColumn(
            "q_heur_ellipsis_line_frac",
            F.try_divide(_size(F.filter(quality_lines, is_ellipsis_end)), _size(quality_lines)),
        )

    if config.enable_alpha_word_fraction:
        # \p{L} = Unicode letter, matching Python str.isalpha() per character.
        alpha_words = F.filter(words, lambda w: w.rlike(r"\p{L}"))
        df = df.withColumn("q_heur_alpha_word_frac", F.try_divide(_size(alpha_words), n_words))

    if config.enable_stop_words:
        # Distinct, case-sensitive membership — array_intersect deduplicates,
        # matching datatrove's set(words) & stop_words.
        stop_arr = F.array(*[F.lit(w) for w in config.stop_words])
        df = df.withColumn("q_heur_stopword_count", _size(F.array_intersect(words, stop_arr)))

    # --- repetition scores (Gopher repetition rules, line/paragraph subset) ---
    if config.enable_dup_lines_paragraphs:
        # datatrove: lines = re.split("\n+", text)  (text NOT stripped)
        #            paragraphs = re.split("\n{2,}", text.strip())
        # char-fraction denominators are len(text), not the sum of element lengths.
        rep_lines = F.split(text, r"\n+")
        rep_paras = F.split(_python_strip(text), r"\n{2,}")
        text_len = F.length(text)

        # Each stats struct lands in its own scratch column so the O(n log n)
        # sort+aggregate is guaranteed to evaluate once per row — deriving two
        # fractions from the same expression in separate projections would
        # otherwise rely on Catalyst subexpression elimination kicking in.
        df = df.withColumn(_TMP_LINE_STATS, _duplicate_stats(rep_lines))
        df = df.withColumn(_TMP_PARA_STATS, _duplicate_stats(rep_paras))
        line_stats = F.col(_TMP_LINE_STATS)
        para_stats = F.col(_TMP_PARA_STATS)

        df = df.withColumn("q_heur_dup_line_frac", F.try_divide(line_stats["dups"], _size(rep_lines)))
        df = df.withColumn("q_heur_dup_line_char_frac", F.try_divide(line_stats["chars"], text_len))
        df = df.withColumn("q_heur_dup_para_frac", F.try_divide(para_stats["dups"], _size(rep_paras)))
        df = df.withColumn("q_heur_dup_para_char_frac", F.try_divide(para_stats["chars"], text_len))

    return df.drop(*scratch_columns)
