# heuristic_kernel.pyx - single-pass-per-concern Cython kernel for the 12 heuristic scores
# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython kernel computing all 12 heuristic quality scores per document.

Fast path for the heuristic layer (see native_heuristics.py for the reference
Spark-SQL implementation this kernel must match — enforced by golden and
differential tests). Semantics are pinned to the same spec:

- words: whitespace runs (str.split() semantics), no empty tokens
- non-symbol word: >=1 char outside the vendored datatrove punctuation set
- alpha word: >=1 Unicode letter (Py_UNICODE_ISALPHA == str.isalpha per char,
  same categories as Java \\p{L} used by the native rlike)
- quality lines: \\R-style terminators (\\r\\n, \\n, \\v, \\f, \\r, \\x85,
  U+2028, U+2029) with ONE trailing terminator dropped; empty text -> 0 lines
  (mirrors native splitlines() emulation, not Python splitlines(), which also
  breaks on \\x1c-\\x1e)
- repetition lines: re.split("\\n+", text) on UNSTRIPPED text
- paragraphs: re.split("\\n{2,}", text.strip())
- duplicate counting: 2nd..Nth occurrences each count once and contribute
  their length; char-fraction denominators are len(text)
- ratios: None on zero denominator (try_divide semantics)

Design: the hot per-character work (word stats, '#'/'...' counting) happens in
one typed codepoint loop without materializing word strings — a word string is
sliced out only when short enough to be a stopword candidate. Line- and
paragraph-level work runs as separate O(len) scans over the same string; the
handful of line slices per document use Python str methods (startswith/endswith
with tuples) for exact datatrove-style semantics.
"""

# ══════════════════════════════════════════════════════════════════════════════
# CHECK NUMBERING (order == KERNEL_COLUMN_ORDER == score_document return tuple)
#
#   #1  q_heur_word_count           <- n_nonsym_words                  (pass 1)
#   #2  q_heur_mean_word_len        <- nonsym_len_sum / n_nonsym_words (pass 1)
#   #3  q_heur_hash_word_ratio      <- hash_count / n_words            (pass 1)
#   #4  q_heur_ellipsis_word_ratio  <- ellipsis_count(dot_run) / n_words (pass 1)
#   #5  q_heur_bullet_line_frac     <- n_bullet / n_lines              (pass 2, _classify_quality_line)
#   #6  q_heur_ellipsis_line_frac   <- n_ellipsis_end / n_lines        (pass 2, _classify_quality_line)
#   #7  q_heur_alpha_word_frac      <- n_alpha_words / n_words         (pass 1)
#   #8  q_heur_stopword_count       <- len(seen_stops)                 (pass 1)
#   #9  q_heur_dup_line_frac        <- rep_line_dups / rep_line_count  (pass 3, _dup_scan_newline_split)
#   #10 q_heur_dup_line_char_frac   <- rep_line_dup_chars / len(text)  (pass 3, _dup_scan_newline_split)
#   #11 q_heur_dup_para_frac        <- para_dups / para_count          (pass 4, _dup_scan_newline_split)
#   #12 q_heur_dup_para_char_frac   <- para_dup_chars / len(text)      (pass 4, _dup_scan_newline_split)
#
# NOTE: checks are fused into shared scans on purpose (that fusion is the
# speedup); one-check-per-unit structure lives in native_heuristics.py, the
# reference implementation. Shared denominators: n_words feeds #3/#4/#7,
# n_lines feeds #5/#6, len(text) feeds #10/#12.
# ══════════════════════════════════════════════════════════════════════════════

from cpython.unicode cimport Py_UNICODE_ISALPHA, Py_UNICODE_ISSPACE

from distributed_curator.quality.config import GOPHER_STOP_WORDS, PUNCTUATION_CHARS

# ── module-level lookup table: punctuation membership by codepoint ────────────
# Max codepoint in the vendored set is 0x1DA88; a flat byte table (~121 KB,
# built once at import) turns set membership into an array index.
cdef int _PUNCT_TABLE_SIZE = 0
for _ch in PUNCTUATION_CHARS:
    if ord(_ch) + 1 > _PUNCT_TABLE_SIZE:
        _PUNCT_TABLE_SIZE = ord(_ch) + 1

_punct_bytes = bytearray(_PUNCT_TABLE_SIZE)
for _ch in PUNCTUATION_CHARS:
    _punct_bytes[ord(_ch)] = 1

cdef const unsigned char[::1] _PUNCT_TABLE = bytes(_punct_bytes)

DEF ELLIPSIS_CP = 0x2026  # '…'


cdef inline bint _is_punct(Py_UCS4 ch) nogil:
    if <int> ch >= _PUNCT_TABLE_SIZE:
        return False
    return _PUNCT_TABLE[<int> ch] == 1


cdef inline bint _is_quality_line_terminator(Py_UCS4 ch) nogil:
    # Java regex \R single-char terminators (\r\n handled by caller).
    return (
        ch == u'\n' or ch == u'\r' or ch == u'\x0b' or ch == u'\x0c'
        or ch == u'\x85' or ch == u'\u2028' or ch == u'\u2029'
    )



cdef inline bint _is_space(Py_UCS4 ch):
    # Java (?U)\\s == Unicode IsWhite_Space. Python's isspace() additionally
    # includes the C0 separators \\x1c-\\x1f, which Java excludes; the native
    # implementation is the parity target, so exclude them here too.
    if 0x1C <= <int> ch <= 0x1F:
        return False
    return Py_UNICODE_ISSPACE(ch)


def score_document(
    str text,
    stop_words: tuple = GOPHER_STOP_WORDS,
    bullet_prefixes: tuple = ("\u2022", "-"),
    ellipsis_suffixes: tuple = ("...", "\u2026"),
):
    """Compute the 12 heuristic scores for one document.

    Returns a 12-tuple in SCORE_COLUMN ORDER (see heuristics.py):
    (word_count, mean_word_len, hash_word_ratio, ellipsis_word_ratio,
     bullet_line_frac, ellipsis_line_frac, alpha_word_frac, stopword_count,
     dup_line_frac, dup_line_char_frac, dup_para_frac, dup_para_char_frac)

    None text -> all None. Ratios are None when their denominator is zero.
    """
    if text is None:
        return (None,) * 12

    cdef Py_ssize_t n = len(text)
    cdef Py_ssize_t i
    cdef Py_UCS4 ch

    # word-scan state
    cdef Py_ssize_t word_start = -1
    cdef Py_ssize_t word_len = 0
    cdef bint word_has_alpha = False
    cdef bint word_has_nonpunct = False
    cdef long n_words = 0            # denominator for #3 #4 #7
    cdef long n_nonsym_words = 0     # -> #1; denominator for #2
    cdef long nonsym_len_sum = 0     # -> #2
    cdef long n_alpha_words = 0      # -> #7
    cdef long hash_count = 0         # -> #3
    cdef long dot_run = 0            # -> #4 ('...' run tracker)
    cdef long ellipsis_count = 0     # -> #4
    cdef Py_ssize_t max_stop_len = 0 # -> #8 (slice-avoidance bound)

    stop_set = frozenset(stop_words)
    for w in stop_set:
        if len(w) > max_stop_len:
            max_stop_len = len(w)
    seen_stops = set()  # -> #8 (distinct stopwords found)

    # ── pass 1: words + symbol counting, one codepoint loop [#1-#4, #7, #8] ──
    for i in range(n):
        ch = text[i]

        if ch == u'#':
            hash_count += 1
        if ch == u'.':
            dot_run += 1
        else:
            ellipsis_count += dot_run // 3  # non-overlapping '...' count
            dot_run = 0
        if ch == ELLIPSIS_CP:
            ellipsis_count += 1

        if _is_space(ch):
            if word_start >= 0:  # close word
                n_words += 1
                if word_has_nonpunct:
                    n_nonsym_words += 1
                    nonsym_len_sum += word_len
                if word_has_alpha:
                    n_alpha_words += 1
                if word_len <= max_stop_len:
                    w = text[word_start:word_start + word_len]
                    if w in stop_set:
                        seen_stops.add(w)
                word_start = -1
                word_len = 0
                word_has_alpha = False
                word_has_nonpunct = False
        else:
            if word_start < 0:
                word_start = i
            word_len += 1
            if not word_has_alpha and Py_UNICODE_ISALPHA(ch):
                word_has_alpha = True
            if not word_has_nonpunct and not _is_punct(ch):
                word_has_nonpunct = True

    ellipsis_count += dot_run // 3  # flush trailing dot run
    if word_start >= 0:  # flush trailing word
        n_words += 1
        if word_has_nonpunct:
            n_nonsym_words += 1
            nonsym_len_sum += word_len
        if word_has_alpha:
            n_alpha_words += 1
        if word_len <= max_stop_len:
            w = text[word_start:word_start + word_len]
            if w in stop_set:
                seen_stops.add(w)

    # ── pass 2: quality lines (\R split, one trailing terminator dropped) [#5 #6]
    cdef long n_lines = 0        # denominator for #5 #6
    cdef long n_bullet = 0       # -> #5
    cdef long n_ellipsis_end = 0 # -> #6
    cdef Py_ssize_t line_start = 0
    cdef Py_ssize_t scan_end = n

    if n > 0:
        # drop exactly one trailing terminator (\r\n counts as one)
        if n >= 2 and text[n - 2] == u'\r' and text[n - 1] == u'\n':
            scan_end = n - 2
        elif _is_quality_line_terminator(text[n - 1]):
            scan_end = n - 1

        i = 0
        while i <= scan_end:
            if i == scan_end:
                _classify_quality_line(
                    text, line_start, i, bullet_prefixes, ellipsis_suffixes,
                    &n_lines, &n_bullet, &n_ellipsis_end,
                )
                break
            ch = text[i]
            if _is_quality_line_terminator(ch):
                _classify_quality_line(
                    text, line_start, i, bullet_prefixes, ellipsis_suffixes,
                    &n_lines, &n_bullet, &n_ellipsis_end,
                )
                if ch == u'\r' and i + 1 < scan_end and text[i + 1] == u'\n':
                    i += 1  # \r\n is one terminator
                line_start = i + 1
            i += 1

    # ── pass 3: repetition lines (\n+ split, unstripped) [#9 #10] ─────────────
    cdef long rep_line_count = 0     # denominator for #9
    cdef long rep_line_dups = 0      # -> #9
    cdef long rep_line_dup_chars = 0 # -> #10
    _dup_scan_newline_split(text, 0, n, 1, &rep_line_count, &rep_line_dups, &rep_line_dup_chars)

    # ── pass 4: paragraphs (\n{2,} split on stripped text) [#11 #12] ──────────
    cdef Py_ssize_t s_start = 0
    cdef Py_ssize_t s_end = n
    while s_start < n and _is_space(text[s_start]):
        s_start += 1
    while s_end > s_start and _is_space(text[s_end - 1]):
        s_end -= 1

    cdef long para_count = 0     # denominator for #11
    cdef long para_dups = 0      # -> #11
    cdef long para_dup_chars = 0 # -> #12
    _dup_scan_newline_split(text, s_start, s_end, 2, &para_count, &para_dups, &para_dup_chars)

    # ── assemble (try_divide semantics: None on zero denominator) ────────────
    word_count = n_nonsym_words  # #1
    mean_word_len = (<double> nonsym_len_sum / n_nonsym_words) if n_nonsym_words > 0 else None  # #2
    hash_ratio = (<double> hash_count / n_words) if n_words > 0 else None  # #3
    ell_ratio = (<double> ellipsis_count / n_words) if n_words > 0 else None  # #4
    bullet_frac = (<double> n_bullet / n_lines) if n_lines > 0 else None  # #5
    ell_line_frac = (<double> n_ellipsis_end / n_lines) if n_lines > 0 else None  # #6
    alpha_frac = (<double> n_alpha_words / n_words) if n_words > 0 else None  # #7
    dup_line_frac = (<double> rep_line_dups / rep_line_count) if rep_line_count > 0 else None  # #9
    dup_line_chars = (<double> rep_line_dup_chars / n) if n > 0 else None  # #10
    dup_para_frac = (<double> para_dups / para_count) if para_count > 0 else None  # #11
    dup_para_chars = (<double> para_dup_chars / n) if n > 0 else None  # #12

    return (
        word_count, mean_word_len, hash_ratio, ell_ratio,
        bullet_frac, ell_line_frac, alpha_frac, len(seen_stops),  # len(seen_stops) is #8
        dup_line_frac, dup_line_chars, dup_para_frac, dup_para_chars,
    )


cdef void _classify_quality_line(
    str text, Py_ssize_t start, Py_ssize_t end,
    tuple bullet_prefixes, tuple ellipsis_suffixes,
    long* n_lines, long* n_bullet, long* n_ellipsis_end,
):
    """Count one quality line; bullet/ellipsis checks on l/r-stripped slice. [checks #5 #6]"""
    cdef Py_ssize_t a = start
    cdef Py_ssize_t b = end
    n_lines[0] += 1
    while a < b and _is_space(text[a]):
        a += 1
    while b > a and _is_space(text[b - 1]):
        b -= 1
    if a >= b:
        return
    line = text[a:b]
    if line.startswith(bullet_prefixes):
        n_bullet[0] += 1
    if line.endswith(ellipsis_suffixes):
        n_ellipsis_end[0] += 1


cdef void _dup_scan_newline_split(
    str text, Py_ssize_t start, Py_ssize_t end, int min_newlines,
    long* out_count, long* out_dups, long* out_dup_chars,
):
    """re.split("\\n{min_newlines,}", text[start:end]) with duplicate counting. [checks #9-#12]

    Matches re.split edge semantics: leading/trailing separator runs produce
    empty elements; an empty input region yields one empty element. Duplicate
    counting matches datatrove find_duplicates: occurrences after the first
    each count once and contribute their length.
    """
    seen = set()
    cdef Py_ssize_t i = start
    cdef Py_ssize_t elem_start = start
    cdef Py_ssize_t run
    cdef Py_ssize_t run_start

    while True:
        if i >= end:
            _dup_add(text, elem_start, end, seen, out_count, out_dups, out_dup_chars)
            break
        if text[i] == u'\n':
            run_start = i
            run = 0
            while i < end and text[i] == u'\n':
                run += 1
                i += 1
            if run >= min_newlines:  # split point: close current element
                _dup_add(text, elem_start, run_start, seen, out_count, out_dups, out_dup_chars)
                elem_start = i
            # run below threshold stays inside the element (e.g. single \n
            # inside a paragraph when min_newlines=2)
        else:
            i += 1


cdef inline void _dup_add(
    str text, Py_ssize_t start, Py_ssize_t end, set seen,
    long* out_count, long* out_dups, long* out_dup_chars,
):
    elem = text[start:end]
    out_count[0] += 1
    if elem in seen:
        out_dups[0] += 1
        out_dup_chars[0] += end - start
    else:
        seen.add(elem)
