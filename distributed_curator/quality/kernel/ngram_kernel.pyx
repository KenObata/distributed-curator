# ngram_kernel.pyx - Gopher n-gram repetition scans (checks #13-#21)
# cython: language_level=3, boundscheck=False, wraparound=False
"""N-gram repetition scoring: the 9 kernel-only heuristic columns.

Split out of heuristic_kernel.pyx for reviewability; cimported at C level so
there is no call overhead (see ngram_kernel.pxd).

    #13-#15  q_heur_top_ngram_char_frac_{2,3,4}
    #16-#21  q_heur_dup_ngram_char_frac_{5..10}

datatrove parity (gopher_repetition_filter.py):
  top n in (2,3,4): windows are " ".join(words[i:i+n]); score =
    len(top) * count(top) / len(text), top = highest count with first-seen
    winning ties (Counter.most_common insertion order); the column is None
    when there are fewer than n words (their `if not n_grams: continue`).
  dup n in (5..10): windows are "".join(...) with skip-ahead (idx += n on a
    repeat); score = repeated chars / len(text); 0.0 when no windows exist.

Window identity is a 64-bit polynomial hash over the window's CHARACTER
sequence, composed from the per-word hashes and BASE^len powers that
heuristic_kernel's pass 1 precomputes:

    H(A||B) = H(A) * BASE^len(B) + H(B)

Composing (rather than re-hashing each window's characters) is what makes
the scan O(words * n) multiplies instead of O(chars) re-reads per n value —
the re-hash version measured ~44x the document's characters across the nine
n values. Because identity is defined over the concatenated character
sequence, "".join("ab","c") == "".join("a","bc") collides exactly as
datatrove's strings do. Stated approximation: 64-bit hash identity, not
string identity (collision odds ~1e-12/doc).
"""

from cpython.mem cimport PyMem_Free, PyMem_Malloc
from libc.string cimport memset

# Polynomial rolling-hash base (mod 2^64 via unsigned overflow). Must match
# the per-word hashes heuristic_kernel builds; both derive from this constant.
cdef unsigned long long POLY_BASE = ((<unsigned long long> 0x100) << 32) | 0x000001B3


cdef unsigned long long combine_window(
    unsigned long long* w_hash,
    unsigned long long* w_pow,
    Py_ssize_t first_word,
    int n_words_in_window,
    bint space_separated,
) noexcept:
    """Window hash from precomputed per-word hashes/powers. [checks #13-#21]

    space_separated=True folds a space between words (mirrors " ".join for
    the top n-gram path); False mirrors "".join (dup path).
    """
    cdef unsigned long long h = 0
    cdef Py_ssize_t w
    for w in range(first_word, first_word + n_words_in_window):
        if space_separated and w > first_word:
            h = h * POLY_BASE + <unsigned long long> 0x20
        h = h * w_pow[w] + w_hash[w]
    return h


cdef long window_char_len(
    int* w_len,
    Py_ssize_t first_word,
    int n_words_in_window,
    bint space_separated,
) noexcept:
    """len() of the joined window string. [checks #13-#21]"""
    cdef long total = 0
    cdef Py_ssize_t w
    for w in range(first_word, first_word + n_words_in_window):
        total += w_len[w]
    if space_separated:
        total += n_words_in_window - 1
    return total


cdef int ngram_scores(
    unsigned long long* w_hash,
    unsigned long long* w_pow,
    int* w_len,
    Py_ssize_t n_words,
    Py_ssize_t text_len,
    double* out_top,
    double* out_dup,
    unsigned char* out_top_valid,
) except -1:
    """Compute checks #13-#21 into caller-owned output arrays.

    Args:
        w_hash/w_pow/w_len: per-word polynomial hash, BASE^len, and length,
            as filled by heuristic_kernel's pass 1.
        n_words: number of words recorded.
        text_len: len(text) — the denominator for every column.
        out_top: 3 doubles, n = 2,3,4 (valid only where out_top_valid is 1).
        out_dup: 6 doubles, n = 5..10 (always valid).
        out_top_valid: 3 flags; 0 means "fewer than n words" -> column is None.

    Returns 0, or raises MemoryError (declared `except -1`).
    """
    cdef int ng
    cdef Py_ssize_t widx
    cdef unsigned long long h
    cdef long best_chars, best_count, repeated_chars
    cdef Py_ssize_t best_first
    cdef Py_ssize_t cap, mask, slot
    cdef unsigned long long* keys
    cdef long* cnt
    cdef Py_ssize_t* first
    cdef long* chlen

    memset(out_top_valid, 0, 3)
    for ng in range(6):
        out_dup[ng] = 0.0

    if text_len <= 0:
        return 0

    # C open-addressing table shared by all 9 n-loops (memset-cleared between
    # loops). Python dict/set here would box every uint64 hash: measured 4x
    # slower. Capacity = 2x max windows, rounded up to a power of two.
    cap = 2
    while cap < 2 * n_words + 2:
        cap <<= 1
    mask = cap - 1
    keys = <unsigned long long*> PyMem_Malloc(cap * sizeof(unsigned long long))
    cnt = <long*> PyMem_Malloc(cap * sizeof(long))
    first = <Py_ssize_t*> PyMem_Malloc(cap * sizeof(Py_ssize_t))
    chlen = <long*> PyMem_Malloc(cap * sizeof(long))
    if keys == NULL or cnt == NULL or first == NULL or chlen == NULL:
        PyMem_Free(keys); PyMem_Free(cnt); PyMem_Free(first); PyMem_Free(chlen)
        raise MemoryError()

    try:
        for ng in range(2, 5):  # top n-grams  [#13 #14 #15]
            if n_words < ng:
                continue  # out_top_valid stays 0 -> column is None
            memset(keys, 0, cap * sizeof(unsigned long long))
            for widx in range(n_words - ng + 1):
                h = combine_window(w_hash, w_pow, widx, ng, True)
                if h == 0:
                    h = 1  # 0 is the empty-slot sentinel
                slot = <Py_ssize_t> (h & mask)
                while keys[slot] != 0 and keys[slot] != h:
                    slot = (slot + 1) & mask
                if keys[slot] == 0:
                    keys[slot] = h
                    cnt[slot] = 1
                    first[slot] = widx
                    chlen[slot] = window_char_len(w_len, widx, ng, True)
                else:
                    cnt[slot] += 1
            best_count = 0
            best_first = -1
            best_chars = 0
            for slot in range(cap):
                if keys[slot] != 0 and (
                    cnt[slot] > best_count
                    or (cnt[slot] == best_count and first[slot] < best_first)
                ):
                    best_count = cnt[slot]
                    best_first = first[slot]
                    best_chars = chlen[slot]
            out_top[ng - 2] = <double> (best_chars * best_count) / text_len
            out_top_valid[ng - 2] = 1

        for ng in range(5, 11):  # dup n-grams, skip-ahead  [#16-#21]
            memset(keys, 0, cap * sizeof(unsigned long long))
            repeated_chars = 0
            widx = 0
            while widx < n_words - ng + 1:
                h = combine_window(w_hash, w_pow, widx, ng, False)
                if h == 0:
                    h = 1
                slot = <Py_ssize_t> (h & mask)
                while keys[slot] != 0 and keys[slot] != h:
                    slot = (slot + 1) & mask
                if keys[slot] != 0:  # seen before -> duplicate window
                    repeated_chars += window_char_len(w_len, widx, ng, False)
                    widx += ng
                else:
                    keys[slot] = h
                    widx += 1
            out_dup[ng - 5] = <double> repeated_chars / text_len
    finally:
        PyMem_Free(keys); PyMem_Free(cnt); PyMem_Free(first); PyMem_Free(chlen)

    return 0
