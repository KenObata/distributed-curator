# ngram_kernel.pxd - C-level interface for the n-gram repetition scans
"""cimport-able declarations for ngram_kernel.pyx.

heuristic_kernel.pyx cimports ngram_scores() from here; the call compiles to
a direct C call (no Python overhead), so the module split costs nothing at
runtime.
"""


cdef int ngram_scores(
    unsigned long long* w_hash,
    unsigned long long* w_pow,
    int* w_len,
    Py_ssize_t n_words,
    Py_ssize_t text_len,
    double* out_top,
    double* out_dup,
    unsigned char* out_top_valid,
) except -1


cdef unsigned long long combine_window(
    unsigned long long* w_hash,
    unsigned long long* w_pow,
    Py_ssize_t first_word,
    int n_words_in_window,
    bint space_separated,
) noexcept


cdef long window_char_len(
    int* w_len,
    Py_ssize_t first_word,
    int n_words_in_window,
    bint space_separated,
) noexcept
