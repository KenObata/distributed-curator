# shingle_hash.pyx

import numpy as np                    
cimport numpy as np_c
"""
np_c can be either python or C obj.
  np_c.ndarray     = Python object + C struct knowledge     
  np_c.uint32_t    = pure C type                            
"""
from libc.stdint cimport uint32_t, uint8_t

cdef extern from "murmurhash3.h":
    void murmurhash3_x86_32(const void *key, Py_ssize_t length, uint32_t seed, void *out)

def hash_shingles(str text,  int ngram, uint32_t seed=0) -> np.ndarray:
    """
    Translate python code to c
    unique_shingles = list({text[i:i+ngram] for i in range(len(text) - ngram + 1)})
    return [mh3 (s) for s in unique_shingles]
    """
    # Python obj
    cdef bytes text_bytes = text.encode('utf-8')

    # Python -> C cast
    cdef const uint8_t* buf = <const uint8_t*>text_bytes

    cdef int text_len = len(text_bytes)

    cdef int n = text_len - ngram + 1
    if n <= 0:
        return np.empty(0, dtype=np.uint32)

    # python obj but declared how much buffer needed in terms of C, no cast needed.
    cdef np_c.ndarray[uint32_t, ndim=1] base_hashes = np.empty(n, dtype=np.uint32)

    cdef int i
    for i in range(n):
        murmurhash3_x86_32(&buf[i], ngram, seed, &base_hashes[i])

    return base_hashes