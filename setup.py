"""
setup.py — required for Cython C extension (shingle_hash).

pyproject.toml handles all metadata and dependencies.
This file only defines the Cython Extension that setuptools
needs to compile .pyx + MurmurHash3.c into a shared object (.so).

Usage:
  pip install -e .          # editable install (dev)
  pip install .             # regular install
  python -m build           # build wheel + sdist for PyPI
"""

from setuptools import setup

# Defer Cython import — it's listed in [build-system].requires
# so it's guaranteed to be available at build time, but not at
# import time when setup.py is first parsed by pip.
try:
    import numpy as np  # to tell gcc where NumPy's .h files live
    from Cython.Build import cythonize
    from setuptools import Extension

    extensions = cythonize(
        [
            Extension(
                # Python import path: from distributed_curator.cython_minhash.shingle_hash import hash_shingles
                "distributed_curator.cython_minhash.shingle_hash",
                sources=[
                    "distributed_curator/cython_minhash/shingle_hash.pyx",
                    "distributed_curator/cython_minhash/murmurhash3.c",
                ],
                # where to find all .h header files
                include_dirs=[
                    np.get_include(),  # shingle_hash.pyx uses cimport numpy
                    "distributed_curator/cython_minhash",  # For murmurhash3.h
                ],
                # flags passed directly to gcc:
                extra_compile_args=[
                    "-O3",  # maximum optimization level
                    "-march=native",  # enables NEON SIMD instructions
                ],
            )
        ],
        compiler_directives={"language_level": "3"},  # python3.x
    )
except ImportError:
    import sys

    # Fail loudly during actual build, silently during metadata query
    if "dist_info" not in sys.argv and "egg_info" not in sys.argv:
        raise
    extensions = []

setup(ext_modules=extensions)
