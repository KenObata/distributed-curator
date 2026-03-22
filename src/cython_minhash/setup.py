from setuptools import setup
from setuptools import Extension # compile C & Cython code into a .so file
from Cython.Build import cythonize # .pyx files → .c files before gcc compiles them
import numpy as np # to tell gcc where NumPy's .h files live

extensions = [
    Extension(
            # output file name is shingle_hash.cpython-312-x86_64-linux-gnu
            "shingle_hash",
            sources=[
                "shingle_hash.pyx", 
                "murmurhash3.c"     # By pip download mmh3 --no-binary :all:
            ],
            # where to find all .h header files
            include_dirs=[
                np.get_include(),    # shingle_hash.pyx uses cimport numpy
                 "."],              # For murmurhash3.h
            # flags passed directly to gcc:
            extra_compile_args=[
                "-O3",              # maximum optimization level     
                "-march=native"     # enables NEON SIMD instructions
            ],
    )
]

setup(
    name="shingle_hash",
    ext_modules=cythonize(module_list=extensions)
)