"""Offline converter: official fastText ``.bin`` -> flat artifacts the JVM loads.

The JVM scorer (PR-7) never parses Facebook's binary format. It loads only what
this module writes:

    out_dir/
        manifest.json        config, shapes, checksums
        input_matrix.f32     row-major little-endian float32, (nwords + bucket, dim)
        output_matrix.f32    row-major little-endian float32, (nlabels, dim)
        vocab.txt            UTF-8, one word per line, input-matrix row order
        labels.txt           UTF-8, one label per line, output-matrix row order

Design:

* Python-only. The official ``fasttext`` package is the parser; we never
  reimplement it.
* Export-and-assert, never assume. Every model property the JVM depends on is
  read from the model and checked here, so an unsupported model fails at
  conversion time rather than producing silently wrong scores at scale.
* No quantized-model support, by design.
* ``loss`` must be softmax. Hierarchical softmax makes output-matrix rows
  internal Huffman-tree nodes rather than labels, which PR-7 does not implement.
  softmax is okay because there are only 2 labels in fasttext.

Verification gate: :func:`verify_export` reloads the exported artifacts and
compares them **bytewise** against ``get_input_matrix()`` / ``get_output_matrix()``.
Bytewise rather than ``==`` so NaN and -0.0 are compared honestly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from typing import Any, BinaryIO

import numpy as np

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1

INPUT_MATRIX_FILE = "input_matrix.f32"
OUTPUT_MATRIX_FILE = "output_matrix.f32"
VOCAB_FILE = "vocab.txt"
LABELS_FILE = "labels.txt"
MANIFEST_FILE = "manifest.json"

# Exported dtype. Little-endian float32 so the JVM can do
# ByteBuffer.order(LITTLE_ENDIAN).asFloatBuffer() with no header parsing.
EXPORT_DTYPE = np.dtype("<f4")

# Rows written per chunk. Keeps peak extra allocation bounded while hashing.
# treat this as a tuning knob. It doesn't affect correctness.
# Common practical range for sequential file I/O in Python is roughly 1~64 MB per chunk
# ex) oh-eli5-scale dim (~100): 16384 * 400 bytes ≈ 6.5 MB per chunk
_ROW_CHUNK = 16384

# Byte-size chunk for hashing the source .bin.
_FILE_CHUNK = 8 * 1024 * 1024

# fastText splits on these ASCII bytes and is not aware of UTF-8 whitespace.
# A vocabulary word can therefore never contain one, which is what makes a
# line-oriented vocab.txt lossless. Asserted, not assumed.
_FASTTEXT_SEPARATORS = frozenset(" \n\t\v\r\f\0")

SUPPORTED_LOSS = "softmax"
SUPPORTED_MODEL = "supervised"


class UnsupportedModelError(ValueError):
    """The model uses a feature the JVM scorer does not implement."""


class ExportVerificationError(RuntimeError):
    """Exported artifacts did not reload bytewise-identically."""


@dataclass(frozen=True)
class ModelConfig:
    """The subset of fastText args the JVM scorer needs.

    Read from the loaded model, never inferred from the filename or defaults.
    """

    dim: int
    word_ngrams: int
    bucket: int
    minn: int
    maxn: int
    loss: str
    model: str
    label_prefix: str
    n_words: int
    n_labels: int

    def to_json(self) -> dict[str, Any]:
        return {
            "dim": self.dim,
            "wordNgrams": self.word_ngrams,
            "bucket": self.bucket,
            "minn": self.minn,
            "maxn": self.maxn,
            "loss": self.loss,
            "model": self.model,
            "label": self.label_prefix,
            "n_words": self.n_words,
            "n_labels": self.n_labels,
        }


def _enum_name(value: Any) -> str:
    """fastText returns pybind11 enums for ``loss`` and ``model``.
    - pybind11 is a lightweight C++ header library that lets you write Python bindings for C++ code
    - it exposes C++ classes, functions, and types to Python without writing the raw CPython C-API boilerplate

    ``.name`` is the documented accessor; fall back to the string form so a
    pybind11 version change degrades into a readable assertion failure rather
    than an AttributeError.
    """
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    return str(value).rsplit(".", 1)[-1]


def read_model_config(model: Any) -> ModelConfig:
    """Read and validate every model property the JVM scorer depends on."""
    if model.is_quantized():
        raise UnsupportedModelError("Quantized models are out of scope. Convert the unquantized .bin.")

    args = model.f.getArgs()
    loss = _enum_name(args.loss)
    model_kind = _enum_name(args.model)

    if loss != SUPPORTED_LOSS:
        raise UnsupportedModelError(
            f"loss={loss!r} is not supported; only {SUPPORTED_LOSS!r} is. "
            "With hierarchical softmax the output matrix rows are internal "
            "Huffman-tree nodes, not labels, and PR-7 inference would need the "
            "tree as well."
        )
    if model_kind != SUPPORTED_MODEL:
        raise UnsupportedModelError(
            f"model={model_kind!r} is not supported; only {SUPPORTED_MODEL!r} is. "
            "The JVM scorer performs classification only."
        )

    return ModelConfig(
        dim=int(args.dim),
        word_ngrams=int(args.wordNgrams),
        bucket=int(args.bucket),
        minn=int(args.minn),
        maxn=int(args.maxn),
        loss=loss,
        model=model_kind,
        label_prefix=str(args.label),
        n_words=len(model.get_words(on_unicode_error="strict")),
        n_labels=len(model.get_labels(on_unicode_error="strict")),
    )


def _check_matrix(name: str, matrix: np.ndarray, rows: int, cols: int) -> None:
    if matrix.ndim != 2:
        raise UnsupportedModelError(f"{name} has ndim={matrix.ndim}, expected 2")
    if matrix.dtype != np.float32:
        raise UnsupportedModelError(f"{name} has dtype={matrix.dtype}, expected float32")
    if matrix.shape != (rows, cols):
        raise UnsupportedModelError(
            f"{name} has shape={matrix.shape}, expected {(rows, cols)}. "
            "The row-count contract the JVM loader relies on does not hold "
            "for this model."
        )


def _check_vocab_lines(name: str, entries: list[str]) -> None:
    """A line-oriented file is only lossless if no entry contains a separator."""
    for index, entry in enumerate(entries):
        bad = _FASTTEXT_SEPARATORS.intersection(entry)
        if bad:
            raise UnsupportedModelError(
                f"{name}[{index}]={entry!r} contains separator character(s) "
                f"{sorted(bad)!r}, which a line-oriented file cannot round-trip."
            )


def _write_matrix(matrix: np.ndarray, handle: BinaryIO) -> str:
    """Write row-major little-endian float32 in chunks; return the SHA-256.

    Why not not just `hashlib.sha256(open(path,'rb').read()).hexdigest()`?
    - the matrix is written in row chunks (_ROW_CHUNK rows at a time)
      specifically so peak memory stays bounded.
    """
    hasher = hashlib.sha256()
    for start in range(0, matrix.shape[0], _ROW_CHUNK):
        block = np.ascontiguousarray(matrix[start : start + _ROW_CHUNK], dtype=EXPORT_DTYPE)
        payload = block.tobytes()
        handle.write(payload)
        hasher.update(payload)  # feeds the same bytes into a running SHA-256 computation.
        hex_string_output = hasher.hexdigest()  # convert to string
    return hex_string_output


def _write_lines(entries: list[str], path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "wb") as handle:
        for entry in entries:
            payload = (entry + "\n").encode("utf-8")
            handle.write(payload)
            hasher.update(payload)
    return hasher.hexdigest()


def _hash_file(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(_FILE_CHUNK)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_model(bin_path: str) -> Any:
    # Imported lazily: fasttext is an offline-conversion dependency, not a
    # runtime dependency of the Spark job.
    import fasttext

    return fasttext.load_model(bin_path)


def convert_fasttext_model(
    bin_path: str,
    out_dir: str,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Convert ``bin_path`` into JVM-loadable artifacts under ``out_dir``.

    Returns the manifest that was written.
    """
    if os.path.exists(out_dir):
        if not overwrite:
            raise FileExistsError(f"{out_dir} already exists; pass overwrite=True to replace it")
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    logger.info("Loading %s", bin_path)
    model = _load_model(bin_path)
    config = read_model_config(model)

    words = model.get_words(on_unicode_error="strict")
    labels = model.get_labels(on_unicode_error="strict")
    _check_vocab_lines("word", words)
    _check_vocab_lines("label", labels)

    input_matrix = model.get_input_matrix()
    output_matrix = model.get_output_matrix()

    # Subword and word-ngram vectors share the bucket range, appended after the
    # in-vocabulary words. Verified against the model rather than assumed.
    expected_input_rows = config.n_words + config.bucket
    _check_matrix("input_matrix", input_matrix, expected_input_rows, config.dim)
    _check_matrix("output_matrix", output_matrix, config.n_labels, config.dim)

    input_path = os.path.join(out_dir, INPUT_MATRIX_FILE)
    output_path = os.path.join(out_dir, OUTPUT_MATRIX_FILE)

    with open(input_path, "wb") as handle:
        input_sha = _write_matrix(input_matrix, handle)
    with open(output_path, "wb") as handle:
        output_sha = _write_matrix(output_matrix, handle)

    vocab_sha = _write_lines(words, os.path.join(out_dir, VOCAB_FILE))
    labels_sha = _write_lines(labels, os.path.join(out_dir, LABELS_FILE))

    manifest = {
        "format_version": FORMAT_VERSION,
        "source": {
            "filename": os.path.basename(bin_path),
            "sha256": _hash_file(bin_path),
            "bytes": os.path.getsize(bin_path),
        },
        "config": config.to_json(),
        "matrices": {
            "input": {
                "file": INPUT_MATRIX_FILE,
                "rows": expected_input_rows,
                "cols": config.dim,
                "dtype": "float32",
                "byte_order": "little",
                "layout": "row_major",
                "sha256": input_sha,
            },
            "output": {
                "file": OUTPUT_MATRIX_FILE,
                "rows": config.n_labels,
                "cols": config.dim,
                "dtype": "float32",
                "byte_order": "little",
                "layout": "row_major",
                "sha256": output_sha,
            },
        },
        "vocab": {
            "file": VOCAB_FILE,
            "count": config.n_words,
            "encoding": "utf-8",
            "sha256": vocab_sha,
        },
        "labels": {
            "file": LABELS_FILE,
            "count": config.n_labels,
            "encoding": "utf-8",
            "sha256": labels_sha,
        },
    }

    with open(os.path.join(out_dir, MANIFEST_FILE), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    logger.info(
        "Wrote %s (input %d x %d, output %d x %d, %d words, %d labels)",
        out_dir,
        expected_input_rows,
        config.dim,
        config.n_labels,
        config.dim,
        config.n_words,
        config.n_labels,
    )
    return manifest


def load_manifest(out_dir: str) -> dict[str, Any]:
    with open(os.path.join(out_dir, MANIFEST_FILE), encoding="utf-8") as handle:
        return json.load(handle)


def load_exported_matrix(out_dir: str, which: str) -> np.ndarray:
    """Reload an exported matrix. Mirrors what the JVM loader will do."""
    manifest = load_manifest(out_dir)
    spec = manifest["matrices"][which]
    path = os.path.join(out_dir, spec["file"])
    matrix = np.fromfile(path, dtype=EXPORT_DTYPE)
    expected = spec["rows"] * spec["cols"]
    if matrix.size != expected:
        raise ExportVerificationError(f"{spec['file']} holds {matrix.size} floats, manifest declares {expected}")
    return matrix.reshape(spec["rows"], spec["cols"])


def load_exported_lines(out_dir: str, which: str) -> list[str]:
    manifest = load_manifest(out_dir)
    spec = manifest[which]
    path = os.path.join(out_dir, spec["file"])
    with open(path, encoding="utf-8") as handle:
        entries = handle.read().split("\n")
    # Trailing newline on the last entry produces one empty tail element.
    if entries and entries[-1] == "":
        entries.pop()
    if len(entries) != spec["count"]:
        raise ExportVerificationError(f"{spec['file']} holds {len(entries)} entries, manifest declares {spec['count']}")
    return entries


def verify_export(bin_path: str, out_dir: str) -> None:
    """Reload the export and compare bytewise against the source model.

    Bytewise rather than elementwise ``==`` so NaN and -0.0 are compared
    honestly. Vocabulary and label order are part of the gate: they are the
    implicit matrix-row contract PR-7 depends on.
    """
    model = _load_model(bin_path)

    checks = (
        ("input", model.get_input_matrix()),
        ("output", model.get_output_matrix()),
    )
    for which, source in checks:
        reloaded = load_exported_matrix(out_dir, which)
        if reloaded.shape != source.shape:
            raise ExportVerificationError(f"{which} matrix shape {reloaded.shape} != source {source.shape}")
        expected_bytes = np.ascontiguousarray(source, dtype=EXPORT_DTYPE).tobytes()
        if reloaded.tobytes() != expected_bytes:
            raise ExportVerificationError(f"{which} matrix is not bytewise identical to the source model")

    if load_exported_lines(out_dir, "vocab") != model.get_words(on_unicode_error="strict"):
        raise ExportVerificationError("vocab.txt does not match get_words() order")
    if load_exported_lines(out_dir, "labels") != model.get_labels(on_unicode_error="strict"):
        raise ExportVerificationError("labels.txt does not match get_labels() order")

    logger.info("Verified %s against %s", out_dir, bin_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m distributed_curator.quality.fasttext_convert",
        description="Convert a fastText .bin into JVM-loadable artifacts.",
    )
    parser.add_argument("bin_path", help="Path to the unquantized fastText .bin")
    parser.add_argument("out_dir", help="Output directory for the artifacts")
    parser.add_argument("--overwrite", action="store_true", help="Replace out_dir if it exists")
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the bytewise reload check (not recommended)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    manifest = convert_fasttext_model(args.bin_path, args.out_dir, overwrite=args.overwrite)
    if not args.no_verify:
        verify_export(args.bin_path, args.out_dir)

    print(json.dumps(manifest["config"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
