"""Tests for the offline fastText -> JVM artifact converter.

The tiny supervised fixture model committed in PR-5 is the input. The 2.39 GB
production models are covered by the documented manual verification run in
docs/fasttext.md, not by CI.

The unsupported-model paths are exercised with stubs rather than by training
throwaway hs/skipgram models: the code under test only reads properties, so a
stub is a faithful and much cheaper oracle.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pytest

from distributed_curator.quality import fasttext_convert as ftc

pytestmark = pytest.mark.usefixtures()

# =============================================================
# Set up Fixtures and mock objects
# =============================================================
FIXTURE_MODEL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "fixtures",
    "fasttext",
    "tiny_supervised.bin",
)


class _MockArgs:
    def __init__(self, **overrides: Any) -> None:
        self.dim = 4
        self.wordNgrams = 2
        self.bucket = 8
        self.minn = 0
        self.maxn = 0
        self.loss = "softmax"
        self.model = "supervised"
        self.label = "__label__"
        for key, value in overrides.items():
            setattr(self, key, value)


class _MockInnerFunction:
    def __init__(self, args: _MockArgs) -> None:
        self._args = args

    def getArgs(self) -> _MockArgs:
        return self._args


class _MockModel:
    """Minimal stand-in for fasttext.FastText._FastText."""

    def __init__(
        self,
        *,
        args: _MockArgs | None = None,
        quantized: bool = False,
        words: list[str] | None = None,
        labels: list[str] | None = None,
    ) -> None:
        self._args = args or _MockArgs()
        self.f = _MockInnerFunction(self._args)
        self._quantized = quantized
        self._words = words if words is not None else ["a", "b", "</s>"]
        self._labels = labels if labels is not None else ["__label__x", "__label__y"]

    def is_quantized(self) -> bool:
        return self._quantized

    def get_words(self, on_unicode_error: str = "strict") -> list[str]:
        return list(self._words)

    def get_labels(self, on_unicode_error: str = "strict") -> list[str]:
        return list(self._labels)

    def get_input_matrix(self) -> np.ndarray:
        rows = len(self._words) + self._args.bucket
        return np.arange(rows * self._args.dim, dtype=np.float32).reshape(rows, self._args.dim)

    def get_output_matrix(self) -> np.ndarray:
        rows = len(self._labels)
        return np.arange(rows * self._args.dim, dtype=np.float32).reshape(rows, self._args.dim)


@pytest.fixture
def generate_mock_model(monkeypatch: pytest.MonkeyPatch):
    """Install a stub loader and hand the test the model it will return."""

    def _install(model: _MockModel) -> _MockModel:
        monkeypatch.setattr(ftc, "_load_model", lambda _path: model)
        return model

    return _install


@pytest.fixture
def fake_bin(tmp_path):
    path = tmp_path / "stub.bin"
    path.write_bytes(b"not a real model, only hashed")
    return str(path)


# =============================================================
# Real test starts
# =============================================================


class TestConfigValidation:
    def test_rejects_quantized(self):
        with pytest.raises(ftc.UnsupportedModelError, match="Quantized"):
            ftc.read_model_config(_MockModel(quantized=True))

    def test_rejects_hierarchical_softmax(self):
        model = _MockModel(args=_MockArgs(loss="hs"))
        with pytest.raises(ftc.UnsupportedModelError, match="loss='hs'"):
            ftc.read_model_config(model)

    def test_rejects_unsupervised(self):
        model = _MockModel(args=_MockArgs(model="skipgram"))
        with pytest.raises(ftc.UnsupportedModelError, match="model='skipgram'"):
            ftc.read_model_config(model)

    def test_reads_every_field_the_jvm_needs(self):
        config = ftc.read_model_config(_MockModel())
        assert config.to_json() == {
            "dim": 4,
            "wordNgrams": 2,
            "bucket": 8,
            "minn": 0,
            "maxn": 0,
            "loss": "softmax",
            "model": "supervised",
            "label": "__label__",
            "n_words": 3,
            "n_labels": 2,
        }

    def test_rejects_word_containing_separator(self, generate_mock_model, fake_bin, tmp_path):
        generate_mock_model(_MockModel(words=["ok", "bad\tword", "</s>"]))
        with pytest.raises(ftc.UnsupportedModelError, match="separator"):
            ftc.convert_fasttext_model(fake_bin, str(tmp_path / "out"))


class TestExportShape:
    def test_manifest_declares_input_rows_as_words_plus_bucket(self, generate_mock_model, fake_bin, tmp_path):
        model = generate_mock_model(_MockModel())
        out_dir = str(tmp_path / "out")
        manifest = ftc.convert_fasttext_model(fake_bin, out_dir)
        assert manifest["matrices"]["input"]["rows"] == 3 + 8
        assert manifest["matrices"]["input"]["rows"] == model.get_input_matrix().shape[0]

    def test_matrix_files_are_little_endian_float32(self, generate_mock_model, fake_bin, tmp_path):
        model = generate_mock_model(_MockModel())
        out_dir = str(tmp_path / "out")
        ftc.convert_fasttext_model(fake_bin, out_dir)
        with open(os.path.join(out_dir, ftc.OUTPUT_MATRIX_FILE), "rb") as handle:
            raw = handle.read()
        assert raw == model.get_output_matrix().astype("<f4").tobytes()

    def test_manifest_checksums_match_files(self, generate_mock_model, fake_bin, tmp_path):
        generate_mock_model(_MockModel())
        out_dir = str(tmp_path / "out")
        manifest = ftc.convert_fasttext_model(fake_bin, out_dir)
        for spec in (
            manifest["matrices"]["input"],
            manifest["matrices"]["output"],
            manifest["vocab"],
            manifest["labels"],
        ):
            assert ftc._hash_file(os.path.join(out_dir, spec["file"])) == spec["sha256"]

    def test_refuses_to_clobber_without_overwrite(self, generate_mock_model, fake_bin, tmp_path):
        generate_mock_model(_MockModel())
        out_dir = str(tmp_path / "out")
        ftc.convert_fasttext_model(fake_bin, out_dir)
        with pytest.raises(FileExistsError):
            ftc.convert_fasttext_model(fake_bin, out_dir)
        ftc.convert_fasttext_model(fake_bin, out_dir, overwrite=True)


class TestVerification:
    def test_happy_path(self, generate_mock_model, fake_bin, tmp_path):
        generate_mock_model(
            _MockModel()
        )  # swaps out the real fastText model loader with a fake in-memory stub (_MockModel)
        out_dir = str(tmp_path / "out")
        ftc.convert_fasttext_model(fake_bin, out_dir)
        ftc.verify_export(fake_bin, out_dir)

    def test_detects_single_flipped_bit(self, generate_mock_model, fake_bin, tmp_path):
        """
        Goal is to check ExportVerificationError is raised when a last bit does not match.
        """
        generate_mock_model(_MockModel())
        out_dir = str(tmp_path / "out")
        ftc.convert_fasttext_model(fake_bin, out_dir)

        path = os.path.join(out_dir, ftc.INPUT_MATRIX_FILE)
        with open(path, "rb") as handle:
            data = bytearray(handle.read())
        data[0] ^= 0x01
        with open(path, "wb") as handle:
            handle.write(bytes(data))

        with pytest.raises(ftc.ExportVerificationError, match="bytewise"):
            ftc.verify_export(fake_bin, out_dir)

    def test_detects_reordered_vocab(self, generate_mock_model, fake_bin, tmp_path):
        generate_mock_model(_MockModel())
        out_dir = str(tmp_path / "out")
        ftc.convert_fasttext_model(fake_bin, out_dir)

        path = os.path.join(out_dir, ftc.VOCAB_FILE)
        with open(path, encoding="utf-8") as handle:
            words = handle.read().split("\n")[:-1]
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(reversed(words)) + "\n")

        with pytest.raises(ftc.ExportVerificationError, match="get_words"):
            ftc.verify_export(fake_bin, out_dir)


class TestBytewiseNotElementwise:
    """``==`` is false for NaN and true for -0.0 vs 0.0. Bytes are honest."""

    @pytest.mark.parametrize("sentinel", [np.float32("nan"), np.float32("-0.0")])
    def test_sentinel_values_survive_roundtrip(self, generate_mock_model, fake_bin, tmp_path, sentinel):
        model = _MockModel()
        original = model.get_output_matrix
        patched = original()
        patched[0, 0] = sentinel
        model.get_output_matrix = lambda: patched.copy()  # type: ignore[method-assign]
        generate_mock_model(model)

        out_dir = str(tmp_path / "out")
        ftc.convert_fasttext_model(fake_bin, out_dir)
        ftc.verify_export(fake_bin, out_dir)

        reloaded = ftc.load_exported_matrix(out_dir, "output")
        assert reloaded.tobytes() == patched.astype("<f4").tobytes()


@pytest.mark.skipif(not os.path.exists(FIXTURE_MODEL), reason="PR-5 fixture model not present")
class TestRealFixtureModel:
    """End-to-end against the committed tiny supervised model."""

    def test_converts_and_verifies(self, tmp_path):
        out_dir = str(tmp_path / "out")
        manifest = ftc.convert_fasttext_model(FIXTURE_MODEL, out_dir)
        ftc.verify_export(FIXTURE_MODEL, out_dir)

        config = manifest["config"]
        assert config["loss"] == "softmax"
        assert config["model"] == "supervised"
        assert config["dim"] > 0
        assert config["n_labels"] == len(ftc.load_exported_lines(out_dir, "labels"))

    def test_manifest_is_stable_across_runs(self, tmp_path):
        first = ftc.convert_fasttext_model(FIXTURE_MODEL, str(tmp_path / "a"))
        second = ftc.convert_fasttext_model(FIXTURE_MODEL, str(tmp_path / "b"))
        assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
