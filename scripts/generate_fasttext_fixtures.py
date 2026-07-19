#!/usr/bin/env python3
"""Generate fastText parity fixtures using the OFFICIAL fastText bindings.

This script is the ORACLE for the fastText scoring layer: it produces the
expected scores that test/unit_test/quality_fasttext_test.py asserts against.
It is run offline by a maintainer, never at test time — the tests read the
committed JSON and require no fasttext installation at all.

Why the isolation
-----------------
The official ``fasttext`` package (0.9.3, last released 2022) is INCOMPATIBLE
with NumPy 2.x: ``model.predict()`` raises

    ValueError: Unable to avoid copy while creating an array as requested

from ``np.array(..., copy=False)`` in FastText.py. Training, ``save_model``,
``labels`` and ``get_input_matrix()`` still work; only ``predict()`` — the
exact call that defines this oracle — is broken. Rather than pin the whole
repo to ``numpy<2`` (it would conflict with the runtime NumPy used by the
MinHash path), the oracle runs in a throwaway virtualenv:

    python -m venv /tmp/ftenv
    /tmp/ftenv/bin/pip install "numpy<2" fasttext
    /tmp/ftenv/bin/python scripts/generate_fasttext_fixtures.py

A NumPy-2-compatible community fork (``fasttext-numpy2``, MIT, one-line
change) exists and was verified to return bit-identical scores. It is NOT
used here on purpose: the correctness gate should reference the authoritative
implementation, not a fork. Contributors who want ``import fasttext`` to work
in a modern environment may install it for convenience.

Regenerating fixtures is only necessary when the tiny models or the document
set change; both are committed, so results are reproducible.
"""

import json
import os
import sys

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test", "unit_test", "fixtures")
QUALITY_MODEL = os.path.join(FIXTURE_DIR, "tiny_quality_model.bin")
LID_MODEL = os.path.join(FIXTURE_DIR, "tiny_lid_model.bin")
OUTPUT = os.path.join(FIXTURE_DIR, "fasttext_golden.json")

NEGATIVE_LABEL = "__label__cc"
LABEL_PREFIX = "__label__"

# Documents chosen to exercise: clear positive/negative class, the newline
# normalization path (\n, \r\n, and the exotic separators splitlines() breaks
# on but replace("\n") would not), CJK, emoji, leading/trailing whitespace,
# and degenerate inputs.
DOCUMENTS = [
    {"doc_id": "hq_prose", "text": "explain the reason carefully because a thoughtful answer requires evidence"},
    {"doc_id": "cc_spam", "text": "buy cheap widgets now click here free shipping limited time offer"},
    {"doc_id": "multiline_lf", "text": "explain the reason carefully\nbecause a thoughtful answer requires evidence"},
    {"doc_id": "multiline_crlf", "text": "buy cheap widgets now\r\nclick here free shipping"},
    {"doc_id": "exotic_separator", "text": "explain the reason\u2028carefully because evidence matters"},
    {"doc_id": "leading_trailing_ws", "text": "   buy cheap widgets now click here   \n"},
    {"doc_id": "japanese", "text": "東京 は 日本 の 首都 であり 多く の 人 が 住んで いる 都市 です"},
    {"doc_id": "english_lid", "text": "the quick brown fox jumps over the lazy dog every single morning"},
    {"doc_id": "emoji", "text": "explain the reason carefully 🎉 because evidence matters 🎉"},
    {"doc_id": "mixed_script", "text": "東京 tokyo は the 首都 capital です"},
    {"doc_id": "single_word", "text": "widgets"},
    {"doc_id": "whitespace_only", "text": "   \n  "},
    {"doc_id": "empty", "text": ""},
]


def normalize_text(content):
    """DCLM-verbatim (classify_fasttext_hq_prob)."""
    return " ".join(content.strip().splitlines())


def main():
    try:
        import fasttext
        import numpy as np
    except ImportError:
        sys.exit("fasttext not installed; see this script's docstring for the venv recipe")

    print(f"numpy {np.__version__}")
    if np.__version__.startswith("2."):
        print(
            "WARNING: official fasttext predict() is broken on NumPy 2.x; "
            "expect ValueError. Use the numpy<2 venv from the docstring."
        )

    qm = fasttext.load_model(QUALITY_MODEL)
    lm = fasttext.load_model(LID_MODEL)

    out = []
    for doc in DOCUMENTS:
        text = normalize_text(doc["text"])

        q_labels, q_probs = qm.predict(text)
        q_label, q_prob = q_labels[0], float(q_probs[0])
        q_score = 1.0 - q_prob if q_label == NEGATIVE_LABEL else q_prob

        l_labels, l_probs = lm.predict(text)
        l_label = l_labels[0]
        if l_label.startswith(LABEL_PREFIX):
            l_label = l_label[len(LABEL_PREFIX) :]

        out.append(
            {
                "doc_id": doc["doc_id"],
                "text": doc["text"],
                "_derivation": (
                    f"normalized={text!r}; quality top label {q_labels[0]} p={q_prob!r} "
                    f"-> q_ft_score={'1-p' if q_label == NEGATIVE_LABEL else 'p'}; "
                    f"lid top label {l_labels[0]} p={float(l_probs[0])!r}"
                ),
                "expected": {
                    "q_ft_score": q_score,
                    "q_lid_lang": l_label,
                    "q_lid_score": float(l_probs[0]),
                },
            }
        )

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"wrote {len(out)} fixtures -> {OUTPUT}")


if __name__ == "__main__":
    main()
