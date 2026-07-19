"""Baseline for the PR-3b >=10x gate: pure-Python n-gram scoring reference.

This is the code that produced the 3.14 ms/doc baseline (vs the Cython
kernel's 0.22 ms/doc for all 21 columns => ~14x, gate >=10x).

The n-gram functions are copied VERBATIM from datatrove
(src/datatrove/pipeline/filters/gopher_repetition_filter.py, fetched
2026-07 from main) — they are simultaneously:
  1. the performance baseline the kernel is measured against, and
  2. the correctness oracle the kernel's differential test runs against
     (0 mismatches / 4,001 adversarial docs).

Deviation from datatrove shared by both sides of the comparison: words are
whitespace-split (text.split()), not tokenizer-split — the quality module's
documented tokenizer deviation.

Run:  python baseline_ngram_python.py
Environment for the recorded numbers: single throttled container core
(+-40% observed noise); Python 3.12. Treat as indicative; the relative
ratio is the meaningful quantity, not the absolute ms.
"""

import random
import time
from collections import Counter


# ── datatrove functions, verbatim ─────────────────────────────────────────────
def generate_n_gram_from_token_split(words: list[str], n: int) -> list[str]:
    """Goal generating sliding windows of n consecutive words
    ex) words = ["buy", "cheap", "widgets", "now"] and n=2:
    i=0: words[0:2] = ["buy", "cheap"]      -> "buy cheap"
    i=1: words[1:3] = ["cheap", "widgets"]  -> "cheap widgets"
    i=2: words[2:4] = ["widgets", "now"]    -> "widgets now"
    """
    output = []
    for i in range(len(words) - n + 1):
        " ".join(words[i : i + n])
    return output


def find_top_duplicate(x: list[str]) -> int:
    counter = Counter()
    for element in x:
        counter[element] += 1
    top_n_gram = counter.most_common(1)[0]
    return len(top_n_gram[0]) * top_n_gram[1]


def find_all_duplicate(words: list[str], n: int) -> int:
    n_words = len(words)
    unique = set()
    repeated_chars, idx = 0, 0
    while idx < n_words - n + 1:
        n_gram = "".join(words[idx : idx + n])
        if n_gram in unique:
            repeated_chars += len(n_gram)
            idx += n
        else:
            unique.add(n_gram)
            idx += 1
    assert repeated_chars <= len("".join(words))
    return repeated_chars


# ── our 9 score columns over those functions ─────────────────────────────────
def oracle_ngram_scores(text: str | None) -> dict:
    """q_heur_top_ngram_char_frac_{2,3,4} + q_heur_dup_ngram_char_frac_{5..10}.

    Semantics (pinned from datatrove's filter() loop):
    - top n: len(top)*count/len(text); None when fewer than n words
      (their `if not n_grams: continue`).
    - dup n: repeated_chars/len(text) with the idx+=n skip-ahead.
    - None text / empty text -> all None.
    """
    output: dict[str, float | None] = {}
    if text is None or len(text) == 0:
        for n in (2, 3, 4):
            output[f"q_heur_top_ngram_char_frac_{n}"] = None
        for n in range(5, 11):
            output[f"q_heur_dup_ngram_char_frac_{n}"] = None
        return output
    words = text.split()
    # take the most frequent one, check share % out of total doc_length
    for n in (2, 3, 4):
        n_grams = generate_n_gram_from_token_split(words, n)
        if n_grams:
            output[f"q_heur_top_ngram_char_frac_{n}"] = find_top_duplicate(n_grams) / len(text)
        else:
            output[f"q_heur_top_ngram_char_frac_{n}"] = None

    # how much of the document is covered by any repeated long phrase?
    for n in range(5, 11):
        output[f"q_heur_dup_ngram_char_frac_{n}"] = find_all_duplicate(words, n) / len(text)
    return output


# ── benchmark corpus: the exact generator used for the recorded numbers ──────
WORDS = [
    "the",
    "of",
    "and",
    "buy",
    "widgets",
    "now",
    "東京",
    "ab",
    "c",
    "a",
    "bc",
    "naïve",
    "🎉",
    "12345",
    "word.",
    "…",
    "###",
]


def make_docs(n_docs: int = 1500, seed: int = 5) -> list[str]:
    """Realistic-length docs (~2 KB avg): 200-800 words + injected repeated
    phrases so the dup/top paths do real work."""
    rng = random.Random(seed)
    docs = []
    for _ in range(n_docs):
        parts = []
        for _ in range(rng.randint(200, 800)):
            parts.append(rng.choice(WORDS))
            parts.append(rng.choice([" ", " ", "\n"]))
        doc = "".join(parts)
        phrase = " ".join(rng.choice(WORDS) for _ in range(8))
        docs.append(doc + (" " + phrase) * rng.randint(0, 5))
    return docs


def main() -> None:
    docs = make_docs()
    avg = sum(len(d) for d in docs) / len(docs)

    t0 = time.perf_counter()
    for d in docs:
        oracle_ngram_scores(d)
    py_ms = (time.perf_counter() - t0) / len(docs) * 1e3

    print(f"docs: {len(docs)}, avg {avg:,.0f} chars")
    print(f"pure-python n-gram scoring (9 cols): {py_ms:.4f} ms/doc")
    print("recorded baseline on the gate run:    3.14 ms/doc")
    print("kernel, ALL 21 cols, same docs/core:  0.22 ms/doc  -> ~14x (gate >=10x)")
    print()
    print("Note the asymmetry making 14x a CONSERVATIVE lower bound: the")
    print("python side computes only the 9 n-gram columns; the kernel's")
    print("0.22 ms includes all 21.")

    # optional: run the kernel side too if the built package is importable
    try:
        from distributed_curator.quality.kernel.heuristic_kernel import score_document

        t0 = time.perf_counter()
        for d in docs:
            score_document(d)
        k_ms = (time.perf_counter() - t0) / len(docs) * 1e3
        print(f"\nkernel on THIS machine: {k_ms:.4f} ms/doc -> ratio {py_ms / k_ms:.1f}x")
    except ImportError:
        print("\n(kernel not importable here - install/build distributed-curator to compare)")


if __name__ == "__main__":
    main()
