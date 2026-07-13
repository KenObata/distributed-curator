#!/usr/bin/env python3
"""Benchmark harness for the heuristic quality-scoring layer.

Measures scoring throughput in rows/sec/core. Two input modes:

  --input <path>     parquet with a 'text' column (e.g. WET-derived sample)
  --rows N           synthetic web-like documents generated in-memory

The scored DataFrame is written to the noop sink so the measurement includes
full expression evaluation on every row but no output I/O.

Usage:
  python scripts/benchmark_quality_heuristics.py --rows 200000
  python scripts/benchmark_quality_heuristics.py --input s3a://bucket/wet_sample.parquet
"""

from __future__ import annotations

import argparse
import random
import time

from pyspark.sql import SparkSession

from distributed_curator.quality import HeuristicConfig, compute_heuristic_scores

WORDS = [
    "the",
    "of",
    "and",
    "to",
    "in",
    "is",
    "was",
    "for",
    "that",
    "with",
    "on",
    "as",
    "by",
    "at",
    "from",
    "this",
    "have",
    "are",
    "be",
    "it",
    "data",
    "system",
    "value",
    "page",
    "online",
    "product",
    "service",
    "free",
    "new",
    "more",
    "time",
    "people",
]


def synthetic_doc(rng: random.Random) -> str:
    """Web-like document: 1-8 paragraphs, occasional bullets/repeats."""
    paragraphs = []
    for _ in range(rng.randint(1, 8)):
        lines = []
        for _ in range(rng.randint(1, 6)):
            n = rng.randint(4, 30)
            line = " ".join(rng.choice(WORDS) for _ in range(n))
            if rng.random() < 0.08:
                line = "• " + line
            lines.append(line)
        if rng.random() < 0.1 and lines:
            lines.append(lines[0])  # duplicated line
        paragraphs.append("\n".join(lines))
    return "\n\n".join(paragraphs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="parquet path with a 'text' column")
    parser.add_argument("--rows", type=int, default=100_000, help="synthetic row count (ignored with --input)")
    parser.add_argument("--partitions", type=int, default=8)
    args = parser.parse_args()

    spark = SparkSession.builder.appName("QualityHeuristicsBenchmark").getOrCreate()
    cores = spark.sparkContext.defaultParallelism

    if args.input:
        df = spark.read.parquet(args.input)
        source = args.input
    else:
        rng = random.Random(42)
        rows = [(f"doc_{i}", synthetic_doc(rng)) for i in range(args.rows)]
        df = spark.createDataFrame(rows, ["doc_id", "text"]).repartition(args.partitions)
        source = f"synthetic ({args.rows} rows)"

    df = df.cache()
    n_rows = df.count()  # materialize cache so generation time is excluded

    scored = compute_heuristic_scores(df, config=HeuristicConfig())

    start = time.perf_counter()
    scored.write.format("noop").mode("overwrite").save()
    elapsed = time.perf_counter() - start

    print(f"source:           {source}")
    print(f"rows:             {n_rows}")
    print(f"cores:            {cores}")
    print(f"wall time:        {elapsed:.2f}s")
    print(f"rows/sec:         {n_rows / elapsed:,.0f}")
    print(f"rows/sec/core:    {n_rows / elapsed / cores:,.0f}")

    df.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()
