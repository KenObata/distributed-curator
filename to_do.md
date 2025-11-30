# ToDo list
- Can we make a class for partition aware de-duping ?
- test with real dataset from web crawler.
- create bench mark script
```
# benchmarks/benchmark_ours.py
"""Compare against MLlib and other solutions"""

datasets = [
    ("1GB", "s3://bucket/1gb.parquet"),
    ("10GB", "s3://bucket/10gb.parquet"),
    ("100GB", "s3://bucket/100gb.parquet"),
]

results = []
for name, path in datasets:
    df = spark.read.parquet(path)
    
    # Our implementation
    our_time = benchmark_our_implementation(df)
    
    # MLlib (if it doesn't OOM)
    mllib_time = benchmark_mllib(df)
    
    results.append({
        "dataset": name,
        "ours": our_time,
        "mllib": mllib_time,
        "speedup": mllib_time / our_time
    })

# Generate nice visualization
plot_results(results)
```

- Critical Optimizations for Real Scale
    - Priority 1: Proper Connected Components for Duplicate Clusters
        - DONE
    - Week 6-7: Memory and Performance Optimizations
      Priority 3: Signature Compression and Efficient Storage
        - won't do. we found a problem. if we sort and store delta between
          two min hashes within a doc, it loses ordering. 
          Reminder that the ordering matters because each 128 min hash is based on different seed.
    - complete EMR run
        - DONE
    - optimize UDF by vectorized UDF
        ```
        from pyspark.sql.functions import pandas_udf
        import pandas as pd

        @pandas_udf(ArrayType(IntegerType()))
        def minhash_pandas_udf(texts: pd.Series) -> pd.Series:
            return texts.apply(
                lambda text: compute_minhash_signature(text=text, num_hashes=64, ngram=9, normalize=True)
            )
        ```