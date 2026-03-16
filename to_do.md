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
        DONE
    - try better instance
    main.tf
    ```
    core_instance_group {
    instance_type  = "r5.2xlarge"  # 8 cores, 64GB RAM (was r5.xlarge)
    instance_count = 8              # More nodes (was 4)
    
    ebs_config {
        size                 = "100"
        type                 = "gp3"
        volumes_per_instance = 1
    }
    }
    ```

    and spark-submit
    --executor-cores 4 \
    --num-executors 16 \
    - Implement hash mixing optimization to 
      eliminate string concatenation in MinHash
    - cache edge df (won't do)
    - Eliminate distinct
    - Use more efficient GraphFrames settings (won't do)
        # Add before connectedComponents()
        spark.conf.set("spark.graphx.pregel.checkpointInterval", "10")
        components = g.connectedComponents(algorithm="graphx")  
    - Implement caching mechanism
    - Test 1000 WET files with caching and num_partition= core_count    
    - I implemented skipping boiler template copy such as lisense.
      ```if band_size > MAX_BAND_SIZE:``` - just re-run.
    ☐ Implement incremental union-find for
      faster group building
    ☐ Increase similarity threshold to reduce
      false positives
    ☐ scala UDF
    ☐ Unit test for scala UDF
    ☐ 90k WET

# Learning - scala

- for function attributes like cache obj, in scala, use lazy val which is 
  singleton per JVM worker.
- lazy val should be in the companion object, not inside the UDF:
- val means the reference can't change — it always points to the same Set object. mutable means the contents of that Set can change.
    scala// val + mutable: the container is fixed, but you can modify its contents
    val partitions = scala.collection.mutable.Set[Int]()
    partitions += 1      // ✅ modifying contents
    partitions += 2      // ✅ still the same Set object
    partitions = otherSet // ❌ can't reassign the reference

    // var + immutable: the reference can change, but each Set is frozen
    var partitions = Set[Int]()
    partitions += 1      // ✅ but this creates a NEW Set and reassigns the variable
    partitions = otherSet // ✅ can reassign
- do NOT return in a lambda function
- every declaration required = sign
