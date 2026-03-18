package com.minhash

import org.apache.spark.sql.functions.udf
import scala.util.hashing.MurmurHash3
import scala.collection.mutable
import scala.util.Random
import scala.util.matching.Regex

object MinHashUDF {

    // Companion Object with lazy val - Computed once per JVM, shared across all UDF calls on that executor
    lazy val seedsCache: Array[Long] = {
        val random = new Random(42) // Same seed for reproducibility
        Array.fill(1024)(random.nextLong() & 0xFFFFFFFFL)
    }

    /* For PySpark/SQL users 
    They can consume this UDF by calling:

        spark._jvm.com.minhash.MinHashUDF.registerUdf(spark._jsparkSession)        
        df = input_df.withColumn(
            "minhash_signature",
            expr(f"compute_minhash({text_column}, {num_hashes}, {ngram}, {str(remove_articles).lower()})")
        )
    */
    def registerUdf(spark: org.apache.spark.sql.SparkSession): Unit = {
        spark.udf.register("compute_minhash", 
            (text: String, numHashes: Int, ngram: Int, removeArticles: Boolean) => {
                computeMinHashLogic(text, numHashes, ngram, removeArticles)
            }
        )
    }

    /* For DataFrame API users 
    They can consume this UDF by creating python - JVM bridge:
        minhash_udf = spark._jvm.com.minhash.MinHashUDF.computeMinHash()
        def minhash_batch_udf(text_col, num_hashes, ngram, remove_articles):
            return Column(minhash_udf.apply(
                _to_java_column(col(text_column)),
                _to_java_column(lit(num_hashes)),
                _to_java_column(lit(ngram)),
                _to_java_column(lit(remove_articles))
            ))

        df_with_signatures = input_df.withColumn(
            "minhash_signature",
            minhash_batch_udf(col(text_column))
        )
    */
    val computeMinHash = udf((text: String, numHashes: Int, ngram: Int, removeArticles: Boolean) => {
        computeMinHashLogic(text, numHashes, ngram, removeArticles)
    })

    // this function returns array of minHash with size of numHashes (default 128).
    private def computeMinHashLogic(text: String, numHashes: Int, ngram: Int, removeArticles: Boolean): Array[Long] = {
        var normalizedTexts = text.toLowerCase()

        if (removeArticles) {
            val articlesPattern: Regex = """\b(the|a|an|this|that|these|those)\b""".r
            normalizedTexts = articlesPattern.replaceAllIn(normalizedTexts, "").replaceAll("""\s+""", " ").trim
        }
        
        if (normalizedTexts.length < ngram) {
            Array.fill(numHashes)(0L)
        } else {
            // Generate unique shingles for this specific text string (memory-optimized)
            val shinglesIterator: Iterator[String] = normalizedTexts.sliding(ngram)
            val baseHashes = mutable.HashSet[Long]()
            while (shinglesIterator.hasNext) {
                // HASH MIXING OPTIMIZATION: Hash each shingle ONCE, then mix with seeds
                // .toLong because we want unsigned 32 bits, so first make it to 64 bit
                // what's the problem of signed ? signature[i] = min(signature[i], hash_val)
                // so negative will mess up min().
                // 0xFFFFFFFFL's L is to make it Long.
                val hash: Long = MurmurHash3.stringHash(shinglesIterator.next(), 0).toLong & 0xFFFFFFFFL
                baseHashes += hash
            }
            // Apply hash mixing: (base_hash XOR seed) for each combination
            val baseHashesArray = baseHashes.toArray  // Set → Array for indexed access
            if (baseHashesArray.isEmpty) {
                Array.fill(numHashes)(0L)
            }
            else {
                // Initialize. Eventually get minimum across all shingles for each hash function
                val signature = Array.fill(numHashes)(0xFFFFFFFFL)

                // For each shingle × each 128 hash function, track minimum. Output is array of 128 samples of minimum hashes
                var j: Int = 0
                while (j < baseHashesArray.length) {
                    val baseHash: Long = baseHashesArray(j)
                    var i: Int = 0
                    while (i < numHashes) {
                        val mixedHash = (baseHash ^ seedsCache(i)) & 0xFFFFFFFFL
                        if (mixedHash < signature(i)) {
                            signature(i) = mixedHash
                        }
                        i += 1
                    }
                    j += 1
                }
            
                signature
            }
        }
        
    }
}