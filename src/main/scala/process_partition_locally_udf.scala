package com.processPartitionLocally

import org.apache.spark.sql.{DataFrame, Row} // need DataFrame and Row
import scala.collection.mutable
import scala.util.hashing.MurmurHash3
import org.slf4j.LoggerFactory

object ProcessPartitionLocallyUDF {
  /* Difference from other UDFs is that this function runs per partition, not per row.
    This means we directly calls JVM method from df_partitioned._jdf
    so no Spark UDF registry involved.

    They can consume this JVM DataFrame by calling:

        jvm_helper = spark._jvm.com.processPartitionLocally.ProcessPartitionLocallyUDF

        similar_pairs_jdf = jvm_helper.processPartitions(
            df_partitioned._jdf,  # Pass the underlying JVM DataFrame
            num_bands,
            rows_per_band,
            similarityThreshold
            )
        similar_pairs_df = DataFrame(similar_pairs_jdf, spark)
   */
  val logger = LoggerFactory.getLogger(getClass)

  // Data class for processPartitions's resultRDD
  case class LocalDoc(docId: String, signature: Array[Long], partitionId: Int)

  case class SimilarityPairs(
    doc1: String,
    doc2: String,
    similarity: Double,
    partition_id: Int
  ) // snake case because pyspark

  private def estimateSimilarity(sig1: Array[Long], sig2: Array[Long]): Double =
    /* Estimate Jaccard similarity from MinHash signatures
     */
    if (sig1.isEmpty || sig2.isEmpty || sig1.length != sig2.length) {
      0.0
    } else {
      // Count matching MinHash values
      var matches: Int = 0
      var i            = 0
      while (i < sig1.length) {
        if (sig1(i) == sig2(i) && sig1(i) != 0L) matches += 1
        i += 1
      }
      matches.toDouble / sig1.length
    }

  def processPartitions(
    df: DataFrame,
    numBands: Int,
    rowsPerBand: Int,
    similarityThreshold: Double
  ): DataFrame = {
    /*
            Process all documents within a single partition locally.
            This is where the magic happens - no network I/O needed!

            Why do we do this function per partition, not per row like other UDFs?
            It's because we need to compare documents within the same partition, not transform each row independently.
            With per row UDF process, it can't compare to neighbors.
            This function needs to see ALL docs in partition.
     */

    val resultRDD = df.rdd.mapPartitions { iterator: Iterator[Row] =>
      /*
            - iterator: as a reminder mapPartitions runs your function once per partition
            - Each partition gives us an Iterator[Row] of ALL rows in that partition
       */

      // Collect documents in this partition
      val localDocs = mutable.ArrayBuffer[LocalDoc]()
      for (row <- iterator) localDocs += LocalDoc(
        row.getAs[String]("doc_id"),
        row.getAs[Seq[Long]]("minhash_signature").toArray,
        row.getAs[Int]("partition_id")
      )

      if (localDocs.isEmpty) {
        Iterator.empty
      } else {
        /* Build local LSH index for this partition
         */
        // there can be multiple docs per band key of {band_id}_{band_hash}
        val bandIndex = mutable.HashMap[String, mutable.ArrayBuffer[LocalDoc]]()
        for (doc <- localDocs) {
          val sig: Array[Long] = doc.signature // 128 MinHash
          if (sig != null && sig.nonEmpty) {
            // Generate bands
            for (bandId <- 0 until numBands) {
              val start: Int = bandId * rowsPerBand
              val end: Int   = math.min(start + rowsPerBand, sig.length)
              if (start < sig.length) {
                val bandHash = MurmurHash3.arrayHash(sig.slice(start, end))

                // Add to local index
                val bandKey = s"${bandId}_$bandHash"
                if (!bandIndex.contains(bandKey)) {
                  bandIndex(bandKey) = mutable.ArrayBuffer[LocalDoc]()
                }
                bandIndex(bandKey) += doc
              }

            }
          }

        }

        val seenPairs    = mutable.Set[Tuple2[String, String]]()
        val similarPairs = mutable.ArrayBuffer[SimilarityPairs]()

        // Safety cap: skip bands with too many docs to prevent O(n²) explosion
        // Bands with >1000 docs are likely hash collisions on common patterns
        val MAX_BAND_SIZE     = 1000
        var skippedBands: Int = 0
        var maxBandSeen: Int  = 0
        for ((bandKey, docsInBand) <- bandIndex) {
          val bandSize = docsInBand.length
          maxBandSeen = math.max(maxBandSeen, bandSize)
          if (bandSize > MAX_BAND_SIZE) {
            skippedBands += 1
          } else if (bandSize >= 2) {
            // Compare all pairs in this "{band_id}_{band_hash}" in this partition
            var i = 0
            while (i < docsInBand.length) {
              val doc1 = docsInBand(i)
              var j    = i + 1
              while (j < docsInBand.length) {
                /* Create canonical pair ID
                                Sort ensures there is no dups in similar_pairs with this case (doc1,doc2) and (doc2, doc1).
                                This case never happens.
                 */
                val doc2 = docsInBand(j)
                val pairId: Tuple2[String, String] =
                  if (doc1.docId < doc2.docId) (doc1.docId, doc2.docId) else (doc2.docId, doc1.docId)
                if (!seenPairs.contains(pairId)) {
                  seenPairs += pairId
                  val similarity = estimateSimilarity(doc1.signature, doc2.signature)
                  if (similarity >= similarityThreshold) {
                    similarPairs += SimilarityPairs(
                      pairId._1,
                      pairId._2,
                      similarity,
                      doc1.partitionId
                    )
                  }
                }
                j += 1
              }
              i += 1
            }
          }

        } // end of for loop
        logger.info(s"skippedBands: $skippedBands")
        logger.info(s"maxBandSeen: $maxBandSeen")
        similarPairs.iterator // each executor write to val resultRDD
      }                       // End of else block
    }                         // End of val resultRDD

    val spark = df.sparkSession
    import spark.implicits._
    resultRDD.toDF() // driver collects similarPairs.iterator
  }                  // End of def processPartitions()

}                    // End of object
