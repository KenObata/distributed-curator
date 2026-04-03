package com.partitionAssignment

import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.Row
import scala.util.hashing.MurmurHash3
import org.apache.spark.sql.types._

object ComputePartitionAssignmentsUDF {

  /* For PySpark/SQL users
    They can consume this UDF by calling:

        spark._jvm.com.minhash.ComputePartitionAssignmentsUDF.registerUdf(spark._jsparkSession)
        df = input_df.withColumn(
            "target_partitions",
            expr(f"compute_partition_assignments({minhash_signature}, {num_bands}, {rows_per_band}, {num_partitions})")
        )
    Note for Row() conversion within UDF:
      - with returnType, we tell spark the output schema but Spark doesn't know UDF input types.
        so if it gets null for a primitive (ex.Int, Long), your closure silently gets 0 instead of null.
        Spark 3.5 blocks this by default.
      - without returnType, — Spark infers types
      - Row maps directly to StructType, no type ambiguity.
   */
  val returnType = StructType(
    Seq(
      StructField("target_partitions", ArrayType(IntegerType), false),
      StructField("band_hashes", ArrayType(IntegerType), false)
    )
  )

  def registerUdf(spark: org.apache.spark.sql.SparkSession): Unit = spark.udf.register(
    "compute_partition_assignments",
    udf(
      // Seq because Spark passes array columns as WrappedArray, not Array[Long]
      (signature: Seq[java.lang.Long], numBands: Int, rowsPerBand: Int, numPartitions: Int) => {
        // box signatureArray because Spark passes WrappedArray[java.lang.Long] (boxed objects),
        // but Seq[Long] expects Scala primitive longs
        val signatureArray = signature.map(sig => sig.longValue()).toArray
        val (partitions, bandHashes) =
          ComputePartitionAssignmentsLogic(signatureArray, numBands, rowsPerBand, numPartitions)
        Row(partitions, bandHashes)
      },
      returnType
    )
  )

  /* For DataFrame API users
    They can consume this UDF by creating python - JVM bridge:
        compute_partition_assignments_udf = spark._jvm.com.minhash.ComputePartitionAssignmentsUDF.ComputePartitionAssignments()
        def compute_partition_assignments_batch_udf(signature: Array[Long]):
            return Column(compute_partition_assignments_udf.apply(
                _to_java_column(signature)
            ))

        df_with_partitions = input_df.withColumn(
            "target_partitions",
            compute_partition_assignments_batch_udf(col("minhash_signature"))
        )
   */
  lazy val ComputePartitionAssignments = udf(
    (signature: Array[java.lang.Long], numBands: Int, rowsPerBand: Int, numPartitions: Int) => {
      val signatureArray = signature.map(sig => sig.longValue()).toArray
      val (partitions, bandHashes) =
        ComputePartitionAssignmentsLogic(signatureArray, numBands, rowsPerBand, numPartitions)
      Row(partitions, bandHashes)
    },
    returnType
  )

  /*
    Determine which partitions this document needs to be sent to
    based on its LSH bands. This ensures similar docs end up in same partition.

     - Input:
        - signature: 128 MinHash samples in Array.
        - numBands: default 16
        - rowsPerBand: numHashes // numBands
        - numPartitions: equivalent to spark.sql.shuffle.partitions
    - Output:
        - StructType: Array[Int], Array[Int]:
            - the number of elements equals numBands (one per band).
              Different bands may map to the same partition_id but with different band_hashes.
            - E.g. If 128 MinHash and 16 bands, then max partitions count is 8.

    - we expose bandHash in the output so that later in step3, we can do
      deterministic salting for hot partitions.

    - In process_partition_locally(), we compare docs by ${bandId}_${bandHash}
    but in this function, the purpose is to map bandHash -> partition_id,
    so no need to create the composite key, ${bandId}_${bandHash}, in return output.
   */
  private[partitionAssignment] def ComputePartitionAssignmentsLogic(
    signature: Array[Long],
    numBands: Int,
    rowsPerBand: Int,
    numPartitions: Int
  ): (Array[Int], Array[Int]) =
    if (signature == null || signature.isEmpty || signature.forall(_ == 0)) {
      (Array(0), Array(0))
    } else {
      val partitions = new Array[Int](numBands)
      val bandHashes = new Array[Int](numBands)

      /*
        Use count for Array index access in case signature.length < numBands * rowsPerBand.
        .take(count) excludes uninitialized trailing positions.
       */
      var count = 0
      for (bandId <- 0 until numBands) {
        val start: Int = bandId * rowsPerBand
        val end: Int   = math.min(start + rowsPerBand, signature.length)

        if (start < signature.length) {
          // Hash the band to get partition assignment
          val bandValues = signature.slice(start, end)
          val bandHash   = MurmurHash3.arrayHash(bandValues)

          // abs because MurmurHash3.arrayHash returns signed Int.
          // Here, signed int is okay because we are not deriving MIN().
          val partitionId = math.abs(bandHash) % numPartitions
          partitions(count) = partitionId
          bandHashes(count) = bandHash
          count += 1
        }
      }
      (partitions.take(count), bandHashes.take(count))
    }

}
