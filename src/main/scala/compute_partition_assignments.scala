package com.partitionAssignment

import org.apache.spark.sql.functions.udf
import scala.util.hashing.MurmurHash3
import scala.collection.mutable
import org.apache.spark.sql.types._

object ComputePartitionAssignmentsUDF {

  /* For PySpark/SQL users
    They can consume this UDF by calling:

        spark._jvm.com.minhash.ComputePartitionAssignmentsUDF.registerUdf(spark._jsparkSession)
        df = input_df.withColumn(
            "target_partitions",
            expr(f"compute_partition_assignments({minhash_signature}, {num_bands}, {rows_per_band}, {num_partitions})")
        )
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
      (signature: Array[Long], numBands: Int, rowsPerBand: Int, numPartitions: Int) =>
        ComputePartitionAssignmentsLogic(signature, numBands, rowsPerBand, numPartitions),
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
  val ComputePartitionAssignments = udf((signature: Array[Long], numBands: Int, rowsPerBand: Int, numPartitions: Int) =>
    ComputePartitionAssignmentsLogic(signature, numBands, rowsPerBand, numPartitions)
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
        - partitions: Array[Int]:
            - the number of elements is the unique band hash consists from 8 MinHash.
            - E.g. If 128 MinHash and 16 bands, then max partitions count is 8.

    Note: in process_partition_locally(), we compare bandHash to bandHash
    here we don't store bandHash in dataframe to not increase shuffle memory.
    it's a trade-off between latency gain vs minimizing memory.

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
