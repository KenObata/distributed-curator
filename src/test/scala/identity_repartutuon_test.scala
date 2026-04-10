package com.partitionAssignment

import org.scalatest.funsuite.AnyFunSuite

import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.types.{ArrayType, IntegerType, LongType, StringType, StructField, StructType}
import org.apache.spark.sql.SparkSession
import scala.collection.mutable
import scala.collection.{mutable => dataSeq}
import org.apache.spark.sql.functions.spark_partition_id
import org.apache.spark.rdd.RDD

class IdentityRepartitionTest extends AnyFunSuite {

  /* ScalaTest needs class because the test runner creates a fresh instance per test
       because each test should be independent
   */
  lazy val spark: SparkSession = SparkSession
    .builder()
    .master("local[*]")
    .appName("IdentityRepartitionTest")
    .getOrCreate()

  val numPartitions = 10

  val inputFields: Seq[StructField] = Seq(
    StructField("doc_id", StringType),
    StructField("partition_id", IntegerType),
    StructField("minhash_signature", ArrayType(LongType))
  )

  val inputSchema: StructType = StructType(inputFields)

  /* =========================================================================
     Output schema & length
   ========================================================================= */
  test("test_schema") {
    // Output schema should match input schema exactly.
    val minhashSignature: Array[Long] = Array(1L, 2L, 3L)
    val rddExploded: RDD[Row] = spark.sparkContext.parallelize(
      Seq(
        Row("doc_1", 0, minhashSignature)
      )
    )
    val dfExploded: DataFrame = spark.createDataFrame(rddExploded, inputSchema)

    val dfPartitioned: DataFrame = IdentityRepartition.repartition(dfExploded, "partition_id", numPartitions)
    assert(dfPartitioned.schema == inputSchema)
  }

  test("test_identity_mapping") {
    val minhashSignature: Array[Long] = Array(1L, 2L, 3L)
    /*
        val data = mutable.ArrayBuffer[Row]()
        for (i <- 0 until numPartitions) {
            data += Row(s"doc_$i", i, minhashSignature)
        }
     */
    val inputRows: IndexedSeq[Row] = (0 until numPartitions).map(i => Row(s"doc_$i", i, minhashSignature))

    val rddExploded              = spark.sparkContext.parallelize(inputRows)
    val dfExploded: DataFrame    = spark.createDataFrame(rddExploded, inputSchema)
    val dfPartitioned: DataFrame = IdentityRepartition.repartition(dfExploded, "partition_id", numPartitions)
    val outputRows: Array[Row]   = dfPartitioned.withColumn("physical_partition_id", spark_partition_id()).collect()
    for (row <- outputRows) assert(row.getAs[Int]("physical_partition_id") == row.getAs[Int]("partition_id"))
  }

  test("test_no_collision") {
    val minhashSignature: Array[Long] = Array(1L, 2L, 3L)
    val rows: IndexedSeq[Row]         = (0 until numPartitions).map(i => Row(s"doc_$i", i, minhashSignature))
    val rddExploded                   = spark.sparkContext.parallelize(rows)
    val dfExploded: DataFrame         = spark.createDataFrame(rddExploded, inputSchema)
    val dfPartitioned: DataFrame      = IdentityRepartition.repartition(dfExploded, "partition_id", numPartitions)

    val physical_partition_count: Long = dfPartitioned
      .withColumn("physical_partition_id", spark_partition_id())
      .select("physical_partition_id")
      .distinct()
      .count()
    assert(physical_partition_count == numPartitions)
  }

  test("test_same_partition_id_should_be_colocated") {
    // same partition_id should be assigned the same physical partition_id
    val minhashSignature: Array[Long] = Array(1L, 2L, 3L)
    val rddExploded: RDD[Row] = spark.sparkContext.parallelize(
      Seq(
        Row("doc_1", 0, minhashSignature),
        Row("doc_2", 0, minhashSignature),
        Row("doc_3", 1, minhashSignature)
      )
    )
    val dfExploded: DataFrame = spark.createDataFrame(rddExploded, inputSchema)

    val dfPartitioned: DataFrame = IdentityRepartition.repartition(dfExploded, "partition_id", numPartitions)
    val rows: Array[Row]         = dfPartitioned.withColumn("physical_partition_id", spark_partition_id()).collect()

    val doc_id2physical_partition_id =
      rows.map(row => row.getAs[String]("doc_id") -> row.getAs[Int]("physical_partition_id")).toMap
    /*for (row <- rows) {
        doc_id2physical_partition_id += (row.getAs[String]("doc_id") -> row.getAs[Int]("physical_partition_id"))
    }*/
    assert(doc_id2physical_partition_id("doc_1") == doc_id2physical_partition_id("doc_2"))
    assert(doc_id2physical_partition_id("doc_1") != doc_id2physical_partition_id("doc_3"))

  }

  test("test_salted_partition_ids_beyond_num_partitions") {
    /*
    salting can make logical partition_id > num_parittions as follows:
    F.col("partition_id") + F.abs(F.col("band_hash")) % num_splits

    In identity_repartition, partitionBy is based on
    Utils.nonNegativeMod(key.hashCode, numPartitions) so this shoud wrap via modulo without any errors.
     */
    val minhashSignature: Array[Long] = Array(1L, 2L, 3L)
    val rddExploded: RDD[Row] = spark.sparkContext.parallelize(
      Seq(
        Row("doc_1", 5, minhashSignature),  // normal
        Row("doc_2", 15, minhashSignature), // 15 % 10 = 5, same physical as doc_1
        Row("doc_3", 25, minhashSignature), // 25 % 10 = 5, same physical
        Row("doc_4", 8, minhashSignature),  // normal
        Row("doc_5", 18, minhashSignature)  // 18 % 10 = 8, same physical as doc_4
      )
    )
    val dfExploded: DataFrame    = spark.createDataFrame(rddExploded, inputSchema)
    val dfPartitioned: DataFrame = IdentityRepartition.repartition(dfExploded, "partition_id", numPartitions)
    val rows: Array[Row]         = dfPartitioned.withColumn("physical_partition_id", spark_partition_id()).collect()
    val doc_id2physical_partition_id: Map[String, Int] =
      rows.map(row => row.getAs[String]("doc_id") -> row.getAs[Int]("physical_partition_id")).toMap

    val expectedPhysicalPartitionId: Map[String, Int] = rddExploded
      .collect()
      .map(row =>
        // raw Row without schema, so use getString instead of getAs[String]
        row.getString(0) -> row.getInt(1) % numPartitions
      )
      .toMap

    assert(doc_id2physical_partition_id == expectedPhysicalPartitionId)
  }

}
