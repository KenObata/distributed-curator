package com.processPartitionLocally

import org.scalatest.funsuite.AnyFunSuite
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.types.ArrayType
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.types.StringType
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.LongType // used in ArrayType(LongType)
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.Dataset
import org.apache.spark.rdd.RDD
import scala.util.Random
import java.lang.reflect.Method

class ProcessPartitionLocallyTest extends AnyFunSuite {

  // computed once and cached
  lazy val spark: SparkSession = SparkSession
    .builder()
    .master("local[*]")
    .appName("ProcessPartitionLocallyTest")
    .getOrCreate()

  val fields: Seq[StructField] = Seq(
    StructField("doc_id", StringType),
    StructField("minhash_signature", ArrayType(LongType)),
    StructField("partition_id", IntegerType)
  )

  val schema: StructType = StructType(fields)

  val numHashes: Int = 64

  test("empty df returns empty result") {

    val rowRDD: RDD[Row]   = spark.sparkContext.emptyRDD[Row]
    val emptyDf: DataFrame = spark.createDataFrame(rowRDD, schema)

    val result: DataFrame = ProcessPartitionLocallyUDF.processPartitions(
      emptyDf,
      8,
      8,
      0.9
    )
    assert(result.count() == 0)
  }

  test("identical documents found as similar pair") {
    import spark.implicits._ // for as[]
    val randomVal              = new Random(42).nextLong()
    val signature: Array[Long] = Array.fill(numHashes)(randomVal & 0xffffffffL)

    val rowRDD: RDD[Row] = spark.sparkContext.parallelize(
      Seq(
        Row("doc1", signature, 0),
        Row("doc2", signature, 0) // identical signature
      )
    )

    // .repartition(1) to mock Step2 ComputePartitionAssignmentsLogic
    // so that two docs are in the same partition
    val inputDf: DataFrame  = spark.createDataFrame(rowRDD, schema).repartition(1)
    val resultDf: DataFrame = ProcessPartitionLocallyUDF.processPartitions(inputDf, 8, 8, 0.9)
    val resultDataset: Dataset[ProcessPartitionLocallyUDF.SimilarityPairs] =
      resultDf.as[ProcessPartitionLocallyUDF.SimilarityPairs]
    val resultArray: Array[ProcessPartitionLocallyUDF.SimilarityPairs] = resultDataset.collect() // to access each index

    assert(resultArray(0).doc1 == "doc1")
    assert(resultArray(0).doc2 == "doc2")
    assert(resultArray(0).similarity == 1.0)
  }

  test("completely different documents no similar pair") {
    import spark.implicits._ // for as[]
    val randomVal1              = new Random(42).nextLong()
    val randomVal2              = new Random(43).nextLong()
    val signature1: Array[Long] = Array.fill(numHashes)(randomVal1 & 0xffffffffL)
    val signature2: Array[Long] = Array.fill(numHashes)(randomVal2 & 0xffffffffL)

    val rowRDD: RDD[Row] = spark.sparkContext.parallelize(
      Seq(
        Row("doc1", signature1, 0),
        Row("doc2", signature2, 0)
      )
    )

    // .repartition(1) to mock Step2 ComputePartitionAssignmentsLogic
    // so that two docs are in the same partition
    val inputDf: DataFrame  = spark.createDataFrame(rowRDD, schema).repartition(1)
    val resultDf: DataFrame = ProcessPartitionLocallyUDF.processPartitions(inputDf, 8, 8, 0.9)

    assert(resultDf.isEmpty)
  }

  test("ordering doc1 < doc2 test") {
    import spark.implicits._ // for as[]
    val randomVal              = new Random(42).nextLong()
    val signature: Array[Long] = Array.fill(numHashes)(randomVal & 0xffffffffL)

    val rowRDD: RDD[Row] = spark.sparkContext.parallelize(
      Seq(
        Row("zzz_doc", signature, 0),
        Row("aaa_doc", signature, 0) // identical signature
      )
    )

    // .repartition(1) to mock Step2 ComputePartitionAssignmentsLogic
    // so that two docs are in the same partition
    val inputDf: DataFrame  = spark.createDataFrame(rowRDD, schema).repartition(1)
    val resultDf: DataFrame = ProcessPartitionLocallyUDF.processPartitions(inputDf, 8, 8, 0.9)
    assert(!resultDf.isEmpty)

    val resultDataset: Dataset[ProcessPartitionLocallyUDF.SimilarityPairs] =
      resultDf.as[ProcessPartitionLocallyUDF.SimilarityPairs]
    val resultArray: Array[ProcessPartitionLocallyUDF.SimilarityPairs] = resultDataset.collect() // to access each index

    // alphabetically sorted in ASC
    assert(resultArray(0).doc1 == "aaa_doc")
    assert(resultArray(0).doc2 == "zzz_doc")
  }

  test("similarity threshold filters correctly") {
    import spark.implicits._ // for as[]
    // Create two signatures that share ~50% of values
    val signature1 = Array.tabulate(64)(i => i.toLong)
    val signature2 = Array.tabulate(64)(i => if (i < 32) i.toLong else (i + 1000).toLong)

    val rowRDD: RDD[Row] = spark.sparkContext.parallelize(
      Seq(
        Row("doc1", signature1, 0),
        Row("doc2", signature2, 0) // identical signature
      )
    )
    val inputDf: DataFrame         = spark.createDataFrame(rowRDD, schema).repartition(1)
    val highThresholdDf: DataFrame = ProcessPartitionLocallyUDF.processPartitions(inputDf, 8, 8, 0.9)
    assert(highThresholdDf.isEmpty)

    val lowThresholdDf: DataFrame = ProcessPartitionLocallyUDF.processPartitions(inputDf, 8, 8, 0.3)
    assert(!lowThresholdDf.isEmpty)
    val lowThresholdDataset: Dataset[ProcessPartitionLocallyUDF.SimilarityPairs] =
      lowThresholdDf.as[ProcessPartitionLocallyUDF.SimilarityPairs]
    val lowThresholdArray: Array[ProcessPartitionLocallyUDF.SimilarityPairs] =
      lowThresholdDataset.collect() // to access each index
    assert(lowThresholdArray(0).similarity > 0.3)
  }

  test("estimateSimilarity correctness") {
    /* Because estimateSimilarity is a private method, we need to talk to JVM metadata
           to bypass compiler validation.

           classOf[ProcessPartitionLocallyUDF.type]
            access below metadata:
              -   ProcessPartitionLocallyUDF:
                -     methods:
                  - processPartitions (public)
                  - estimateSimilarity (private)
     */
    val estimateSimilarityMethod: Method = ProcessPartitionLocallyUDF.getClass.getDeclaredMethod(
      "estimateSimilarity",
      classOf[Array[Long]],
      classOf[Array[Long]]
    )
    estimateSimilarityMethod.setAccessible(true)

    //                             (object or class name, args)
    val identicalResult: Double = estimateSimilarityMethod
      .invoke(ProcessPartitionLocallyUDF, Array(1L, 2L), Array(1L, 2L))
      .asInstanceOf[Double]

    assert(identicalResult == 1.0)

    val halfMatchResult: Double = estimateSimilarityMethod
      .invoke(ProcessPartitionLocallyUDF, Array(1L, 2L), Array(1L, 4L))
      .asInstanceOf[Double]

    assert(halfMatchResult == 0.5)

    val noMatchResult: Double = estimateSimilarityMethod
      .invoke(ProcessPartitionLocallyUDF, Array(1L, 2L), Array(3L, 4L))
      .asInstanceOf[Double]

    assert(noMatchResult == 0.0)
  }

}
