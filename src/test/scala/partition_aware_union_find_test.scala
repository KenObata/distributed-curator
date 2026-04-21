package com.unionFind

import org.scalatest.funsuite.AnyFunSuite
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.sql.types.{DoubleType, IntegerType, LongType, StringType, StructField, StructType}
import org.apache.spark.sql.Encoders
import scala.collection.mutable
import org.apache.spark.util.LongAccumulator
import scala.Tuple2

class PartitionAwareUnionFindUDFTest extends AnyFunSuite {

// computed once and cached
  lazy val spark: SparkSession = SparkSession
    .builder()
    .master("local[*]")
    .appName("PartitionAwareUnionFindUDFTest")
    .getOrCreate()

  import spark.implicits._ // for .as[T] needs to convert DataFrame → Dataset[T].

  val inputFields: Seq[StructField] = Seq(
    StructField("doc1", StringType),
    StructField("doc2", StringType),
    StructField("similarity", DoubleType),
    StructField("partition_id", IntegerType)
  )

  val inputSchema: StructType = StructType(inputFields)

  test("test_output_schema_and_length") {
    val rowRDD: RDD[Row] = spark.sparkContext.parallelize(
      Seq(
        Row("A", "B", 0.9, 0)
      )
    )

    val similarPairsDf: DataFrame                  = spark.createDataFrame(rowRDD, inputSchema).repartition(1)
    val result: Tuple2[DataFrame, LongAccumulator] = PartitionAwareUnionFindUDF.runPhase1LocalUnionFind(similarPairsDf)
    val resultDf: DataFrame                        = result._1
    val pairCountAccumulator: LongAccumulator      = result._2

    val resultDataset: Dataset[PartitionAwareUnionFindUDF.LocalUnionFindSchema] =
      resultDf.as[PartitionAwareUnionFindUDF.LocalUnionFindSchema]
    val resultArray: Array[PartitionAwareUnionFindUDF.LocalUnionFindSchema] =
      resultDataset.collect() // to access each index

    val expectedColumns = Encoders.product[PartitionAwareUnionFindUDF.LocalUnionFindSchema].schema.fieldNames.toSet
    assert(resultDf.columns.toSet == expectedColumns)
    assert(resultArray.length == 2)
  }

  test("test_single_pair_should_point_same_local_representative") {
    /*
    One pair in one partition => both docs should map to same representative.

        Input:  (docA, docB) in partition 0
        Expect: docA => docA, docB => docA  (because UnionFind picks first arg)
     */
    val rowRDD: RDD[Row] = spark.sparkContext.parallelize(
      Seq(
        Row("A", "B", 0.9, 0)
      )
    )

    val similarPairsDf: DataFrame                  = spark.createDataFrame(rowRDD, inputSchema).repartition(1)
    val result: Tuple2[DataFrame, LongAccumulator] = PartitionAwareUnionFindUDF.runPhase1LocalUnionFind(similarPairsDf)
    val resultDf: DataFrame                        = result._1
    val pairCountAccumulator: LongAccumulator      = result._2
    val resultDataset: Dataset[PartitionAwareUnionFindUDF.LocalUnionFindSchema] =
      resultDf.as[PartitionAwareUnionFindUDF.LocalUnionFindSchema]
    val resultArray: Array[PartitionAwareUnionFindUDF.LocalUnionFindSchema] =
      resultDataset.collect() // to access each index

    // Both docs should have the same local_representative
    assert(resultArray(0).local_representative == resultArray(1).local_representative)
  }

  test("test_multiple_and_transitive_pairs_should_point_same_local_representative") {
    /*
    (A,B), (B,C) in same partition => all three map to same representative.
        Because union(B, C) doesn't union B and C — it unions find(B) and find(C), which is union(A, C)

        Input:  (A,B), (B,C) in partition 0
        Expect: A, B, C => same representative
     */
    val rowRDD: RDD[Row] = spark.sparkContext.parallelize(
      Seq(
        Row("A", "B", 0.9, 0),
        Row("B", "C", 0.9, 0)
      )
    )

    val similarPairsDf: DataFrame                  = spark.createDataFrame(rowRDD, inputSchema).repartition(1)
    val result: Tuple2[DataFrame, LongAccumulator] = PartitionAwareUnionFindUDF.runPhase1LocalUnionFind(similarPairsDf)
    val resultDf: DataFrame                        = result._1
    val pairCountAccumulator: LongAccumulator      = result._2
    val resultDataset: Dataset[PartitionAwareUnionFindUDF.LocalUnionFindSchema] =
      resultDf.as[PartitionAwareUnionFindUDF.LocalUnionFindSchema]
    val resultArray: Array[PartitionAwareUnionFindUDF.LocalUnionFindSchema] = resultDataset.collect()
    val resultMap: Map[String, String] = resultArray.map(r => r.doc_id -> r.local_representative).toMap

    assert(resultArray.length == 3)
    assert(resultMap("A") == resultMap("B"))
    assert(resultMap("A") == resultMap("C"))
  }

  test("test_two_disconnected_pairs_in_same_partition") {
    /*
    Two independent clusters in one partition => two different representatives.

        Input:  (A,B), (C,D) in partition 0
        Expect: A,B => same rep; C,D => same rep; two localRepresentative are different
     */
    val rowRDD: RDD[Row] = spark.sparkContext.parallelize(
      Seq(
        Row("A", "B", 0.9, 0),
        Row("C", "D", 0.9, 0)
      )
    )

    val similarPairsDf: DataFrame                  = spark.createDataFrame(rowRDD, inputSchema).repartition(1)
    val result: Tuple2[DataFrame, LongAccumulator] = PartitionAwareUnionFindUDF.runPhase1LocalUnionFind(similarPairsDf)
    val resultDf: DataFrame                        = result._1
    val pairCountAccumulator: LongAccumulator      = result._2
    val resultDataset: Dataset[PartitionAwareUnionFindUDF.LocalUnionFindSchema] =
      resultDf.as[PartitionAwareUnionFindUDF.LocalUnionFindSchema]
    val resultArray: Array[PartitionAwareUnionFindUDF.LocalUnionFindSchema] = resultDataset.collect()
    val resultMap: Map[String, String] = resultArray.map(r => r.doc_id -> r.local_representative).toMap

    assert(resultMap.size == 4)
    assert(resultMap("A") == resultMap("B"))
    assert(resultMap("C") == resultMap("D"))
    assert(resultMap("A") != resultMap("C"))

  }

  test("test_two_partitions_independent_diff_partition") {
    /*
    Different pairs in different partitions => each partition runs UF independently.

        Partition 0: (A,B)
        Partition 1: (C,D)

        Phase 2 merges them.
     */
    val rowRDD: RDD[Row] = spark.sparkContext.parallelize(
      Seq(
        Row("A", "B", 0.9, 0),
        Row("C", "D", 0.9, 1)
      )
    )

    val similarPairsDf: DataFrame = spark.createDataFrame(rowRDD, inputSchema).repartition(2, col("partition_id"))
    val result: Tuple2[DataFrame, LongAccumulator] = PartitionAwareUnionFindUDF.runPhase1LocalUnionFind(similarPairsDf)
    val resultDf: DataFrame                        = result._1
    val pairCountAccumulator: LongAccumulator      = result._2
    val resultDataset: Dataset[PartitionAwareUnionFindUDF.LocalUnionFindSchema] =
      resultDf.as[PartitionAwareUnionFindUDF.LocalUnionFindSchema]
    val resultArray: Array[PartitionAwareUnionFindUDF.LocalUnionFindSchema] = resultDataset.collect()

    val resultMap: Map[String, String] = resultArray.map(row => row.doc_id -> row.local_representative).toMap

    assert(resultMap.size == 4)
    assert(resultMap("A") == resultMap("B"))
    assert(resultMap("C") == resultMap("D"))
    assert(resultMap("A") != resultMap("C"))

  }

  test("test_duplicate_pairs_same_partition") {
    /* Same pair appears twice in same partition => UF handles gracefully.
        Should still produce one entry per unique doc.
     */
    val rowRDD: RDD[Row] = spark.sparkContext.parallelize(
      Seq(
        Row("A", "B", 0.9, 0),
        Row("A", "B", 0.9, 0)
      )
    )
    val similarPairsDf: DataFrame                  = spark.createDataFrame(rowRDD, inputSchema).repartition(1)
    val result: Tuple2[DataFrame, LongAccumulator] = PartitionAwareUnionFindUDF.runPhase1LocalUnionFind(similarPairsDf)
    val resultDf: DataFrame                        = result._1
    val pairCountAccumulator: LongAccumulator      = result._2
    val resultDataset: Dataset[PartitionAwareUnionFindUDF.LocalUnionFindSchema] =
      resultDf.as[PartitionAwareUnionFindUDF.LocalUnionFindSchema]
    val resultArray: Array[PartitionAwareUnionFindUDF.LocalUnionFindSchema] = resultDataset.collect()

    val resultMap: Map[String, String] = resultArray.map(row => row.doc_id -> row.local_representative).toMap

    assert(resultMap.size == 2)
    assert(resultMap("A") == resultMap("B"))
  }

  test("test_same_doc_pair_appeared_different_partitions_gets_multiple_rows") {
    /*
    Doc A appears in pairs in partition 0 AND partition 1.
    If Spark places them in different partitions, A appears twice.
     */
    val rowRDD: RDD[Row] = spark.sparkContext.parallelize(
      Seq(
        Row("A", "B", 0.9, 0),
        Row("A", "C", 0.9, 1)
      )
    )
    val similarPairsDf: DataFrame = spark.createDataFrame(rowRDD, inputSchema).repartition(2, col("partition_id"))
    val result: Tuple2[DataFrame, LongAccumulator] = PartitionAwareUnionFindUDF.runPhase1LocalUnionFind(similarPairsDf)
    val resultDf: DataFrame                        = result._1
    val pairCountAccumulator: LongAccumulator      = result._2
    val resultDataset: Dataset[PartitionAwareUnionFindUDF.LocalUnionFindSchema] =
      resultDf.as[PartitionAwareUnionFindUDF.LocalUnionFindSchema]
    val resultArray: Array[PartitionAwareUnionFindUDF.LocalUnionFindSchema] = resultDataset.collect()
    val rowA: Array[PartitionAwareUnionFindUDF.LocalUnionFindSchema]        = resultArray.filter(_.doc_id == "A")

    assert(rowA.size >= 1)

    val resultDocIdSet: Set[String] = resultArray.map(row => row.doc_id).toSet
    assert(resultDocIdSet == Set("A", "B", "C"))
  }

  test("test_accumulator_counts_pairs") {
    /*
      partition 0 has 2 rows, partition 1 has 1 row.
      so pairCountAccumulator should be 3.
     */
    val rowRDD: RDD[Row] = spark.sparkContext.parallelize(
      Seq(
        Row("A", "B", 0.9, 0),
        Row("B", "C", 0.9, 0),
        Row("C", "D", 0.9, 1)
      )
    )
    val similarPairsDf: DataFrame = spark.createDataFrame(rowRDD, inputSchema).repartition(2, col("partition_id"))
    val result: Tuple2[DataFrame, LongAccumulator] = PartitionAwareUnionFindUDF.runPhase1LocalUnionFind(similarPairsDf)
    val resultDf: DataFrame                        = result._1
    val pairCountAccumulator: LongAccumulator      = result._2

    resultDf.count() // trigger mapPartitions
    assert(pairCountAccumulator.value == 3)
  }

}

class GlobalUnionFindUDFTest extends AnyFunSuite {

  // ===========================================================
  //  Phase2 Global Union Find test
  // ===========================================================
  // computed once and cached
  lazy val spark: SparkSession = SparkSession
    .builder()
    .master("local[*]")
    .appName("GlobalUnionFindUDFTest")
    .getOrCreate()

  import spark.implicits._ // for .as[T] needs to convert DataFrame → Dataset[T].

  val inputFields: Seq[StructField] = Seq(
    StructField("src", LongType),
    StructField("dst", LongType)
  )

  val outputFields: Seq[StructField] = Seq(
    StructField("node_id", LongType, nullable = false),
    StructField("component_id", LongType, nullable = false)
  )

  val inputSchema: StructType  = StructType(inputFields)
  val outputSchema: StructType = StructType(outputFields)

  test("test_output_schema_and_length") {
    val rowRDD: RDD[Row] = spark.sparkContext.parallelize(
      Seq(
        Row(1L, 2L)
      )
    )

    val multipleRepsEdgesDf: DataFrame = spark.createDataFrame(rowRDD, inputSchema).repartition(1)
    val resultDf: DataFrame            = PartitionAwareUnionFindUDF.runGlobalUnionFind(multipleRepsEdgesDf)

    assert(resultDf.schema == outputSchema)
    assert(resultDf.count() == 2)
  }

  test("test_path_compression") {
    /*
      (1,2), (2,3) => all three map to same representative.
     */
    val rowRDD: RDD[Row] = spark.sparkContext.parallelize(
      Seq(
        Row(1L, 2L),
        Row(2L, 3L)
      )
    )

    val multipleRepsEdgesDf: DataFrame = spark.createDataFrame(rowRDD, inputSchema).repartition(1)
    val resultDf: DataFrame            = PartitionAwareUnionFindUDF.runGlobalUnionFind(multipleRepsEdgesDf)

    val resultMap: Map[Long, Long] =
      resultDf.collect().map(row => row.getAs[Long]("node_id") -> row.getAs[Long]("component_id")).toMap

    assert(resultDf.count() == 3)
    assert(resultMap(1) == resultMap(2))
    assert(resultMap(2) == resultMap(3))
  }

  test("test_independent_pairs") {
    val rowRDD: RDD[Row] = spark.sparkContext.parallelize(
      Seq(
        Row(1L, 2L),
        Row(4L, 5L)
      )
    )

    val multipleRepsEdgesDf: DataFrame = spark.createDataFrame(rowRDD, inputSchema).repartition(1)
    val resultDf: DataFrame            = PartitionAwareUnionFindUDF.runGlobalUnionFind(multipleRepsEdgesDf)

    val resultMap: Map[Long, Long] =
      resultDf.collect().map(row => row.getAs[Long]("node_id") -> row.getAs[Long]("component_id")).toMap

    assert(resultDf.count() == 4)
    assert(resultMap(1) == resultMap(2))
    assert(resultMap(4) == resultMap(5))
    assert(resultMap(1) != resultMap(4))
  }

  test("test_star_topology") {
    // One node connected to many — all resolve to same component
    val rowRDD: RDD[Row] = spark.sparkContext.parallelize(
      Seq(
        Row(1L, 2L),
        Row(1L, 3L),
        Row(1L, 4L)
      )
    )
    val multipleRepsEdgesDf: DataFrame = spark.createDataFrame(rowRDD, inputSchema).repartition(1)
    val resultDf: DataFrame            = PartitionAwareUnionFindUDF.runGlobalUnionFind(multipleRepsEdgesDf)

    val resultMap: Map[Long, Long] =
      resultDf.collect().map(row => row.getAs[Long]("node_id") -> row.getAs[Long]("component_id")).toMap

    assert(resultDf.count() == 4)
    assert(resultMap(1L) == resultMap(2L))
    assert(resultMap(1L) == resultMap(3L))
    assert(resultMap(1L) == resultMap(4L))
  }

  test("test_duplicate_edges_idempotent") {
    /* Same edge twice should produce same result as once
       Because parent HashMap deduped key
     */
    val rowRDD: RDD[Row] = spark.sparkContext.parallelize(
      Seq(
        Row(1L, 2L),
        Row(1L, 2L),
        Row(2L, 3L),
        Row(2L, 3L)
      )
    )
    val multipleRepsEdgesDf: DataFrame = spark.createDataFrame(rowRDD, inputSchema).repartition(1)
    val resultDf: DataFrame            = PartitionAwareUnionFindUDF.runGlobalUnionFind(multipleRepsEdgesDf)

    val resultMap: Map[Long, Long] =
      resultDf.collect().map(row => row.getAs[Long]("node_id") -> row.getAs[Long]("component_id")).toMap

    assert(resultDf.count() == 3) // 3 unique nodes, not 4
    assert(resultMap(1L) == resultMap(2L))
    assert(resultMap(2L) == resultMap(3L))
  }

  test("test_output_has_single_partition") {
    /*
      runGlobalUnionFind coalesce(1) inside.
     */
    val rowRDD: RDD[Row] = spark.sparkContext.parallelize(
      Seq(
        Row(1L, 2L),
        Row(3L, 4L)
      )
    )
    val multipleRepsEdgesDf: DataFrame = spark.createDataFrame(rowRDD, inputSchema).repartition(10)
    val resultDf: DataFrame            = PartitionAwareUnionFindUDF.runGlobalUnionFind(multipleRepsEdgesDf)

    assert(resultDf.rdd.getNumPartitions == 1)
  }

  test("test_driver_matches_executor") {
    val rowRDD: RDD[Row] = spark.sparkContext.parallelize(
      Seq(
        Row(1L, 2L),
        Row(2L, 3L),
        Row(4L, 5L)
      )
    )
    val multipleRepsEdgesDf: DataFrame = spark.createDataFrame(rowRDD, inputSchema).repartition(1)

    val executorResult: DataFrame = PartitionAwareUnionFindUDF.runGlobalUnionFind(multipleRepsEdgesDf)
    val driverResult: DataFrame   = PartitionAwareUnionFindUDF.runGlobalUnionFindFromDriver(multipleRepsEdgesDf)

    val executorMap: Map[Long, Long] =
      executorResult.collect().map(row => row.getAs[Long]("node_id") -> row.getAs[Long]("component_id")).toMap
    val driverMap: Map[Long, Long] =
      driverResult.collect().map(row => row.getAs[Long]("node_id") -> row.getAs[Long]("component_id")).toMap

    // Same node count
    assert(executorMap.size == driverMap.size)

    // Same component groupings
    assert(driverMap(1L) == driverMap(2L))
    assert(driverMap(2L) == driverMap(3L))
    assert(driverMap(4L) == driverMap(5L))
    assert(driverMap(1L) != driverMap(4L))
  }

}
