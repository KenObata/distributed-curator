package com.partitionAssignment

import org.apache.spark.HashPartitioner
import org.apache.spark.sql.{DataFrame, Row}

/**
 * Identity partitioner: uses partition_id directly as physical partition index.
 *
 * Spark's default DataFrame repartition applies pmod(Murmur3Hash(col), numPartitions), which scrambles well-distributed
 * logical partition IDs and causes physical partition collisions. This identity partitioner ensures 1:1 mapping from
 * logical to physical partition, eliminating physical partition skew.
 *
 * This Scala implementation avoids the Python pickle serialization overhead of the RDD-based Python approach (~5-10 min
 * at 2B rows). By staying on the JVM, the DataFrame → RDD → DataFrame round-trip uses Spark's internal serialization.
 *
 * Why not just use RDD HashPartitioner from Python?
 *   - RDD HashPartitioner with Int keys IS identity (Integer.hashCode() = value itself)
 *   - But going through PySpark RDD requires Python pickle serialization for each row
 *   - At 2B rows × ~520 bytes/row, that's ~1TB through Python pickle
 *   - This Scala version keeps everything on JVM: no pickle, no Python overhead
 *
 * Reference:
 *   - DataFrame repartition uses Murmur3: https://github.com/apache/spark/blob/master/sql/catalyst/src/main/scala/
 *     org/apache/spark/sql/catalyst/plans/physical/partitioning.scala
 *   - RDD HashPartitioner uses key.hashCode(): https://github.com/apache/spark/blob/master/core/src/main/scala/
 *     org/apache/spark/Partitioner.scala
 */

object IdentityRepartition {

  /**
   * Repartition a DataFrame using identity partitioning instead of hash partitioning.
   *
   * Stays entirely on JVM — no Python pickle serialization.
   *
   * @param df
   *   DataFrame to repartition
   * @param numPartitions
   *   Number of physical partitions
   * @param partitionColName
   *   Name of the partition_id column
   * @return
   *   Repartitioned DataFrame with 1:1 logical-to-physical mapping
   */
  def repartition(
    df: DataFrame,
    partitionColName: String,
    numPartitions: Int
  ): DataFrame = {
    val spark                  = df.sparkSession
    val schema                 = df.schema
    val partitionColIndex: Int = df.columns.indexOf(partitionColName)

    require(
      partitionColIndex >= 0,
      s"Column '$partitionColName' not found in DataFrame. Available: ${df.columns.mkString(", ")}"
    )

    // DataFrame → RDD[(partitionId, Row)] → partitionBy → RDD[Row] → DataFrame
    val rdd = df.rdd
      .map(row => (row.getInt(partitionColIndex), row))
      .partitionBy(new HashPartitioner(numPartitions))
      .map(tuple => tuple._2)

    spark.createDataFrame(rdd, schema)
  }

  /**
   * JVM entry point for PySpark consumers.
   *
   * Usage from PySpark: jvm_helper = spark._jvm.com.partitionAssignment.IdentityRepartition df_partitioned = DataFrame(
   * jvm_helper.repartitionFromPython(df_exploded._jdf, 27000, "partition_id"), spark )
   *
   * Python consumer function: def identity_repartition_scala(spark, df, num_partitions): jvm_helper =
   * spark._jvm.com.partitionAssignment.IdentityRepartition return DataFrame( jvm_helper.repartitionFromPython(df._jdf,
   * num_partitions, "partition_id"), spark )
   */
  def repartitionFromPython(
    jdf: org.apache.spark.sql.Dataset[Row],
    partitionColName: String,
    numPartitions: Int
  ): DataFrame = repartition(jdf.toDF(), partitionColName, numPartitions)

}
