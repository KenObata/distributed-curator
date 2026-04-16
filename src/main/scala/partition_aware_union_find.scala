package com.unionFind

import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.sql.types.{LongType, StringType, StructField, StructType}
import scala.collection.mutable
import org.apache.spark.util.LongAccumulator
import com.utils.Utils

/**
 * Partition-aware local Union-Find with path compression + union by rank. Runs inside mapPartitions — zero shuffle.
 *
 * Goal is to replace graphframe becaise it shuffles globally. in this scala, we run phase 1 within partition to map
 * local representative doc id in a pair. phase 2 is written in udf.py
 *
 * Usage from PySpark: this function runs per partition, not per row. This means we directly calls JVM method from
 * similar_pairs_df._jdf
 */
object PartitionAwareUnionFindUDF {
  // Data class for runPhase1LocalUnionFind's resultRDD.
  // Need python naming style for downstream pyspark
  case class LocalUnionFindSchema(doc_id: String, local_representative: String)

  private class UnionFind {
    /*
    Parent and Rank need to be under each executor to find localRepresentative
    So live under class, not Object (=singleton)
     */
    val parent: mutable.HashMap[String, String] = new mutable.HashMap[String, String]()
    val rank: mutable.HashMap[String, Int]      = new mutable.HashMap[String, Int]()

    def initialSetup(node: String): Unit = if (!parent.contains(node)) {
      parent(node) = node
      rank(node) = 0
    }

    def find(node: String): String = {
      var root = node
      while (parent(root) != root) root = parent(root)

      var pointer: String = node
      // Path compression: point every node on the path directly to root
      while (parent(pointer) != root) {
        val nextParent = parent(pointer)
        parent(pointer) = root
        pointer = nextParent
      }
      root
    }

    def union(doc1: String, doc2: String): Unit = {
      val root1: String = find(doc1)
      val root2: String = find(doc2)
      if (root1 != root2) { // else, return
        // Union by rank: attach shorter tree under taller tree
        if (rank(root1) < rank(root2)) {
          parent(root1) = root2
        } else if (rank(root2) < rank(root1)) {
          parent(root2) = root1
        } else {
          parent(root2) = root1
          rank(root1) = rank(root1) + 1
        }
      }
    }

  } // end of class UnionFind

  private class LongUnionFind {
    /*
    Unlike UnionFind, this class is encoded key value into Long value.
    This is more memory efficient.
    This class is called from phase2.
    Reason why this class is only callled from Phase2 is because
    phase 1 is not deduped yet. Convert from string to Long without dedupe
    degrades performanece.
     */
    val parent: mutable.HashMap[Long, Long] = new mutable.HashMap[Long, Long]()
    val rank: mutable.HashMap[Long, Int]    = new mutable.HashMap[Long, Int]()

    def initialSetup(node: Long): Unit = if (!parent.contains(node)) {
      parent(node) = node
      rank(node) = 0
    }

    def find(node: Long): Long = {
      var root = node
      while (parent(root) != root) root = parent(root)

      var pointer: Long = node
      // Path compression: point every node on the path directly to root
      while (parent(pointer) != root) {
        val nextParent = parent(pointer)
        parent(pointer) = root
        pointer = nextParent
      }
      root
    }

    def union(doc1: Long, doc2: Long): Unit = {
      val root1: Long = find(doc1)
      val root2: Long = find(doc2)
      if (root1 != root2) { // else, return
        // Union by rank: attach shorter tree under taller tree
        if (rank(root1) < rank(root2)) {
          parent(root1) = root2
        } else if (rank(root2) < rank(root1)) {
          parent(root2) = root1
        } else {
          parent(root2) = root1
          rank(root1) = rank(root1) + 1
        }
      }
    }

  } // end of class UnionFind

  def runPhase1LocalUnionFind(
    similarPairsDf: DataFrame
  ): (DataFrame, LongAccumulator) = {
    /*
    Phase 1: Partition-aware local Union-Find in each partition.
    In phase 1, union-find happens only within each partition so that there is no shuffle.

    Reads all pairs within each Spark partition, runs true Union-Find with
    path compression + union by rank, emits (doc_id, localRepresentative) per unique doc.

    Args:
    - similarPairsDf: (doc1, doc2, similarity, partition_id) from Step 3

    Returns:
    - (doc_id, localRepresentative) — one per unique doc per partition
      - only subset of input_df that exceeded similarity score are included.
     */
    val spark                      = similarPairsDf.sparkSession
    val sc                         = spark.sparkContext
    val pairCount: LongAccumulator = sc.longAccumulator("similar_pairs_count")

    val resultRDD = similarPairsDf.rdd.mapPartitions { iterator: Iterator[Row] =>
      /*
      Run partition-local Union-Find using shared UnionFind class.

          Args:
          - iterator: Iterator[Row] = (doc1, doc2, similarity, partition_id) from Step 3
            - this function receives each partition in a batch,
              that is why there is a for row in iterator loop.
              so this function is per partition, not per row.
       */
      val unionFindInstance: UnionFind = new UnionFind()

      // for replacing similar_pair_df.count()
      var localPairCount: Long = 0L

      for (row <- iterator) {
        localPairCount += 1
        val doc1 = row.getAs[String]("doc1")
        val doc2 = row.getAs[String]("doc2")
        unionFindInstance.initialSetup(doc1)
        unionFindInstance.initialSetup(doc2)

        unionFindInstance.union(doc1, doc2)
      }
      pairCount.add(localPairCount)

      /* unionFindInstance.parent is dict, so one row per unique doc_id (NOT one per pair)
         Note: avoid using ArrayBuffer collects everything into memory then converts to iterator
       */
      // use keysIterator over parent.key to avoid creating a new collection in memory
      unionFindInstance.parent.keysIterator.map { docId =>
        LocalUnionFindSchema(docId, unionFindInstance.find(docId))
      } // keysIterator.map returns Iterator
    }   // End of val resultRDD

    import spark.implicits._
    (resultRDD.toDF(), pairCount)
  } // End of def runPhase1LocalUnionFind()

  def runGlobalUnionFind(multipleRepsEdgesDf: DataFrame): DataFrame = {
    /*
    Phase 2: Global Union-Find in a single partition.
    Args:
    - DataFrame with (src: Long, dst: Long)
      - ex) For a doc with reps [A, B, C], emit edges: (A,B), (A,C)
            Note that docId is converted in Long.
    Retusn:
    - DataFrame with (node_id: Long, component_id: Long)
      - node_id is artificially generated.

    Usage from PySpark:
      jvm_helper = spark._jvm.com.unionFind.GlobalUnionFind
      result_jdf = jvm_helper.runGlobalUnionFind(multipleRepsEdgesDf._jdf)
      result = DataFrame(result_jdf, spark)
     */
    val spark = multipleRepsEdgesDf.sparkSession

    val resultSchema = StructType(
      Seq(
        StructField("node_id", LongType, nullable = false),
        StructField("component_id", LongType, nullable = false)
      )
    )
    val resultRDD = multipleRepsEdgesDf.coalesce(1).rdd.mapPartitions { iterator =>
      System.gc()
      Thread.sleep(100)
      Utils.plotHeapMemory(label = "Before_global_UnionFind")
      val uf = new LongUnionFind()
      for (row <- iterator) {
        val src = row.getLong(0)
        val dst = row.getLong(1)
        uf.initialSetup(src)
        uf.initialSetup(dst)
        uf.union(src, dst)
      }

      System.gc()
      Thread.sleep(100)
      Utils.plotHeapMemory(label = "After_global_UnionFind")

      uf.parent.keysIterator.map { node =>
        Row(node, uf.find(node))
      }
    } // end of resultRDD
    spark.createDataFrame(resultRDD, resultSchema)
  } // enf of def runGlobalUnionFind()

}
