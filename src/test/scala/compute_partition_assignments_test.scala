package com.partitionAssignment

import org.scalatest.funsuite.AnyFunSuite
import scala.util.Random
import scala.util.hashing.MurmurHash3

class ComputePartitionAssignmentsTest extends AnyFunSuite {
  /* ScalaTest needs class because the test runner creates a fresh instance per test
       because each test should be independent
   */
  val numHashes: Int = 64

  /* =========================================================================
     Output schema & length
   ========================================================================= */
  test("test_output_schemaand_length") {
    /* Return type should be return type is (Array[Int], Array[Int])
     */
    val signature: Array[Long] = Array.fill(numHashes)(42L)
    val numBands               = 8
    val rowsPerBand            = 8
    val (partitions, bandHashes) =
      ComputePartitionAssignmentsUDF.ComputePartitionAssignmentsLogic(signature, numBands, rowsPerBand, 1000)
    assert(partitions.isInstanceOf[Array[Int]])
    assert(bandHashes.isInstanceOf[Array[Int]])

    assert(partitions.length == numBands)
    assert(bandHashes.length == numBands)
  }

  /* =========================================================================
     Edge case input
   ========================================================================= */
  test("null signature returns Array(0)") {
    val (partitions, bandHashes) = ComputePartitionAssignmentsUDF.ComputePartitionAssignmentsLogic(null, 8, 8, 1000)
    assert(partitions sameElements Array(0))
    assert(bandHashes sameElements Array(0))
  }

  test("empty signature returns Array(0)") {
    val signature: Array[Long] = Array.empty[Long]
    val (partitions, bandHashes) =
      ComputePartitionAssignmentsUDF.ComputePartitionAssignmentsLogic(signature, 8, 8, 1000)
    assert(partitions sameElements Array(0))
    assert(bandHashes sameElements Array(0))
  }

  test("all-zero signature returns Array(0)") {
    val signature: Array[Long] = Array.fill(numHashes)(0L)
    val (partitions, bandHashes) =
      ComputePartitionAssignmentsUDF.ComputePartitionAssignmentsLogic(signature, 8, 8, 1000)
    assert(partitions sameElements Array(0))
    assert(bandHashes sameElements Array(0))
  }

  /* =========================================================================
   Partition ID correctness: abs(bandHash) % numPartitions
  ======================================================================== */
  test("each partition_id equals abs(bandHash) % numPartitions and within range of numPartitions") {
    val random                 = new Random(42)
    val signature: Array[Long] = Array.fill(numHashes)(random.nextLong() & 0xffffffffL)
    val numPartitions: Int     = 1000
    val (partitions, bandHashes) =
      ComputePartitionAssignmentsUDF.ComputePartitionAssignmentsLogic(null, 8, 8, numPartitions)
    for ((partition, bandHash) <- partitions.zip(bandHashes)) {
      assert(partition == math.abs(bandHash) % numPartitions)
      assert(0 <= partition && partition < numPartitions)
    }
  }

  test("bandHashes match MurmurHash3.arrayHash of band slices") {
    val random                 = new Random(42).nextLong()
    val signature: Array[Long] = Array.fill(numHashes)(random & 0xffffffffL)
    val rowsPerBand            = 8
    val (_, bandHashes) =
      ComputePartitionAssignmentsUDF.ComputePartitionAssignmentsLogic(signature, 8, rowsPerBand, 1000)

    for (bandId <- 0 until 8) {
      val start        = bandId * rowsPerBand
      val end          = math.min(start + rowsPerBand, signature.length)
      val expectedHash = MurmurHash3.arrayHash(signature.slice(start, end))
      assert(bandHashes(bandId) == expectedHash, s"band $bandId: expected $expectedHash, got ${bandHashes(bandId)}")
    }
  }

  /* =========================================================================
     Idempotency check
   ========================================================================= */

  test("deterministic - same signature always returns same partitions") {
    val random_val = new Random(42).nextLong()
    /* reminder - random nextLong() is signed but mh3 is unsigned32 bit Long.
          we don't want negative values from random nextLong().
          so use & opetration to make it positive.
     */
    val signature: Array[Long] = Array.fill(numHashes)(random_val & 0xffffffffL)
    val (partitions1, bandHashes1) =
      ComputePartitionAssignmentsUDF.ComputePartitionAssignmentsLogic(signature, 8, 8, 1000)
    val (partitions2, bandHashes2) =
      ComputePartitionAssignmentsUDF.ComputePartitionAssignmentsLogic(signature, 8, 8, 1000)

    for ((partition1, partition2) <- partitions1.zip(partitions2)) assert(partition1 == partition2)
    for ((bandHash1, bandHash2) <- bandHashes1.zip(bandHashes2)) assert(bandHash1 == bandHash2)
  }

  test("identical signatures get same partitions") {
    val random_val              = new Random(42).nextLong()
    val signature1: Array[Long] = Array.fill(numHashes)(random_val & 0xffffffffL)
    val signature2: Array[Long] = signature1.clone()
    val (partitions1, bandHashes1) =
      ComputePartitionAssignmentsUDF.ComputePartitionAssignmentsLogic(signature1, 8, 8, 1000)
    val (partitions2, bandHashes2) =
      ComputePartitionAssignmentsUDF.ComputePartitionAssignmentsLogic(signature2, 8, 8, 1000)
    for ((partition1, partition2) <- partitions1.zip(partitions2)) assert(partition1 == partition2)
    for ((bandHash1, bandHash2) <- bandHashes1.zip(bandHashes2)) assert(bandHash1 == bandHash2)
  }

}
