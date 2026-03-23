package com.partitionAssignment

import org.scalatest.funsuite.AnyFunSuite
import scala.util.Random

class ComputePartitionAssignmentsTest extends AnyFunSuite {
    /* ScalaTest needs class because the test runner creates a fresh instance per test
       because each test should be independent
    */
    val numHashes:Int = 64

    test("null signature returns Array(0)") {
        val result = ComputePartitionAssignmentsUDF.ComputePartitionAssignmentsLogic(null, 8, 8, 1000)
        assert(result sameElements Array(0))
    }

    test("empty signature returns Array(0)") {
        val signature: Array[Long] = Array.empty[Long]
        val result = ComputePartitionAssignmentsUDF.ComputePartitionAssignmentsLogic(signature, 8, 8, 1000)
        assert(result sameElements Array(0))
    }

    test("all-zero signature returns Array(0)") {
        val signature: Array[Long] = Array.fill(numHashes)(0L)
        val result = ComputePartitionAssignmentsUDF.ComputePartitionAssignmentsLogic(signature, 8, 8, 1000)
        assert(result sameElements Array(0))
    }

    test("deterministic - same signature always returns same partitions") {
        val random_val = new Random(42).nextLong()
        /* reminder - random nextLong() is signed but mh3 is unsigned32 bit Long. 
          we don't want negative values from random nextLong(). 
          so use & opetration to make it positive.
        */
        val signature: Array[Long] = Array.fill(numHashes)(random_val & 0xFFFFFFFFL)
        val result1 = ComputePartitionAssignmentsUDF.ComputePartitionAssignmentsLogic(signature, 8, 8, 1000)
        val result2 = ComputePartitionAssignmentsUDF.ComputePartitionAssignmentsLogic(signature, 8, 8, 1000)
        assert(result1 sameElements result2)
    }

    test("identical signatures get same partitions") {
        val random_val = new Random(42).nextLong()
        val signature1: Array[Long] = Array.fill(numHashes)(random_val & 0xFFFFFFFFL)
        val signature2: Array[Long] = signature1.clone()
        val result1 = ComputePartitionAssignmentsUDF.ComputePartitionAssignmentsLogic(signature1, 8, 8, 1000)
        val result2 = ComputePartitionAssignmentsUDF.ComputePartitionAssignmentsLogic(signature2, 8, 8, 1000)
        assert(result1 sameElements result2)
    }
}