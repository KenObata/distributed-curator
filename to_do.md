
- Critical Optimizations for Real Scale
    - Priority 1: Proper Connected Components for Duplicate Clusters
        - DONE
    - Week 6-7: Memory and Performance Optimizations
      Priority 3: Signature Compression and Efficient Storage
        - won't do. we found a problem. if we sort and store delta between
          two min hashes within a doc, it loses ordering. 
          Reminder that the ordering matters because each 128 min hash is based on different seed.
    - complete EMR run
        - DONE
    - Implement hash mixing optimization to 
      eliminate string concatenation in MinHash
    - cache edge df (won't do)
    - Eliminate distinct
    - Use more efficient GraphFrames settings (won't do)
        # Add before connectedComponents()
        spark.conf.set("spark.graphx.pregel.checkpointInterval", "10")
        components = g.connectedComponents(algorithm="graphx")  
    - Implement caching mechanism
    - Test 1000 WET files with caching and num_partition= core_count    
    - I implemented skipping boiler template copy such as lisense.
      ```if band_size > MAX_BAND_SIZE:``` - just re-run.
    - Implement incremental union-find for
      faster group building
      - check if processPartitionLocally needs to be fixed for .map to remove arrayBuffer
          - DONE. Won't do improvement is for memory, not latency,
           and we lose logging maxBandSeen if we go with lazy flatMap because maxBandSeen is within a partition.
    - missing one test scala file for MinHash UDF
        - Expected because it's Cython.

    ☐ Increase similarity threshold to reduce
      false positives
    - scala UDF
    - Unit test for scala UDF
    - test 1000 partition count with python MinHash UDF for Found xxx similar pair.
      - DONE. simiar pair is more impacted by spark partition count rather than hash algo.
    - test 100 WET file with Cython UDF.
      - DONE. 4 min in Step1. 15 min E2E.
    - Setup mandatory pre-commit script for pytest and sbt test
    - optimize fetching WET files?
    - 9k WET test after following optimization:
     Step4 in scala (prev. 1 hour), Step5 two phase UnionFind (prev. 1 hour).
    - Optimize Step4: dropDlicates
      - Explore deduplicating inside Step 4's Scala mapPartitions across partitions
      - Actually this won't help because we need global dedupe.
    - Optimize Step3: implement deterministic partitioning
     (identity partitioning) instead of hash partitioning. Include unit tests.
    - Reduce shuffle partitions for Phase 2 only 
      - defer now that we use single pass global union find.
    - Disable Spark UI storage 
      spark.ui.retainedStages=50 and spark.ui.retainedTasks=1000 caps the UI memory footprint
      - won't do
    - Local checkpoint instead of HDFS
     — localCheckpoint() writes to executor disk, avoids HDFS overhead, and the driver doesn't track HDFS file metadata. Less durable but Phase 2 iterations are short enough that re-execution on failure is cheap
     - defer now that we use single pass global union find. 
    - Scala IdentityPartitioner — eliminates the 6.6 min Python pickle step, which also reduces driver-side serialization metadata
    - Distributed Union-Find
     — at 90K, replace the iterative SQL label propagation entirely with an RDD-based Union-Find that runs in a single pass via mapPartitions on the meta-graph edges, similar to Phase 1 but on the cross-partition edges. This eliminates iterations altogether
    ☐ Step5 phase2 - convert to Long type for docId.
       ☐ potentially phase1
    ☐ 90k WET
    ☐ Package as library
    ☐ consume from library
    ☐ [Low priority] Unit test for driver memory diagnoser

# Learning - scala

- object vs class
  - Object means singleton, 1 instance per JVM.
  - class has a public constructor 
  - object — no public constructor
    object MyTest extends AnyFunSuite { ... }
    // testClass.newInstance() → fails, constructor is private
    // Scala compiles object to:
    //   class MyTest$ {
    //     private MyTest$() { ... }        ← private constructor
    //     public static MyTest$ MODULE$;   ← singleton accessed here
    //   }
- for function attributes like cache obj, in scala, use lazy val which is 
  singleton per JVM worker.
- lazy val should be in the companion object, not inside the UDF:
  - lazy val — "compute once, cache the result" (per instance)
  - object   — "one instance for the entire JVM" (singleton)
- val means the reference can't change — it always points to the same Set object. mutable means the contents of that Set can change.
    scala// val + mutable: the container is fixed, but you can modify its contents
    val partitions = scala.collection.mutable.Set[Int]()
    partitions += 1      // ✅ modifying contents
    partitions += 2      // ✅ still the same Set object
    partitions = otherSet // ❌ can't reassign the reference

    // var + immutable: the reference can change, but each Set is frozen
    var partitions = Set[Int]()
    partitions += 1      // ✅ but this creates a NEW Set and reassigns the variable
    partitions = otherSet // ✅ can reassign
- val vs var Again:
    - if it's Array, we can declare as val and then do += operation, but in Int, we need var if we want to increment value later?
    - ArrayBuffer — val is fine, you're modifying contents, not the reference
    - Int — need var, because you're replacing the value entirely
- do NOT return in a lambda function
- every declaration required = sign
- Map vs HashMap
    - Map: immutable
    - HashMap: can be mutable. loop through and add into HashMap
        - [Important] declareing HashMap type does not auto-create entries for new keys
        similar to python dict, we need to declare initial value as empty ArrayBuffer.
        - A cleaner shortcut is getOrElseUpdate, which is Scala's equivalent of defaultdict:
        bandIndex.getOrElseUpdate(bandKey, mutable.ArrayBuffer[LocalDoc]()) += doc
- case class: python data class. Function name needs to start with Capital letter.
- Seq[Long] vs Array[Long]: when processing all rows in a partition, use Seq for rdd operation.
- ArrayBuffer vs Array: ArrayBuffer — resizable, like Python's list. Array — fixed size
- zipWithIndex: same as python's enumerate()
- spark.createDataFrame(Row iterator, outputSchema): does not fit with data (=case) class. 
    - use Row when you create 
    - Alternative: case class's iterator.toDF()
        - you need following code to do that:
        ```
        val spark = df.sparkSession
        spark.implicits._
        ```
        this is because RDD has no .toDF() method. 
        implicits._ lets you run rddToDatasetHolder(resultRDD).toDF() in the background.
        
- data class should be declared at object level, not function level. 
    - because lambda function from df.rdd.mapPartitions will serialize everything in lambda and if data class 
      is in the lambda function, it references outer class of case class, which can contain dataframe. 
      But dataframe is not serializable.
- when we create a row in Spark from Array of something, always need to convert Array toSeq
  - Reason: Spark's ArrayType stores data internally as Seq, so when creating a Row, it expects Seq for array columns.
  - This is Spark thing.
- a variable in args is a method parameter — immutable
 ex) def find(node: String): String = {
  here, node is immutable in scala

- == vs string.equals
  - in Java, == checks if memory address is the same or not and equals check value of adddress.
  - in scala, == under the hood is calling equals so they are essentially the same.
    - for address == check in scala, use eq or ne
- [mapPartitions and map returns iterator] for example, 
   ```
   val resultRDD = "some DF".rdd.mapPartitions { iterator: Iterator[Row] =>
    {some Array or hashMap}.map { docId => localUnionFindSchema( docId, unionFindInstance.find(docId) ) }
   } 
   ```
   this .map returns iterator. so iterator => iterator holds.

   This is better than
   ```
   val resultRDD = "some DF".rdd.mapPartitions { iterator: Iterator[Row] =>

      val localResults = mutable.ArrayBuffer[localUnionFindSchema]()
      for (docId <- unionFindInstance.parent.keysIterator) {
        localResults += localUnionFindSchema(
                      docId,
                      unionFindInstance.find(docId) // localRepresentative
                    )

      }
      localResults.iterator 
   }
   ```
- collect() order is not guaranteed.
  that is why you need to use Map
  ```
  val reps: Map[String, String] = resultArray.map(r => r.doc_id -> r.local_representative).toMap

  assert(reps.size == 4)
  assert(reps("A") == reps("B"))
  assert(reps("C") == reps("D"))
  assert(reps("A") != reps("C"))
  ```
- import spark.implicits._ because for .as[T] needs to convert DataFrame → Dataset[T].
- keys vs keySet: keys is iterable for doing for loop, and keySet is to actually get Set data.
- rdd.partitionBy vs df.repartition
  - rdd.partitionBy uses https://github.com/apache/spark/blob/master/core/src/main/scala/org/apache/spark/Partitioner.scala#L114
    - this uses identical mapping if partition key value is less than numPartitions
  - df.repartition uses https://github.com/apache/spark/blob/master/sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/plans/physical/partitioning.scala
    - when you call df.repartition(27000, col("partition_id")), Spark computes:
      ```physical_partition = pmod(Murmur3Hash(partition_id), 27000)```
- [Any type] Any is the root of Scala's type hierarchy — every class inherits from it.
  hashCode is defined on Any
  Scala type hierarchy:
    Any                    ← hashCode(), equals(), toString() defined here
    ├── AnyVal             ← Int, Long, Double, Boolean, etc.
    └── AnyRef             ← all classes (maps to java.lang.Object)
        ├── String
        ├── List
        ├── your classes
        └── ...
- Any.hashCode() 
  - Scala's Any.hashCode() maps to Java's Object.hashCode() at the JVM level. So when Spark calls key.hashCode, it's calling Java's Integer.hashCode() on a partition_id
- instead of count, use accumulator (actually not tied to scala at all)
  - What is accumulator: 
    Accumulators are write-only on executors, read-only on driver. No synchronization between executors:
  - Why accumulator:
    - we can skip df.count()
