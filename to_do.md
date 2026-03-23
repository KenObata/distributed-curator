
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
    ☐ Implement incremental union-find for
      faster group building
    ☐ Increase similarity threshold to reduce
      false positives
    - scala UDF
    - Unit test for scala UDF
    - test 1000 partition count with python MinHash UDF for Found xxx similar pair.
      - DONE. simiar pair is more impacted by spark partition count rather than hash algo.
    - test 100 WET file with Cython UDF.
      - DONE. 4 min in Step1. 15 min E2E.
    ☐ Setup mandatory pre-commit script for pytest and sbt test
    ☐ optimize fetching WET files?
    ☐ 90k WET
    ☐ Package as library
    ☐ consume from library

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