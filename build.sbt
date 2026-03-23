name := "minhash-udf"
version := "0.1"
scalaVersion := "2.12.18"

// NOT included in the JAR because EMR already has it
libraryDependencies += "org.apache.spark" %% "spark-sql" % "3.5.0" % "provided"

// unit test - only available in src/test/scala/  NOT included in the JAR
libraryDependencies += "org.scalatest" %% "scalatest" % "3.2.17" % "test"