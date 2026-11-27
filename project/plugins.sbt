// Scala formatter — enables `sbt scalafmtCheckAll` and `sbt scalafmtAll`
// scalafmtAll: rewrites files in place
// scalafmtCheckAll: read-only. This repo uses read-only.
addSbtPlugin("org.scalameta" % "sbt-scalafmt" % "2.5.4")
addSbtPlugin("com.eed3si9n" % "sbt-assembly" % "2.1.5")