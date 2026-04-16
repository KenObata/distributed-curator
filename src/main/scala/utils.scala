package com.utils

import org.slf4j.{Logger, LoggerFactory}

object Utils {
  val logger = LoggerFactory.getLogger(Utils.getClass)

  def plotHeapMemory(label: String = ""): Unit = {
    val runtime       = Runtime.getRuntime
    val usageMB: Long = (runtime.totalMemory - runtime.freeMemory) / (1024 * 1024)
    logger.info(s"[Executor MEM] $label: $usageMB MB")

  }

}
