package com.utils

import org.slf4j.{Logger, LoggerFactory}

object Utils {
  val logger = LoggerFactory.getLogger(Utils.getClass)

  def plotHeapMemory(label: String = ""): Unit = {
    val runtime       = Runtime.getRuntime
    val usageMB: Long = (runtime.totalMemory - runtime.freeMemory) / (1024 * 1024)
    System.out.println(s"[MEM] $label: $usageMB MB")

  }

}
