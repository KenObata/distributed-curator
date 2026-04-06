"""
driver_diagnostics.py — Spark driver memory observability utilities.

Following functions and spark config heap dunp are in an exclusive relationship:
    - Application OOMed on JVM:
        - -XX:+HeapDumpOnOutOfMemoryError config fires at the moment of crash, before the JVM dies
        - writes a .hprof file
    - Application completed successfully:
        - capture_heap_histogram() runs jmap -histo
    - Off-heap killed
        - capture_nmt_summary() runs

Unlike above functions, start_memory_logger() runs on every case to track free/used memory.


Usage:
    from driver_diagnostics import start_memory_logger, capture_heap_histogram

    spark = SparkSession.builder.getOrCreate()

    # Start periodic memory logging (appears in yarn logs -am 1 stdout)
    start_memory_logger(spark.SparkContext, interval_seconds=30)

    # ... run pipeline steps ...

    # At the very end, capture heap state for the diagnosis script
    capture_heap_histogram(spark.SparkContext)
    capture_nmt_summary(spark.SparkContext)     # only works if NMT flag is set
"""

import logging
import subprocess
import threading
import time

from pyspark import SparkContext

logger = logging.getLogger(__name__)
BYTES_PER_MB = 1024**2


def _get_driver_jvm_pid(sc: SparkContext) -> str:
    """Get the driver JVM process ID via py4j_gateway.
    py4j_gateway is a driver (=gateway) to the JVM - executors have no py4j bridge.

    This function is called from external CLI tool such as jmap -histo
    because this is external from JVM, MemoryMXBean.getHeapMemoryUsage() cannot be accessed.
    """
    py4j_gateway = sc._jvm
    runtime_bean = py4j_gateway.java.lang.management.ManagementFactory.getRuntimeMXBean()

    # getName() returns "pid@hostname"
    return runtime_bean.getName().split("@")[0]


def start_memory_logger(sc: SparkContext, interval_seconds: int = 30) -> threading.Thread:
    """
    Context: JVM heap dum .hprof is only when OOM happened.
    This function is regardless of OOM or not.

    What this does:
    Start a daemon thread that logs driver heap usage to stdout at
    regular intervals. Output appears in `yarn logs -am 1 -log_files stdout`.

    Args:
        sc: SparkContext
        interval_seconds: How often to log (default 30s)

    Returns:
        The daemon thread (for testing; you don't need to join it).

    Terminology:
    - used: memory occupied by live objects right now
    - committed (total): memory the JVM has reserved from the OS. This only goes up (or stays flat)
    - max: It is the ceiling (-Xmx)

    Note:
    - this function runs in-process via py4j, so no need to get driver pid externally.
      Just use getRuntime()
    """
    runtime = sc._jvm.java.lang.Runtime.getRuntime()
    py4j_obj = sc._jvm
    # https://docs.oracle.com/en/java/javase/17/docs/api/java.management/java/lang/management/MemoryMXBean.html
    memory_bean = py4j_obj.java.lang.management.ManagementFactory.getMemoryMXBean()

    def _loop():
        while True:
            try:
                heap = memory_bean.getHeapMemoryUsage()
                non_heap = memory_bean.getNonHeapMemoryUsage()

                heap_used = heap.getUsed() / (BYTES_PER_MB)
                heap_committed = heap.getCommitted() / (BYTES_PER_MB)
                heap_max = heap.getMax() / (BYTES_PER_MB)

                non_heap_used = non_heap.getUsed() / (BYTES_PER_MB)
                non_heap_committed = non_heap.getCommitted() / (BYTES_PER_MB)

                free = runtime.freeMemory() / (BYTES_PER_MB)  # heap_committed - heap_used

                print(
                    f"[DRIVER MEM] "
                    f"used={heap_used:.0f}MB "
                    f"total={heap_committed:.0f}MB "
                    f"free_space={free:.0f}MB "
                    f"max={heap_max:.0f}MB "
                    f"| non_heap={non_heap_used:.0f}MB/{non_heap_committed:.0f}MB",
                    flush=True,
                )
            except Exception as e:
                # py4j connection may drop if JVM is shutting down
                logger.debug(f"Memory logger stopping: {e}")
                break

            time.sleep(interval_seconds)

    t = threading.Thread(target=_loop, daemon=True, name="driver-mem-logger")
    t.start()
    logger.info(
        f"Driver memory logger started (interval={interval_seconds}s). "
        f"View with: yarn logs -applicationId <app_id> -log_files stdout -am 1"
    )
    return t


def capture_heap_histogram(sc: SparkContext, output_path: str = "/tmp/driver_heap_histo.txt") -> bool:
    """
    heap histogram (equivalent to jmap -histo) is something only get it on crash (e.g.OOM).
    To get it on every run, we need subprocess.run(jmap - histo).

    This function and JVM heap dump is exclusive relationship - see top doc string on this file.

    Args:
        sc: SparkContext
        output_path: Where to write the histogram (default: /tmp, picked up
                     by the shutdown upload script)

    Returns:
        True if capture succeeded, False otherwise.

    Note:
    - This function is called from external CLI tool such as jmap -histo
      that is why we need pid from the driver.

    ex)
    num     #instances         #bytes  class name (module)
    -------------------------------------------------------
    1:      18420841     2947334560  org.apache.spark.sql.catalyst.trees.TreeNode
    2:      12280432     1965268840  org.apache.spark.sql.catalyst.expressions.AttributeReference
    3:       9840221     1574435360  org.apache.spark.sql.catalyst.plans.logical.Join
    -------------------------------------------------------
    Total      64101575    10312849600
    """
    try:
        pid = _get_driver_jvm_pid(sc)
        logger.info(f"Capturing heap histogram for JVM pid={pid}...")

        result = subprocess.run(
            ["jmap", "-histo", pid],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode == 0 and result.stdout:
            with open(output_path, "w") as f:
                f.write(result.stdout)
            print(f"[DRIVER DIAG] Heap histogram written to {output_path}", flush=True)
            return True
        else:
            logger.warning(f"jmap -histo failed: {result.stderr}")
            return False

    except FileNotFoundError:
        logger.warning("jmap not found on PATH. Skipping heap histogram capture.")
        return False
    except subprocess.TimeoutExpired:
        logger.warning("jmap -histo timed out after 120s. Skipping.")
        return False
    except Exception as e:
        logger.warning(f"Heap histogram capture failed: {e}")
        return False


def capture_nmt_summary(sc: SparkContext, output_path: str = "/tmp/driver_nmt.txt") -> bool:
    """
    Capture NativeMemoryTracking summary (off-heap breakdown).
    Only works if JVM was started with -XX:NativeMemoryTracking=summary.

    Args:
        sc: SparkContext
        output_path: Where to write the NMT summary

    Returns:
        True if capture succeeded, False otherwise.
    """
    try:
        pid = _get_driver_jvm_pid(sc)
        logger.info(f"Capturing NMT summary for JVM pid={pid}...")

        result = subprocess.run(
            ["jcmd", pid, "VM.native_memory", "summary"],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode == 0 and result.stdout:
            with open(output_path, "w") as f:
                f.write(result.stdout)
            print(f"[DRIVER DIAG] NMT summary written to {output_path}", flush=True)
            return True
        else:
            # NMT not enabled — this is expected if the flag wasn't set
            if "not enabled" in (result.stderr or "").lower():
                logger.info("NMT not enabled. Add -XX:NativeMemoryTracking=summary to enable.")
            else:
                logger.warning(f"jcmd NMT failed: {result.stderr}")
            return False

    except Exception as e:
        logger.warning(f"NMT capture failed: {e}")
        return False
