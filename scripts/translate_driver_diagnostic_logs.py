#!/usr/bin/env python3
"""
diagnose_driver_oom.py — Translate Spark driver diagnostic logs into plain English.

Usage:
    # Analyze all three logs:
    python diagnose_driver_oom.py \
        --heap driver_heap_histo.txt \
        --gc driver_gc.log \
        --mem driver_mem.log

    # Any combination works:
    python diagnose_driver_oom.py --heap driver_heap_histo.txt
    python diagnose_driver_oom.py --gc driver_gc.log

Prerequisites:
    # Generate the heap histogram from the .hprof file:
    jmap -histo driver_heap.hprof > driver_heap_histo.txt

    # Or if the JVM is still alive:
    jmap -histo <pid> > driver_heap_histo.txt

No external dependencies — stdlib only.
"""

import argparse
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

# =============================================================================
# Known Spark/JVM class patterns → human-readable categories
# =============================================================================

# Each entry: (category_name, description, fix)
# Classes are matched in order; first match wins.

HEAP_PATTERNS = [
    # --- Lineage / plan accumulation ---
    {
        "category": "Spark Logical Plan (Lineage Accumulation)",
        "classes": [
            "catalyst.trees.TreeNode",
            "catalyst.expressions.AttributeReference",
            "catalyst.expressions.Alias",
            "catalyst.expressions.BoundReference",
            "catalyst.expressions.Literal",
            "catalyst.plans.logical.Join",
            "catalyst.plans.logical.Project",
            "catalyst.plans.logical.Filter",
            "catalyst.plans.logical.Aggregate",
            "catalyst.plans.logical.SubqueryAlias",
            "catalyst.analysis.UnresolvedRelation",
            "catalyst.plans.logical.LocalRelation",
        ],
        "description": (
            "These are Spark's internal representation of your query plan. "
            "When you build DataFrames iteratively (e.g., joining in a loop) "
            "without checkpointing, the plan tree doubles each iteration. "
            "The driver must hold the entire plan in memory to submit jobs."
        ),
        "fix": (
            "Add .checkpoint() or .localCheckpoint() at the end of each "
            "iteration to truncate the lineage. localCheckpoint() avoids "
            "HDFS writes but loses fault tolerance. For a batch pipeline, "
            "localCheckpoint() is usually fine."
        ),
    },
    # --- Spark execution plan (physical) ---
    {
        "category": "Spark Physical Plan",
        "classes": [
            "spark.sql.execution.SparkPlan",
            "spark.sql.execution.WholeStageCodegen",
            "spark.sql.execution.ShuffleExchange",
            "spark.sql.execution.BroadcastExchange",
            "spark.sql.execution.joins.",
        ],
        "description": (
            "Physical execution plan objects. Usually grows alongside the "
            "logical plan — same root cause as lineage accumulation."
        ),
        "fix": "Same as lineage accumulation — checkpoint to truncate.",
    },
    # --- Broadcast variables ---
    {
        "category": "Broadcast Variables",
        "classes": [
            "spark.broadcast.TorrentBroadcast",
            "spark.broadcast.Broadcast",
            "spark.sql.execution.joins.HashedRelation",
            "spark.sql.execution.joins.UnsafeHashedRelation",
            "spark.sql.execution.joins.LongHashedRelation",
        ],
        "description": (
            "Data broadcast to all executors lives on the driver too. "
            "If you're broadcasting large DataFrames (e.g., a lookup table "
            "or a large dimension table), each one consumes driver memory."
        ),
        "fix": (
            "Check for broadcast joins on large tables. Either: "
            "(1) increase spark.sql.autoBroadcastJoinThreshold to -1 to "
            "disable auto-broadcast, (2) explicitly unpersist broadcast "
            "variables after use with .unpersist(), or (3) reduce the size "
            "of the broadcasted data."
        ),
    },
    # --- Collected data (results pulled to driver) ---
    {
        "category": "Collected Data on Driver",
        "classes": [
            "spark.sql.catalyst.InternalRow",
            "spark.sql.catalyst.expressions.UnsafeRow",
            "spark.unsafe.types.UTF8String",
            "org.apache.spark.util.collection.CompactBuffer",
        ],
        "description": (
            "Row data that was pulled to the driver via .collect(), "
            ".toPandas(), .toLocalIterator(), or a Spark action that "
            "materializes results on the driver."
        ),
        "fix": (
            "Avoid .collect() on large DataFrames. If you need aggregated "
            "results, compute them on executors first (groupBy/agg) and "
            "collect only the summary. For Union-Find, keep the graph "
            "distributed rather than collecting edges to the driver."
        ),
    },
    # --- Raw byte arrays / strings (generic data accumulation) ---
    {
        "category": "Raw Data (byte arrays / Strings)",
        "classes": [
            "[B",
            "[C",
            "java.lang.String",
            "[Ljava.lang.Object;",
            "[Ljava.lang.String;",
            "[J",
            "[I",
        ],
        "description": (
            "Generic JVM arrays and strings. By themselves these don't "
            "tell you much — look at what OTHER categories are large. "
            "If this is the dominant category with no plan/broadcast "
            "objects, it likely means raw data was collected to the driver "
            "(e.g., via .collect() or driver-side processing)."
        ),
        "fix": (
            "Check the Dominator Tree in Eclipse MAT to see what holds "
            "references to these arrays. Common culprits: collected "
            "DataFrame results, accumulated log strings, or serialized "
            "task results."
        ),
    },
    # --- Scala collections (often from accumulated results) ---
    {
        "category": "Scala/Java Collections",
        "classes": [
            "scala.collection.immutable.$colon$colon",
            "scala.collection.immutable.Vector",
            "scala.collection.mutable.ArrayBuffer",
            "scala.collection.mutable.HashMap",
            "java.util.ArrayList",
            "java.util.HashMap",
            "java.util.LinkedList",
        ],
        "description": (
            "Collections used internally by Spark or your UDFs. Large "
            "counts usually mean data is being accumulated in a collection "
            "on the driver side."
        ),
        "fix": (
            "Identify what's filling these collections. Common causes: "
            "driver-side loops collecting partition results, or Spark's "
            "internal event log / listener accumulation over long jobs."
        ),
    },
    # --- Spark event/listener accumulation ---
    {
        "category": "Spark Event Listeners / UI Metadata",
        "classes": [
            "spark.status.api.",
            "spark.ui.",
            "spark.scheduler.StageInfo",
            "spark.scheduler.TaskInfo",
            "spark.status.TaskDataWrapper",
            "spark.status.StageDataWrapper",
            "spark.status.JobDataWrapper",
        ],
        "description": (
            "Spark's internal bookkeeping for the History Server / UI. "
            "For very long jobs with thousands of stages (common in "
            "iterative algorithms), this metadata accumulates."
        ),
        "fix": (
            "Set spark.ui.retainedStages and spark.ui.retainedJobs to "
            "lower values (e.g., 100 instead of the default 1000). "
            "Also consider spark.ui.retainedTasks=10000."
        ),
    },
]


# =============================================================================
# Heap dump histogram parser
# =============================================================================


@dataclass
class HeapEntry:
    instances: int
    bytes: int
    class_name: str


@dataclass
class CategorySummary:
    name: str
    total_bytes: int = 0
    total_instances: int = 0
    top_classes: list = field(default_factory=list)
    description: str = ""
    fix: str = ""


def parse_heap_histogram(path: str) -> list[HeapEntry]:
    """Parse jmap -histo output."""
    entries = []
    pattern = re.compile(r"\s*\d+:\s+(\d+)\s+(\d+)\s+(.+)")
    with open(path) as f:
        for line in f:
            m = pattern.match(line.strip())
            if m:
                entries.append(
                    HeapEntry(
                        instances=int(m.group(1)),
                        bytes=int(m.group(2)),
                        class_name=m.group(3).strip(),
                    )
                )
    return entries


def classify_heap(entries: list[HeapEntry]) -> list[CategorySummary]:
    """Map heap entries to human-readable categories."""
    categories: dict[str, CategorySummary] = {}
    classified_bytes = 0
    total_bytes = sum(e.bytes for e in entries)

    for entry in entries:
        matched = False
        for pattern in HEAP_PATTERNS:
            for cls_fragment in pattern["classes"]:
                if cls_fragment in entry.class_name:
                    cat_name = pattern["category"]
                    if cat_name not in categories:
                        categories[cat_name] = CategorySummary(
                            name=cat_name,
                            description=pattern["description"],
                            fix=pattern["fix"],
                        )
                    cat = categories[cat_name]
                    cat.total_bytes += entry.bytes
                    cat.total_instances += entry.instances
                    cat.top_classes.append(entry)
                    classified_bytes += entry.bytes
                    matched = True
                    break
            if matched:
                break

    # Sort categories by total bytes descending
    sorted_cats = sorted(categories.values(), key=lambda c: c.total_bytes, reverse=True)
    return sorted_cats, total_bytes, classified_bytes


def format_bytes(b: int) -> str:
    if b >= 1024**3:
        return f"{b / 1024**3:.1f} GB"
    if b >= 1024**2:
        return f"{b / 1024**2:.1f} MB"
    if b >= 1024:
        return f"{b / 1024:.1f} KB"
    return f"{b} B"


def report_heap(path: str) -> str:
    entries = parse_heap_histogram(path)
    if not entries:
        return "  Could not parse heap histogram. Expected jmap -histo format.\n"

    sorted_cats, total_bytes, classified_bytes = classify_heap(entries)

    lines = []
    lines.append("=" * 72)
    lines.append("HEAP DUMP ANALYSIS — What consumed driver memory?")
    lines.append("=" * 72)
    lines.append(f"Total heap: {format_bytes(total_bytes)}")
    lines.append(f"Classified: {format_bytes(classified_bytes)} ({100 * classified_bytes / total_bytes:.0f}%)")
    lines.append("")

    for i, cat in enumerate(sorted_cats, 1):
        pct = 100 * cat.total_bytes / total_bytes
        if pct < 1:
            continue

        lines.append(f"  #{i}  {cat.name}")
        lines.append(f"      Size: {format_bytes(cat.total_bytes)} ({pct:.0f}% of heap)")
        lines.append(f"      Instances: {cat.total_instances:,}")
        lines.append("")
        lines.append("      What is this?")
        lines.append(f"      {cat.description}")
        lines.append("")
        lines.append("      How to fix:")
        lines.append(f"      {cat.fix}")
        lines.append("")

        # Show top 3 classes in this category
        top = sorted(cat.top_classes, key=lambda e: e.bytes, reverse=True)[:3]
        lines.append("      Top classes:")
        for e in top:
            lines.append(f"        - {e.class_name}: {e.instances:,} instances, {format_bytes(e.bytes)}")
        lines.append("")
        lines.append("-" * 72)
        lines.append("")

    # Verdict
    if sorted_cats:
        top_cat = sorted_cats[0]
        top_pct = 100 * top_cat.total_bytes / total_bytes
        lines.append("VERDICT")
        lines.append(f"  The dominant consumer is: {top_cat.name}")
        lines.append(f"  It accounts for {top_pct:.0f}% of the heap.")
        lines.append("")
        lines.append("  Recommended fix:")
        lines.append(f"  {top_cat.fix}")

    return "\n".join(lines)


# =============================================================================
# GC log parser
# =============================================================================


@dataclass
class GCEvent:
    timestamp: str
    uptime: str
    gc_type: str  # "Young" or "Full"
    cause: str
    before_mb: float
    after_mb: float
    heap_mb: float
    pause_ms: float


def parse_gc_log(path: str) -> list[GCEvent]:
    """Parse Java 17+ unified GC log (-Xlog:gc*)."""
    events = []
    # Pattern: [timestamp][uptime][level][gc] GC(N) Pause Young/Full (cause) before->after(heap) time
    pattern = re.compile(
        r"\[([^\]]+)\]\[([^\]]+)\]\[info\]\[gc\]\s*"
        r"GC\(\d+\)\s+Pause\s+(Young|Full)\s+"
        r"\(([^)]+)\)\s+"
        r"(\d+)M->(\d+)M\((\d+)M\)\s+"
        r"([\d.]+)ms"
    )
    with open(path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                events.append(
                    GCEvent(
                        timestamp=m.group(1),
                        uptime=m.group(2),
                        gc_type=m.group(3),
                        cause=m.group(4),
                        before_mb=float(m.group(5)),
                        after_mb=float(m.group(6)),
                        heap_mb=float(m.group(7)),
                        pause_ms=float(m.group(8)),
                    )
                )
    return events


def report_gc(path: str) -> str:
    events = parse_gc_log(path)
    if not events:
        return "  Could not parse GC log. Expected Java 17+ -Xlog:gc* format.\n"

    lines = []
    lines.append("=" * 72)
    lines.append("GC LOG ANALYSIS — How did the JVM respond to memory pressure?")
    lines.append("=" * 72)

    full_gcs = [e for e in events if e.gc_type == "Full"]
    young_gcs = [e for e in events if e.gc_type == "Young"]

    lines.append(f"Total GC events: {len(events)} ({len(young_gcs)} young, {len(full_gcs)} full)")
    lines.append("")

    # Check for GC thrashing: multiple full GCs reclaiming < 5% each
    if full_gcs:
        last_10_full = full_gcs[-10:]
        thrashing_count = 0
        for e in last_10_full:
            reclaimed = e.before_mb - e.after_mb
            reclaim_pct = (reclaimed / e.before_mb * 100) if e.before_mb > 0 else 0
            if reclaim_pct < 5:
                thrashing_count += 1

        if thrashing_count >= 3:
            lines.append("  ⚠ GC THRASHING DETECTED")
            lines.append(
                f"    {thrashing_count} of the last {len(last_10_full)} Full GCs reclaimed less than 5% of the heap."
            )
            lines.append("    This means almost all objects in memory are still in use")
            lines.append("    (live references). The JVM is spending most of its CPU")
            lines.append("    on garbage collection instead of running your code.")
            lines.append("")
            lines.append("    This is NOT a memory leak in the traditional sense —")
            lines.append("    it means your code is holding references to too many")
            lines.append("    objects at once. Check the heap dump analysis above")
            lines.append("    to see what those objects are.")
        else:
            lines.append("  No GC thrashing detected.")
            lines.append("  The JVM was able to reclaim memory effectively until")
            lines.append("  a single large allocation exceeded available heap.")
            lines.append("  This suggests a sudden spike (e.g., .collect() on a")
            lines.append("  large DataFrame) rather than gradual accumulation.")

        lines.append("")
        lines.append("-" * 72)

        # Final GC events
        lines.append("")
        lines.append("  Last 5 Full GC events before crash:")
        lines.append(f"  {'Uptime':<12} {'Before':>10} {'After':>10} {'Reclaimed':>10} {'Pause':>10} {'Cause'}")
        lines.append(f"  {'-' * 12} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 20}")

        for e in full_gcs[-5:]:
            reclaimed = e.before_mb - e.after_mb
            lines.append(
                f"  {e.uptime:<12} {e.before_mb:>8.0f}MB {e.after_mb:>8.0f}MB "
                f"{reclaimed:>8.0f}MB {e.pause_ms:>8.1f}ms {e.cause}"
            )

        lines.append("")

        # Total time in GC
        total_pause = sum(e.pause_ms for e in events)
        full_pause = sum(e.pause_ms for e in full_gcs)
        lines.append(f"  Total time in GC: {total_pause / 1000:.1f}s (Full GC: {full_pause / 1000:.1f}s)")

        if full_gcs:
            max_full = max(full_gcs, key=lambda e: e.pause_ms)
            lines.append(f"  Longest Full GC pause: {max_full.pause_ms:.1f}ms at {max_full.uptime}")

    # Peak heap usage
    if events:
        peak = max(events, key=lambda e: e.before_mb)
        lines.append(
            f"  Peak heap before GC: {format_bytes(int(peak.before_mb * 1024 * 1024))} "
            f"/ {format_bytes(int(peak.heap_mb * 1024 * 1024))} max"
        )

    return "\n".join(lines)


# =============================================================================
# JMX memory log parser (Option 3 output)
# =============================================================================


@dataclass
class MemEntry:
    timestamp: str
    used_mb: float
    total_mb: float
    max_mb: float
    non_heap_used_mb: float = 0
    non_heap_committed_mb: float = 0
    context: Optional[str] = None  # job description if present


def parse_mem_log(path: str) -> list[MemEntry]:
    """Parse the JMX periodic memory logger output."""
    entries = []
    # Match: [DRIVER MEM] used=1842MB total=4096MB max=32768MB | non_heap=120MB/256MB
    mem_pattern = re.compile(
        r"\[DRIVER MEM\]\s+used=(\d+)MB\s+total=(\d+)MB\s+max=(\d+)MB"
        r"(?:\s*\|\s*non_heap=(\d+)MB/(\d+)MB)?"
    )
    # Match: job description lines or Phase 2 iteration lines
    context_pattern = re.compile(r"(?:Setting job description:\s*(.+)|Phase \d+:.+)")

    current_context = None
    with open(path) as f:
        for line in f:
            ctx_match = context_pattern.search(line)
            if ctx_match:
                current_context = ctx_match.group(0).strip()
                # Also add context-only entries for timeline
                if "Setting job description" in line:
                    current_context = ctx_match.group(1).strip()

            m = mem_pattern.search(line)
            if m:
                entries.append(
                    MemEntry(
                        timestamp="",
                        used_mb=float(m.group(1)),
                        total_mb=float(m.group(2)),
                        max_mb=float(m.group(3)),
                        non_heap_used_mb=float(m.group(4)) if m.group(4) else 0,
                        non_heap_committed_mb=float(m.group(5)) if m.group(5) else 0,
                        context=current_context,
                    )
                )
    return entries


def report_mem(path: str) -> str:
    entries = parse_mem_log(path)
    if not entries:
        return "  Could not parse memory log. Expected [DRIVER MEM] format.\n"

    lines = []
    lines.append("=" * 72)
    lines.append("MEMORY TIMELINE — When did memory spike?")
    lines.append("=" * 72)

    max_mem = entries[0].max_mb if entries else 0
    lines.append(f"Max heap configured: {format_bytes(int(max_mem * 1024 * 1024))}")
    lines.append("")

    # Group by context (step), show peak for each
    context_peaks: dict[str, float] = {}
    context_order: list[str] = []
    prev_context = None
    for e in entries:
        ctx = e.context or "(unknown step)"
        if ctx != prev_context:
            if ctx not in context_peaks:
                context_order.append(ctx)
            prev_context = ctx
        context_peaks[ctx] = max(context_peaks.get(ctx, 0), e.used_mb)

    lines.append("  Peak memory by pipeline step:")
    lines.append(f"  {'Step':<55} {'Peak':>8} {'% of max':>8}")
    lines.append(f"  {'-' * 55} {'-' * 8} {'-' * 8}")

    prev_peak = 0
    for ctx in context_order:
        peak = context_peaks[ctx]
        pct = 100 * peak / max_mem if max_mem else 0
        delta = peak - prev_peak
        marker = ""
        if delta > max_mem * 0.15:
            marker = " ← LARGE JUMP"
        elif delta > max_mem * 0.05:
            marker = " ← notable increase"

        lines.append(f"  {ctx:<55} {peak:>6.0f}MB {pct:>6.0f}%{marker}")
        prev_peak = peak

    lines.append("")

    # Detect doubling pattern (lineage accumulation signature)
    peaks_in_order = [context_peaks[ctx] for ctx in context_order]
    if len(peaks_in_order) >= 4:
        diffs = [peaks_in_order[i + 1] - peaks_in_order[i] for i in range(len(peaks_in_order) - 1)]
        # Check if diffs are roughly doubling
        doubling_count = 0
        for i in range(len(diffs) - 1):
            if diffs[i] > 100 and diffs[i + 1] > 100:  # meaningful diffs
                ratio = diffs[i + 1] / diffs[i]
                if 1.5 <= ratio <= 3.0:
                    doubling_count += 1

        if doubling_count >= 2:
            lines.append("  ⚠ EXPONENTIAL GROWTH DETECTED")
            lines.append("    Memory usage appears to be doubling between steps.")
            lines.append("    This is a strong signal of lineage/plan accumulation")
            lines.append("    from iterative DataFrame operations without checkpointing.")
            lines.append("")

    # Final reading
    last = entries[-1]
    pct_used = 100 * last.used_mb / last.max_mb if last.max_mb else 0
    lines.append(f"  Last reading before crash: {last.used_mb:.0f}MB ({pct_used:.0f}% of {last.max_mb:.0f}MB max)")
    if pct_used > 90:
        lines.append("  Driver was at >90% heap utilization when it died.")

    # Off-heap analysis (if non_heap data is available)
    non_heap_entries = [e for e in entries if e.non_heap_used_mb > 0]
    if non_heap_entries:
        lines.append("")
        lines.append("-" * 72)
        lines.append("")
        lines.append("  Off-heap (non-heap) memory:")
        first_nh = non_heap_entries[0].non_heap_used_mb
        last_nh = non_heap_entries[-1].non_heap_used_mb
        peak_nh = max(e.non_heap_used_mb for e in non_heap_entries)
        growth = last_nh - first_nh

        lines.append(f"    First reading: {first_nh:.0f}MB")
        lines.append(f"    Last reading:  {last_nh:.0f}MB")
        lines.append(f"    Peak:          {peak_nh:.0f}MB")
        lines.append(f"    Growth:        {growth:+.0f}MB")

        if growth > 500:
            lines.append("")
            lines.append("  ⚠ SIGNIFICANT OFF-HEAP GROWTH")
            lines.append(f"    Non-heap grew by {growth:.0f}MB over the run.")
            lines.append("    Common causes: Metaspace bloat from codegen classes,")
            lines.append("    Netty direct buffers, or native memory leaks.")
            lines.append("    If the process was killed by the OS (no heap dump, no")
            lines.append("    Java exception), this is likely the culprit.")
            lines.append("    Run with -XX:NativeMemoryTracking=summary and use")
            lines.append("    jcmd <pid> VM.native_memory summary for the full breakdown.")
        elif pct_used < 50 and not non_heap_entries:
            lines.append("")
            lines.append("  Note: Heap was well under pressure when process died.")
            lines.append("  If no .hprof file was generated, suspect an off-heap issue")
            lines.append("  (OS OOM killer). Check dmesg on the master node.")

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Translate Spark driver diagnostic logs into plain English.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze heap dump histogram (generate with: jmap -histo file.hprof > histo.txt)
  python diagnose_driver_oom.py --heap histo.txt

  # Analyze all three logs
  python diagnose_driver_oom.py --heap histo.txt --gc driver_gc.log --mem driver_mem.log

  # Download from S3 and analyze
  aws s3 cp s3://bucket/heapdumps/j-ABC123/20250406/ ./dumps/ --recursive
  jmap -histo ./dumps/driver_heap.hprof > histo.txt
  python diagnose_driver_oom.py --heap histo.txt --gc ./dumps/driver_gc.log --mem ./dumps/driver_mem.log
        """,
    )
    parser.add_argument("--heap", help="Path to jmap -histo output file")
    parser.add_argument("--gc", help="Path to GC log file (-Xlog:gc*)")
    parser.add_argument("--mem", help="Path to JMX memory logger output")

    args = parser.parse_args()

    if not any([args.heap, args.gc, args.mem]):
        parser.print_help()
        sys.exit(1)

    sections = []

    if args.mem:
        sections.append(report_mem(args.mem))

    if args.gc:
        sections.append(report_gc(args.gc))

    if args.heap:
        sections.append(report_heap(args.heap))

    print("\n\n".join(sections))
    print()


if __name__ == "__main__":
    main()
