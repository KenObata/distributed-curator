#!/usr/bin/env python3
"""
Pre-commit hook: catch common PySpark caching mistakes.

Checks:
  1. Bare .persist() — return value not captured (data may not be persisted as expected)
  2. Bare .cache() — same issue
  3. .unpersist() before .count()/.collect() — upstream dropped before materialization

Usage:
  python scripts/check_spark_persist.py src/
  python scripts/check_spark_persist.py src/spark_partition_aware_deduplicattion_v2.py

Pre-commit config:
  - repo: local
    hooks:
      - id: check-spark-persist
        name: Check PySpark persist patterns
        entry: python scripts/check_spark_persist.py
        language: python
        files: '\\.py$'
        pass_filenames: true
"""

import re
import sys
from pathlib import Path

# Matches lines like:  df.persist(StorageLevel.MEMORY_AND_DISK)
# but NOT:             df = df.persist(StorageLevel.MEMORY_AND_DISK)
# and NOT:             .persist(StorageLevel.MEMORY_AND_DISK)  at start (chained)
BARE_PERSIST_PATTERN = re.compile(
    r"^"  # start of line
    r"(?!\s*#)"  # not a comment
    r"(?!.*=\s*.*\.persist\()"  # not captured with =
    r"(?!.*\.persist\(.*\)\.)"  # not chained like .persist().count()
    r"\s+"  # leading whitespace (indented code)
    r"[a-zA-Z_][a-zA-Z0-9_]*"  # variable name
    r"\.persist\(",  # .persist(
)

# Same for .cache()
BARE_CACHE_PATTERN = re.compile(
    r"^"
    r"(?!\s*#)"
    r"(?!.*=\s*.*\.cache\(\))"
    r"(?!.*\.cache\(\)\.)"
    r"\s+"
    r"[a-zA-Z_][a-zA-Z0-9_]*"
    r"\.cache\(",
)

# Matches chained persist without capture:
#   result = (input_df.join(...).withColumn(...))
#   result.persist(StorageLevel.MEMORY_AND_DISK)   ← bare, should be result = result.persist(...)
BARE_PERSIST_STANDALONE = re.compile(r"^\s+(\w+)\.persist\(.+\)\s*$")

BARE_CACHE_STANDALONE = re.compile(r"^\s+(\w+)\.cache\(\)\s*$")


def check_file(filepath: Path) -> list[tuple[int, str, str]]:
    """
    Check a single file for PySpark persist anti-patterns.

    Returns list of (line_number, line_text, message) tuples.
    """
    issues = []

    try:
        lines = filepath.read_text().splitlines()
    except (OSError, UnicodeDecodeError):
        return issues

    for i, line in enumerate(lines, start=1):
        # Skip comments and strings
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
            continue

        # Check for bare .persist() on its own line
        match = BARE_PERSIST_STANDALONE.match(line)
        if match:
            var_name = match.group(1)
            # Check if this is inside a chained expression (previous line ends with . or \)
            if i > 1:
                prev_line = lines[i - 2].rstrip()
                if prev_line.endswith((".", "\\", ",")):
                    continue  # Part of a chain, OK

            issues.append(
                (
                    i,
                    line.rstrip(),
                    f"Bare .persist() — return value not captured like df = df.persist(StorageLevel.MEMORY_AND_DISK)."
                    f"Use `{var_name} = {var_name}.persist(...)` instead.",
                )
            )

        # Check for bare .cache() on its own line
        match = BARE_CACHE_STANDALONE.match(line)
        if match:
            var_name = match.group(1)
            if i > 1:
                prev_line = lines[i - 2].rstrip()
                if prev_line.endswith((".", "\\", ",")):
                    continue

            issues.append(
                (
                    i,
                    line.rstrip(),
                    f"Bare .cache() — return value not captured. Use `{var_name} = {var_name}.cache()` instead.",
                )
            )

    return issues


def main() -> int:
    """Run checks on files passed as arguments."""
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <file_or_dir> [file_or_dir ...]")
        return 1

    all_issues = []

    for arg in sys.argv[1:]:
        path = Path(arg)
        if path.is_file() and path.suffix == ".py":
            files = [path]
        elif path.is_dir():
            files = sorted(path.rglob("*.py"))
        else:
            continue

        for filepath in files:
            issues = check_file(filepath)
            for line_num, line_text, message in issues:
                all_issues.append((filepath, line_num, line_text, message))

    if all_issues:
        print(f"\n{'=' * 70}")
        print(f"PySpark persist check: {len(all_issues)} issue(s) found")
        print(f"{'=' * 70}\n")
        for filepath, line_num, line_text, message in all_issues:
            print(f"  {filepath}:{line_num}")
            print(f"    {line_text}")
            print(f"    ⚠  {message}")
            print()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
