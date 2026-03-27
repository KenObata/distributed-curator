# test/conftest.py
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(REPO_ROOT, "src")
test_dir = os.path.join(REPO_ROOT, "test")
integration_test_dir = os.path.join(test_dir, "integration_test")
# The JVM is crashing at startup — Java 25 is too new for PySpark.
# PySpark 3.5 officially supports Java 8, 11, and 17. Java 25 isn't compatible.
os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@17"

# Because Arrow can't access sun.misc.Unsafe
os.environ["_JAVA_OPTIONS"] = (
    "--add-opens=java.base/java.nio=ALL-UNNAMED "
    "--add-opens=java.base/sun.misc=ALL-UNNAMED "
    "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED"
)

"""
Why I set PYTHONPATH on top of sys.path.insert
pytest process (current process)
├── sys.path.insert → finds wet_file_utils, spark_partition_aware_deduplicattion_v2
│
└── SparkSession local[2]
    ├── Python worker subprocess 1 → needs PYTHONPATH to find udf.py
    └── Python worker subprocess 2 → needs PYTHONPATH to find udf.py
"""
# For pyspark
os.environ["PYTHONPATH"] = src_dir + ":" + os.environ.get("PYTHONPATH", "")

# For pytest
sys.path.insert(0, src_dir)
sys.path.insert(0, integration_test_dir)
