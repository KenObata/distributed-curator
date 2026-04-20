# test/conftest.py
import os
import sys
from enum import Enum

import pytest
from pyspark.sql import SparkSession

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(REPO_ROOT, "src")
test_dir = os.path.join(REPO_ROOT, "test")
integration_test_dir = os.path.join(test_dir, "integration_test")
assembly_jar = os.path.join(REPO_ROOT, "target/scala-2.12/minhash-udf-assembly-0.1.jar")

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
# Force PySpark workers to use the same Python as the test runner
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# For pyspark
os.environ["PYTHONPATH"] = src_dir + ":" + os.environ.get("PYTHONPATH", "")

# For pytest
sys.path.insert(0, src_dir)
sys.path.insert(0, integration_test_dir)


class PytestSparkScope(Enum):
    # if you want to apply this fixutre to only the test file
    MODULE = "module"
    # if you want to apply this fixuture to all tests
    SESSION = "session"


@pytest.fixture(scope=PytestSparkScope.SESSION.value)
def spark():
    session = (
        SparkSession.builder.master("local[2]")
        .appName("UnitTests")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.jars", f"lib/graphframes-0.8.3-spark3.5-s_2.12.jar,{assembly_jar}")
        .getOrCreate()
    )
    yield session
    session.stop()
