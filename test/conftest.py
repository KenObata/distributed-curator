# test/conftest.py
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(REPO_ROOT, "src")

# The JVM is crashing at startup — Java 25 is too new for PySpark.
# PySpark 3.5 officially supports Java 8, 11, and 17. Java 25 isn't compatible.
os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@17"

# Because Arrow can't access sun.misc.Unsafe
os.environ["_JAVA_OPTIONS"] = (
    "--add-opens=java.base/java.nio=ALL-UNNAMED "
    "--add-opens=java.base/sun.misc=ALL-UNNAMED "
    "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED"
)
# test files to recognize src/ directory
os.environ["PYTHONPATH"] = src_dir + ":" + os.environ.get("PYTHONPATH", "")
