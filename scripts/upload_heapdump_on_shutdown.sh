#!/bin/bash
# EMR Bootstrap Action: Upload driver heap dumps to S3 on cluster shutdown.
#
# What this does:
#   1. Installs a systemd service that runs on shutdown (before network goes down)
#   2. The service uploads any .hprof files from /tmp to the specified S3 path
#
# Usage (Terraform):
#   bootstrap_action {
#     name = "install-heapdump-upload"
#     path = "s3://your-bucket/bootstrap/upload_heapdump_on_shutdown.sh"
#     args = ["s3://your-bucket/heapdumps", "s3://your-bucket/bootstrap/diagnose_driver_oom.py"]
#   }
#
# Upload both files to S3 first:
#   aws s3 cp upload_heapdump_on_shutdown.sh s3://your-bucket/bootstrap/
#   aws s3 cp diagnose_driver_oom.py s3://your-bucket/bootstrap/
#
# The JVM flags (-XX:+HeapDumpOnOutOfMemoryError) should be set separately
# via spark.driver.extraJavaOptions in your Spark config.

set -euo pipefail

S3_DEST="${1:?Usage: $0 <s3-destination-path>}"
DIAG_SCRIPT_S3="${2:?Usage: $0 <s3-destination-path> <s3-path-to-diagnose_driver_oom.py>}"

# Download the diagnosis script to a known location
sudo aws s3 cp "${DIAG_SCRIPT_S3}" /usr/local/bin/diagnose_driver_oom.py --no-progress
sudo chmod +x /usr/local/bin/diagnose_driver_oom.py
INSTANCE_ID=$(ec2-metadata -i | awk '{print $2}')
CLUSTER_ID=$(cat /mnt/var/lib/info/job-flow.json | python3 -c "import sys,json; print(json.load(sys.stdin)['jobFlowId'])")
TIMESTAMP_AT_BOOT=$(date +%Y%m%dT%H%M%S)

# Create the upload script
sudo tee /usr/local/bin/upload-heapdumps.sh > /dev/null << 'UPLOAD_SCRIPT'
#!/bin/bash
S3_DEST="__S3_DEST__"
CLUSTER_ID="__CLUSTER_ID__"
INSTANCE_ID="__INSTANCE_ID__"
TIMESTAMP_AT_BOOT="__TIMESTAMP_AT_BOOT__"

S3_PREFIX="${S3_DEST}/${CLUSTER_ID}/${TIMESTAMP_AT_BOOT}"

shopt -s nullglob
DUMP_FILES=(/tmp/*.hprof /tmp/driver_gc.log /tmp/driver_mem.log /tmp/driver_heap_histo.txt /tmp/driver_nmt.txt)

if [ ${#DUMP_FILES[@]} -eq 0 ]; then
    echo "No diagnostic files found in /tmp. Nothing to upload."
    exit 0
fi

for f in "${DUMP_FILES[@]}"; do
    BASENAME=$(basename "$f")
    DEST="${S3_PREFIX}/${INSTANCE_ID}_${BASENAME}"
    echo "Uploading ${f} -> ${DEST}"
    aws s3 cp "$f" "$DEST" --no-progress
done

echo "Uploaded ${#DUMP_FILES[@]} diagnostic file(s) to ${S3_PREFIX}/"

# --- Generate human-readable diagnosis report ---
DIAG_ARGS=""

# Generate heap histogram from .hprof (crash), or use pre-captured one (success)
HPROF_FILES=(/tmp/*.hprof)
if [ ${#HPROF_FILES[@]} -gt 0 ]; then
    HPROF="${HPROF_FILES[0]}"
    echo "Generating heap histogram from ${HPROF}..."
    jmap -histo "${HPROF}" > /tmp/driver_heap_histo_from_hprof.txt 2>/dev/null || true
    if [ -s /tmp/driver_heap_histo_from_hprof.txt ]; then
        # Prefer the crash-time histogram over the end-of-pipeline one
        cp /tmp/driver_heap_histo_from_hprof.txt /tmp/driver_heap_histo.txt
    fi
fi

if [ -f /tmp/driver_heap_histo.txt ] && [ -s /tmp/driver_heap_histo.txt ]; then
    DIAG_ARGS="${DIAG_ARGS} --heap /tmp/driver_heap_histo.txt"
fi

# Extract JMX memory log from YARN AM stdout (if deploy-mode cluster)
if command -v yarn &> /dev/null; then
    # Find the application ID from the most recent Spark app
    APP_ID=$(yarn application -list -appStates FINISHED,FAILED,KILLED 2>/dev/null \
        | grep -i spark | tail -1 | awk '{print $1}') || true
    if [ -n "${APP_ID}" ]; then
        echo "Extracting driver memory log from YARN AM (${APP_ID})..."
        yarn logs -applicationId "${APP_ID}" -log_files stdout -am 1 2>/dev/null \
            | grep "\[DRIVER MEM\]\|\[DRIVER DIAG\]\|Setting job description\|Phase [0-9]" \
            > /tmp/driver_mem.log 2>/dev/null || true
    fi
fi

if [ -f /tmp/driver_gc.log ] && [ -s /tmp/driver_gc.log ]; then
    DIAG_ARGS="${DIAG_ARGS} --gc /tmp/driver_gc.log"
fi

if [ -f /tmp/driver_mem.log ] && [ -s /tmp/driver_mem.log ]; then
    DIAG_ARGS="${DIAG_ARGS} --mem /tmp/driver_mem.log"
fi

if [ -n "${DIAG_ARGS}" ]; then
    echo "Running diagnosis..."
    python3 /usr/local/bin/diagnose_driver_oom.py ${DIAG_ARGS} \
        > /tmp/diagnosis_report.txt 2>&1

    aws s3 cp /tmp/diagnosis_report.txt \
        "${S3_PREFIX}/${INSTANCE_ID}_diagnosis_report.txt" --no-progress
    echo "Diagnosis report uploaded to ${S3_PREFIX}/"

    # Also print the report to the systemd journal so it's visible in syslog
    echo "=== DRIVER OOM DIAGNOSIS ==="
    cat /tmp/diagnosis_report.txt
    echo "=== END DIAGNOSIS ==="
else
    echo "No diagnostic files available for analysis."
fi
UPLOAD_SCRIPT

# Substitute variables into the upload script
sudo sed -i "s|__S3_DEST__|${S3_DEST}|g" /usr/local/bin/upload-heapdumps.sh
sudo sed -i "s|__CLUSTER_ID__|${CLUSTER_ID}|g" /usr/local/bin/upload-heapdumps.sh
sudo sed -i "s|__INSTANCE_ID__|${INSTANCE_ID}|g" /usr/local/bin/upload-heapdumps.sh
sudo sed -i "s|__TIMESTAMP_AT_BOOT__|${TIMESTAMP_AT_BOOT}|g" /usr/local/bin/upload-heapdumps.sh
sudo chmod +x /usr/local/bin/upload-heapdumps.sh

# Install systemd service that runs before network shutdown
sudo cat > /etc/systemd/system/upload-heapdumps.service << 'SERVICE'
[Unit]
Description=Upload Spark driver heap dumps to S3
DefaultDependencies=no
Before=shutdown.target reboot.target halt.target
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStop=/usr/local/bin/upload-heapdumps.sh
TimeoutStopSec=300

[Install]
WantedBy=multi-user.target
SERVICE

sudo systemctl daemon-reload
sudo systemctl enable upload-heapdumps.service
sudo systemctl start upload-heapdumps.service

echo "Heap dump upload service installed successfully."