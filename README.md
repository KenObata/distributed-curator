# How to develop this repo


## physical hardware setting
We need Java 17 for using local spark.
```
brew install openjdk@17
```

## create virtual env

How to change your VIM setting
```
vim ~/.vimrc
```


one-off (3.9 is to align with EMR)
```
python3.9 -m venv venv39
```

Every time
```
deactivate
source venv39/bin/activate
```
You should see (venv) after actuvation.
# Install PySpark locally
```
pip install -r requirements.txt
pip install graphframes
```
graphframes is because on EMR, it's passed as --packages.


### Check common crawl file with curl
```
curl -I https://data.commoncrawl.org/crawl-data/CC-MAIN-2024-22/segments/1715971057216.39/wet/CC-MAIN-20240517233122-20240518023122-00000.warc.wet.gz | head -n 10
```

# How to use this library
setup
```
pip install spark-llm-dedup
```

in codebase, first run vanila spark ml library's text-deduplication. 
This will OOM error out after TB of text documets.
```
from spark_llm_dedup import deduplicate_corpus
deduplicate_corpus("s3://common-crawl/", threshold=0.8)
```


# How to unit/integration test
Every thing is part of pre-commit. But if you want to run test manually,
run ```pytest``` and ```sbt test```

# Terraform
Note ethat terraform init will create .terraform.locl.hcl file 
for dependency package control. We need to upload this file to git as well.

one-off command
```
terraform init
```
Step1:  Get Cluster DNS

Please add scaling variable, -var="wet_file_scale=90k" either 1k, 9k, or 90k.

Start with spot instances for cheaper cost.
```
terraform apply -var="wet_file_scale=1k" -var="instance_strategy=spot"
```

If above plan fails, then pay higher bidding.
```
terraform apply -var="wet_file_scale=1k" -var="instance_strategy=spot" -var="bid_strategy=peak-event"
```

If above plan fails, then do on-demand.
```
terraform apply -var="instance_strategy=on-demand" -var="wet_file_scale=9k"
```

If this failed due to bootstrap, do ssh and run the next command:
```
cat /mnt/var/log/bootstrap-actions/2/stderr
```

Note - you need to create your own terraform.tfvars file looks like this:
```
cluster_name   = "" # EMR cluster name
subnet_id      = "subnet-xxxxxx"          # Your subnet ID
vpc_id         = "vpc-xxxxxx"             # Your VPC ID
scripts_bucket = ""        # Your S3 bucket name
```

Step2:
then run these
```
terraform output master_public_dns
```

Step3: upload your requirements.txt to S3:

Then from SSH session on the master node:

```
ssh -i ./emr-dedupe-key.pem hadoop@<master-public-dns>
aws s3 cp s3://text-deduplication-740959772378/scripts/requirements.txt .
sudo pip3 install --ignore-installed --no-cache-dir --no-deps -r requirements.txt
```

[optional] upload jar if needed
```
aws s3 cp target/scala-2.12/minhash-udf_2.12-0.1.jar s3://text-deduplication-740959772378/scripts/minhash-udf_2.12-0.1.jar
```

Step4: Exit ssh, and on your macbook, install YARN(8088), Spark UI (4040)
Find YAN host name - run
```
hostname -f
```

Step5: setup YARN
```
source /etc/spark/conf/spark-env.sh
export HADOOP_CONF_DIR=/etc/hadoop/conf
export YARN_CONF_DIR=/etc/hadoop/conf
```
Step6: upload helper functions as zip
```
cd /llm_trainining/src
zip -j dependencies.zip spark_utils.py spark_partition_aware_deduplicattion_v2.py shingle_hash_wrapper.py udf.py ../test/integration_test/wet_file_utils.py union_find.py two_phase_partition_aware_union_find.py driver_memory_diagnostics.py
aws s3 cp dependencies.zip s3://text-deduplication-740959772378/scripts/
```
-j strips the directory path so all files end up at the zip root.

Step 6.2 if you want to delete your cache data
```
aws s3 rm s3://text-dedupe-benchmark/development/ --recursive
```

Step 7: Run Your Benchmark
From SSH session:
 use zip file
```
spark-submit \
  --master yarn \
  --py-files s3://text-deduplication-740959772378/scripts/dependencies.zip \
  --jars s3://text-deduplication-740959772378/scripts/minhash-udf_2.12-0.1.jar \
  --packages graphframes:graphframes:0.8.3-spark3.5-s_2.12 \
  --conf spark.sql.execution.arrow.maxRecordsPerBatch=10000 \
  --num-executors 4 \
  --executor-cores 4 \
  --executor-memory 14g \
  --driver-memory 12g \
  --conf spark.sql.shuffle.partitions=1000 \
  --conf spark.hadoop.fs.s3a.signing-algorithm="" \
  --conf spark.hadoop.fs.s3a.aws.credentials.provider=com.amazonaws.auth.DefaultAWSCredentialsProviderChain \
  --deploy-mode client \
  s3://text-deduplication-740959772378/scripts/spark_deduplication_test.py development
```

For 100 of WET files, increase partition count
Cores = 4 node * 8 vCore = 32
RAM = 4 ndoe * 61 GiB = 240gb

Available cores per node: 8 - 1 = 7
  Total usable cores: 7 × 4 = 28
  Executors at 4 cores: 28 / 4 = 7 executors

RAM for each node:
- 2 executors × (executor_memory + 4g overhead) ≤ ~59 GB (61 - 2 OS)
  executor_memory + 4 ≤ 29
  executor_memory ≤ 25g
```
spark-submit \
  --master yarn \
  --py-files s3://text-deduplication-740959772378/scripts/dependencies.zip \
  --jars s3://text-deduplication-740959772378/scripts/minhash-udf_2.12-0.1.jar \
  --packages graphframes:graphframes:0.8.3-spark3.5-s_2.12 \
  --conf spark.sql.execution.arrow.maxRecordsPerBatch=10000 \
  --num-executors 7 \
  --executor-cores 4 \
  --executor-memory 24g \
  --driver-memory 12g \
  --conf spark.sql.shuffle.partitions=50 \
  --conf spark.hadoop.fs.s3a.signing-algorithm="" \
  --conf spark.hadoop.fs.s3a.aws.credentials.provider=com.amazonaws.auth.DefaultAWSCredentialsProviderChain \
  --deploy-mode client \
  s3://text-deduplication-740959772378/scripts/spark_deduplication_test.py validation
```

where --deploy-mode cluster is to run the driver on EMR, not laptop.

For 1000 of WET files, increase partition count
Use 2 nodes of 8xlarge
```
spark-submit \
  --master yarn \
  --py-files s3://text-deduplication-740959772378/scripts/dependencies.zip \
  --jars s3://text-deduplication-740959772378/scripts/minhash-udf_2.12-0.1.jar \
  --conf spark.sql.execution.arrow.maxRecordsPerBatch=10000 \
  --num-executors 13 \
  --executor-cores 4 \
  --executor-memory 27g \
  --driver-memory 16g \
  --driver-cores 2
  --conf spark.sql.shuffle.partitions=1000 \
  --conf spark.network.timeout=800s \
  --conf spark.shuffle.io.connectionTimeout=600s \
  --conf spark.executor.extraJavaOptions="-XX:+UseG1GC -XX:MaxGCPauseMillis=200 -XX:+HeapDumpOnOutOfMemoryError -XX:HeapDumpPath=/tmp/executor_heap_%p.hprof" \
  --conf spark.yarn.maxAppAttempts=1 \
  --conf spark.shuffle.service.enabled=true \
  --conf spark.dynamicAllocation.enabled=false \
  --conf spark.hadoop.fs.s3a.signing-algorithm="" \
  --conf spark.hadoop.fs.s3a.aws.credentials.provider=com.amazonaws.auth.DefaultAWSCredentialsProviderChain \
  --conf spark.executor.memoryOverhead=5g \
  --deploy-mode cluster \
  s3://text-deduplication-740959772378/scripts/spark_deduplication_test.py production_proof
```
Why spark.dynamicAllocation.shuffleTracking.enabled=false 
and spark.shuffle.service.enabled=true
we want to store to external shuffle storage even if an executor ded as a result of dynamic allocation.

I removed these for now due to too aggresive timeout:
```
--conf spark.shuffle.registration.timeout=120s \
--conf spark.shuffle.io.maxRetries=6 \
--conf spark.shuffle.io.retryWait=30s \
```

For 9000 WET files, 
with 9 of r5ad.8xlarge,
* 288 vCPU (9 × 32)
* 2196 GB RAM (9 × 244 GB)
```
spark-submit \
  --master yarn \
  --py-files s3://text-deduplication-740959772378/scripts/dependencies.zip \
  --jars s3://text-deduplication-740959772378/scripts/minhash-udf_2.12-0.1.jar \
  --conf spark.sql.execution.arrow.maxRecordsPerBatch=10000 \
  --num-executors 54 \
  --executor-cores 4 \
  --executor-memory 28g \
  --driver-memory 48g \
  --conf spark.sql.shuffle.partitions=9000 \
  --conf spark.network.timeout=1200s \
  --conf spark.shuffle.io.connectionTimeout=600s \
  --conf spark.executor.extraJavaOptions="-XX:+UseG1GC -XX:MaxGCPauseMillis=200" \
  --conf spark.memory.offHeap.enabled=true \
  --conf spark.memory.offHeap.size=2g \
  --conf spark.yarn.maxAppAttempts=1 \
  --conf spark.shuffle.service.enabled=true \
  --conf spark.dynamicAllocation.enabled=false \
  --conf spark.hadoop.fs.s3a.signing-algorithm="" \
  --conf spark.hadoop.fs.s3a.aws.credentials.provider=com.amazonaws.auth.DefaultAWSCredentialsProviderChain \
  --conf spark.executor.memoryOverhead=10g \
  --deploy-mode cluster \
  s3://text-deduplication-740959772378/scripts/spark_deduplication_test.py scale_proof
```

After optimized by scala and Cython:
```
spark-submit \
  --master yarn \
  --py-files s3://text-deduplication-740959772378/scripts/dependencies.zip \
  --jars s3://text-deduplication-740959772378/scripts/minhash-udf_2.12-0.1.jar \
  --conf spark.sql.execution.arrow.maxRecordsPerBatch=30000 \
  --num-executors 63 \
  --executor-cores 4 \
  --executor-memory 27g \
  --driver-memory 32g \
  --conf spark.sql.shuffle.partitions=27000 \
  --conf spark.network.timeout=1200s \
  --conf spark.shuffle.io.connectionTimeout=600s \
  --conf spark.executor.extraJavaOptions="-XX:+UseG1GC -XX:MaxGCPauseMillis=200" \
  --conf spark.memory.offHeap.enabled=true \
  --conf spark.memory.offHeap.size=2g \
  --conf spark.yarn.maxAppAttempts=1 \
  --conf spark.shuffle.service.enabled=true \
  --conf spark.dynamicAllocation.enabled=false \
  --conf spark.hadoop.fs.s3a.signing-algorithm="" \
  --conf spark.shuffle.io.maxRetries=10 \
  --conf spark.shuffle.io.retryWait=30s \
  --conf spark.task.maxFailures=8 \
  --conf spark.hadoop.fs.s3a.aws.credentials.provider=com.amazonaws.auth.DefaultAWSCredentialsProviderChain \
  --conf spark.executor.memoryOverhead=5g \
  --deploy-mode cluster \
  s3://text-deduplication-740959772378/scripts/spark_deduplication_test.py scale_proof
```
Note:
- maxRecordsPerBatch increased from 10k to 30k to make it one round of arrow serialization.
- partitions increased from 9k to 27k to deal with stragglers
- Do not add spark.driver.extraJavaOptions in this config. Do it in EMR config

until creating df_with_partition,
this was fine:
num-executors=63, memoryOverhead=6g,executor-memory=24g.

For 90k WET files, 
with 64 of r5ad.8xlarge,
* 2048 vCPU (64 × 32)
* 16TB GB RAM (64 × 256 GB)

- reserve 1 core per node for YARN/OS overhead:
  Available cores per node: 32 - 1 = 31
  Total usable cores: 31 × 64 = 1984
  Executors at 4 cores: 1984 / 4 = 496 executors

- Also, one node is the master (driver + ResourceManager), 
  so one node is master + 63 core nodes.
  Usable cores: 31 cores / 4 = 7 executors (3 cores wasted)
  For master node: 30 cores / 4 = 7 executors (2 cores wasted)
  7 executor per node * 64 node = 448 executors
  
- For memory
  - 7 executors × (executor_memory + 5g overhead) ≤ ~254 GB (256 - 2 OS)
  executor_memory + 5 ≤ 36.3
  executor_memory ≤ 31g
```
spark-submit \
  --master yarn \
  --py-files s3://text-deduplication-740959772378/scripts/dependencies.zip \
  --jars s3://text-deduplication-740959772378/scripts/minhash-udf_2.12-0.1.jar \
  --conf spark.sql.execution.arrow.maxRecordsPerBatch=30000 \
  --num-executors 441 \
  --executor-cores 4 \
  --executor-memory 27g \
  --driver-memory 58g \
  --conf spark.sql.shuffle.partitions=90000 \
  --conf spark.network.timeout=1200s \
  --conf spark.shuffle.io.connectionTimeout=600s \
  --conf spark.executor.extraJavaOptions="-XX:+UseG1GC -XX:MaxGCPauseMillis=200" \
  --conf spark.yarn.maxAppAttempts=1 \
  --conf spark.shuffle.service.enabled=true \
  --conf spark.dynamicAllocation.enabled=false \
  --conf spark.hadoop.fs.s3a.signing-algorithm="" \
  --conf spark.shuffle.io.maxRetries=10 \
  --conf spark.shuffle.io.retryWait=30s \
  --conf spark.task.maxFailures=8 \
  --conf spark.hadoop.fs.s3a.aws.credentials.provider=com.amazonaws.auth.DefaultAWSCredentialsProviderChain \
  --conf spark.executor.memoryOverhead=5g \
  --deploy-mode cluster \
  s3://text-deduplication-740959772378/scripts/spark_deduplication_test.py full_corpus
```
- partitions=27000 if just processing input data from WET files.
- --conf spark.memory.offHeap.size=2g, --conf spark.memory.offHeap.enabled=true \ removed

How to save your executor log file.
```
yarn logs -applicationId application_1770437151065_0002 > /tmp/application_1770437151065_0002_executor.txt

aws s3 cp /tmp/application_1770437151065_0002_executor.txt s3://text-deduplication-740959772378/log/application_1770437151065_0002_executor.txt
```

How to find driver log
```
yarn logs -applicationId application_1770437151065_0002 -log_files stdout 2>/dev/null > /tmp/application_1770437151065_0002_driver.txt

aws s3 cp /tmp/application_1770437151065_0002_driver.txt s3://text-deduplication-740959772378/log/application_1770437151065_0002_driver.txt
```

then in your local,
```
aws s3 cp s3://text-deduplication-740959772378/log/application_1770252307636_0002.txt .
```

How to find Resource Manager logs
```
ls /var/log/hadoop-yarn/
```
to find logs 
ex) hadoop-yarn-resourcemanager-ip-172-31-47-166.ec2.internal.log

then

```
grep -a -i "lost\|unhealthy\|decommission\|expired" /var/log/hadoop-yarn/hadoop-yarn-resourcemanager-ip-172-31-47-166.ec2.internal.log | tail -20
```

How to cleanup:
before running terraform destroy, save your spark UI result.
```
aws s3 cp /var/log/spark/apps/application_* s3://text-deduplication-740959772378//application_*
```
then exit and in your macbook terminal run:
```
aws s3 cp s3://text-deduplication-740959772378/application_* ~/Downloads/application_*
```
```
terraform destroy
```

Step 8: If you use ICEBERG
```
spark-submit \
  --master yarn \
  --executor-memory 16g \
  --driver-memory 4g \
  --deploy-mode client \
   s3://text-deduplication-740959772378/scripts/iceberg_setup_test.py development
```

Step9: How to save your sark history UI

```
hdfs dfs -cat /var/log/spark/apps/application_1770437151065_0002_1 > ~/application_1770437151065_0002_1
aws s3 cp application_1770437151065_0002_1 s3://text-deduplication-740959772378/log/application_1770437151065_0002_1
```
then in you macbook,
```
aws s3 cp s3://text-deduplication-740959772378/application_1768106632592_0001_1 ~/Downloads
```
Step 10: How to monitor

Check specific stages:
http://localhost:4040/stages/stage/?id=12&attempt=0

check all applicaiton IDs:
```
yarn application -list -appStates ALL
```

check application logs
```
yarn logs -applicationId
```

check driver log without history server
```
yarn logs -applicationId application_1775424908785_0001 -log_files stdout -size -10000 2>/dev/null | grep -A5 Step | tail -50
```

How to kill yarn application

```
yarn application -kill {you application_id}
```

How to check your shuffle storage usage
```
# From your local machine - add key to agent first
ssh-add ./emr-dedupe-key.pem

# Then SSH to master with agent forwarding (-A)
ssh -A -i ./emr-dedupe-key.pem hadoop@<master-public-ip>

# From master, now you can reach core nodes
yarn node -list 2>/dev/null
ssh ip-172-31-37-206.ec2.internal "df -h | grep mnt"
```

## EMR ssh trouble shooting

### How to extract driver memoey stags/logs
```
# 1. Upload diagnostic files to S3
CLUSTER_ID=$(cat /mnt/var/lib/info/job-flow.json | python3 -c "import sys,json; print(json.load(sys.stdin)['jobFlowId'])")
S3_PREFIX="s3://text-dedupe-benchmark/heapdumps/${CLUSTER_ID}/$(date +%Y%m%dT%H%M%S)"
INSTANCE_ID=$(ec2-metadata -i | awk '{print $2}')

aws s3 cp /tmp/ "${S3_PREFIX}/" --recursive --exclude "*" --include "driver_heap_*.hprof"
aws s3 cp /tmp/ "${S3_PREFIX}/" --recursive --exclude "*" --include "driver_gc_*.log"
aws s3 cp /tmp/ "${S3_PREFIX}/" --recursive --exclude "*" --include "driver_heap_histo_*.log"
aws s3 cp /tmp/ "${S3_PREFIX}/" --recursive --exclude "*" --include "driver_nmt_*.log"

# 2. Extract memory log from YARN AM stdout
APP_ID=$(yarn application -list -appStates FINISHED,FAILED,KILLED 2>/dev/null \
    | grep -i spark | tail -1 | awk '{print $1}')
yarn logs -applicationId "${APP_ID}" -log_files stdout -am 1 2>/dev/null \
    | grep "\[DRIVER MEM\]\|\[DRIVER DIAG\]\|Setting job description\|Phase [0-9]" \
    > /tmp/driver_mem.log

aws s3 cp /tmp/driver_mem.log "${S3_PREFIX}/${INSTANCE_ID}_driver_mem.log"

# 3. Generate histogram from .hprof (only if OOMed)
jmap -histo /tmp/driver_heap.hprof >> /tmp/driver_heap_histo.txt

# 4. Run diagnosis
python3 /usr/local/bin/translate_driver_diagnostic_logs.py \
    --heap /tmp/driver_heap_histo.txt \
    --gc /tmp/driver_gc.log \
    --mem /tmp/driver_mem.log \
    > /tmp/diagnosis_report.txt

cat /tmp/diagnosis_report.txt

# 5. Upload report
aws s3 cp /tmp/diagnosis_report.txt "${S3_PREFIX}/${INSTANCE_ID}_diagnosis_report.txt"
```

### How to check HDDS application logs
```
hdfs dfs -ls /var/log/spark/apps/
```

edit spark-history-server .sh file
```
sudo vim /etc/spark/conf/spark-env.sh
```

Note: you neeed this to increase log size:
```
export SPARK_DAEMON_MEMORY=4g
```

restart soak history
```
sudo systemctl restart spark-history-server
```

Get the list of log file for RM (resource manager)
```
ls /var/log/hadoop-yarn/
```

Check resource maneger logs for disk space issue during shuffle.
```
sudo grep -A5 -B5 "application_1767582478981_0002" /var/log/hadoop-yarn/hadoop-yarn-resourcemanager-ip-172-31-46-228.ec2.internal.log.2026-01-05-05
```

How to check driver log. Note: resource manager != driver.
```
yarn logs -applicationId <app_id> 
```
gives you both driver and executor logs

so if we want only driver log,
```
yarn logs -applicationId <app_id> -am 1
```
where am means application manager and number means attempt.


## How to view downloaded spark history server locally

#### One time setup
```
export SPARK_HISTORY_OPTS="-Dspark.history.fs.logDirectory=file:///Users/kenichiobata/src/llm_trainining/spark_history_logs"
```

check which spark version
```
SPARK_HOME=$(python -c "import pyspark; import os; print(os.path.dirname(pyspark.__file__))")
echo $SPARK_HOME
```

Create config file
```
echo "spark.history.fs.logDirectory file:///Users/kenichiobata/src/llm_trainining/spark_history_logs" > /tmp/spark-history.conf
```

#### After one time setup is done:

Stop/restart when it's done or you want to start fresh
```
$SPARK_HOME/sbin/stop-history-server.sh
```

Download history server files
```
aws s3 sync s3://text-dedupe-benchmark/spark-history/ ~/src/llm_trainining/spark_history_logs/
```

Make sure your spark history log shows this before downloading
```
yarn application -status <app_id> | grep "Final-State"
```
Should show: Final-State : SUCCEEDED


Start runnign history server with more memory (4GB heap)
```
SPARK_DAEMON_MEMORY=4g $SPARK_HOME/sbin/start-history-server.sh --properties-file /tmp/spark-history.conf
```

### On EMR, how to debug history server log
```
cat /var/log/spark/spark-history-server.out
```


## pyspark common errors

- suddenly application shutdown manager was called.
  - check ```df -h``` to make sure you have enough disk space
  - check ```yarn node -list -all``` to make sure nodes are healthy
  ex)
WARNING: YARN_CONF_DIR has been replaced by HADOOP_CONF_DIR. Using value of YARN_CONF_DIR.
2026-01-05 06:13:11,814 INFO client.DefaultNoHARMFailoverProxyProvider: Connecting to ResourceManager at ip-172-31-46-228.ec2.internal/172.31.46.228:8032
2026-01-05 06:13:11,969 INFO client.AHSProxy: Connecting to Application History server at ip-172-31-46-228.ec2.internal/172.31.46.228:10200
Total Nodes:4
         Node-Id             Node-State Node-Http-Address       Number-of-Running-Containers
ip-172-31-37-80.ec2.internal:8041               RUNNING ip-172-31-37-80.ec2.internal:8042                                  0
ip-172-31-37-254.ec2.internal:8041              RUNNING ip-172-31-37-254.ec2.internal:8042                                 0
ip-172-31-42-11.ec2.internal:8041               RUNNING ip-172-31-42-11.ec2.internal:8042                                  0
ip-172-31-47-158.ec2.internal:8041              RUNNING ip-172-31-47-158.ec2.internal:8042                                 0
[hadoop@ip-172-31-46-228 ~]$ 
  - check ```sudo grep -A5 -B5 "UNHEALTHY" /var/log/hadoop-yarn/hadoop-yarn-resourcemanager-ip-172-31-46-228.ec2.internal.log.2026-01-05-05``` 
  to search resource manager <> driver's issue with your keyword.

- [NOT_COLUMN_OR_STR] Argument `col` should be a Column or str, got list.
  -  The error was caused by PySpark function overrides:
  - from pyspark.sql.functions import * overwrote Python's built-in
  max(), min(), sum() functions
  - When calling max(partition_counts) with a Python list, PySpark's
  max() expected a Column object
  - This caused the misleading [NOT_COLUMN_OR_STR] error message
      - Fixed function override conflicts - Used builtin_max(),
      builtin_min(), builtin_sum() instead of overridden function
## Terraform trouble shooting.

### terraform destroy
- An error occurred (DependencyViolation) when calling the DeleteSecurityGroup operation: resource sg-06e7dd56aad4ff534 has a dependent object
    - First check child (=executor EC2's security group deoending on parent (=driver) security group.)
        - ```aws ec2 describe-security-groups --filters "Name=ip-permission.group-id,Values=sg-06e7dd56aad4ff534" --query 'SecurityGroups[*].[GroupId,GroupName]' --output table```
        - then ```aws ec2 delete-security-group --group-id {response from above step}```
        - then delete master ```aws ec2 delete-security-group --group-name text-dedupe-benchmark-master-stg```
    - alternatively, try if this works for clean up secutiry group
        - ```# Delete the security groups again
            aws ec2 delete-security-group --group-name text-dedupe-benchmark-core-sg
            aws ec2 delete-security-group --group-name text-dedupe-benchmark-master-sg
          ```
        - 


# math behind

1.128 sampling called min hash
We take 128 samples of min hash where hash is based on N gram tokens and we take 128 based on different seeds.

2.bands/bucketing (= partition pruning )
We partition 128 samples into partitions called bands. This is based on reasoning that same signature falls into the same band/bucketing so we only need to compare hash in the same bucket.
With this, we don't need to brute force 128 sampels for doc against another doc.

WRONG Understanding:
"We only compare 8 values (one band) to determine similarity"

CORRECT Understanding:
"We use bands to quickly filter 10B × 10B pairs down to maybe 1M pairs,
 then we compare all 128 values for those 1M candidate pairs"

Note:
- Note that order is preserved since 128 samples have to be based on same seeds.
- When we compare 8 sample min hash per partition, we further create hash(tuple(8 sample min hash)).
    - By doing this, we can do full eact match and this is memory efficient than 8 string concatenation.

3.After at least one bucket matches among two doc, run full 128 sample min hash comparison.
This will compare x % match out of 128 samples, then if it exceeds user parameter of threashold %, 
these two docs are considered to be near duplicate.


# FAQ
- does this dedupe take care of transitive duplication? 
  For example, doc1-doc2 are dup, doc2-doc6 are also dup. Will doc 6 also deduped?
  - Yes, that's because similar docs based on MIN shingles falls under the same  partitions. So one run of dedupe SQL is sufficient. 
  In other words, we don't need to run iterative SQL to detect and dedupe duplicates.
- [Empty Spaces in Shingles] does it skip empty string?
  - Shingles do include spaces. They're not skipped.
  - Spaces help accuracy because they capture word boundaries.

- [Character-level vs Word-level Tokenization] it seems don't care about length of a word. For examples, if ngram=3, then if a word is "hello", shingles are = ['hel',''ell,'llo']. Why is this approach better than tokenizing word by word? Is it still accurate because we run these minhash shingles 64 ~ 128 times ?
  - At initial glance, tokenization seems more accurate, but actually, Character n-grams are better. Because of a few reasons:
    - 1) Typo Resistance: typo such as "helo" vs "hello" share the same shingles.
    - 2) Near-Duplicate Detection. Out goal is ""Near"" duplicates. So if we go with tokenization approach, past tense and present tense are categorized as different word. But in ""Near"" duplication, we want to categorize them near duplicates.
- Do you do text normalization? 
  - we apply Lowercase.
  - we do NOT apply Remove articles by default. We provide this as parameter. Remove articles gives minimum impact and it can even fause false positives.
    - ex) 
    Without normalization - correctly different
    "The Who is a band"
    "Who is a band member"

    With article removal - incorrectly similar!
    "Who is band"
    "Who is band member"
    - impact of removing articles only reduce duplicates by 0.01 %

- how do you know mh3 is accurate?

We tested both builtin oython hash vs mh3. 
The result is very similar dedupe rate.

| Metric | Python UDF (`builtin_hash`) | Cython UDF (MurmurHash3) |
|---|---|---|
| Total docs | 29,076 | 29,076 |
| Duplicates | 355 | 358 |
| Dedup rate | 1.22% | 1.23% |
| Similar pairs | 17,906 | 1,387 |

- What does builtin_hash's high similar pair but low duplicates mean compared to mh3?

This means builtin python hash has false positives due to hash collision.
builtin_hash has poor distribution on short strings (9-char shingles), causing hash collisions between unrelated shingles. These collisions inflate MinHash signature similarity, generating thousands of false candidate pairs that must be processed by downstream steps (connected components).

ex)
Document A shingles: "the quick", "he quick ", "e quick b", ...
Document B shingles: "something", "completely", "different", ...

With builtin_hash (poor distribution on 9-char strings):
  hash("the quick") = 0x7F3A    ← collision!
  hash("something") = 0x7F3A    ← same hash for different shingle!
  
  MinHash sees: "these documents share a shingle" (they don't)
  Signature similarity: inflated → 0.92 (above 0.9 threshold)
  → counted as similar pair (false positive)

With MurmurHash3 (better distribution):
  mmh3("the quick") = 0x7F3A2B01
  mmh3("something") = 0xE4C81D55    ← different hash, correct
  
  MinHash sees: "these documents don't share this shingle"
  Signature similarity: accurate → 0.12 (below threshold)
  → not counted

- if A,B are similar and B, D are similar, isn't expected behavior to catch A,D are similar just like builtin_hash found?

Not necessarily. Jaccard similarity is not transitive:
A and B share 92% of shingles → similar (above 0.9)
B and D share 91% of shingles → similar (above 0.9)
A and D share ???% of shingles → could be anything

But it doesn't matter — connected components handles transitivity

MurmurHash3 finds:     (A,B) and (B,D)
Connected components:   A — B — D  → all in same group
Result: A,B,D are all duplicates of representative A

The explicit (A,D) pair found by builtin_hash is unnecessary.
A and D end up in the same group through B.
So MurmurHash3's 1,387 pairs are sufficient — connected components discovers all transitive relationships. 

# scala setup
First setup venv for your project.
```
brew install sbt
```
this will download openjdk as well.
In analogy, sbt is a venv, pip combined from Python view.

sbt --version to confirm
java --version to check which JDK version it installed.

Start your local session in scala JVM
```
spark-shell
```
this is equivalent to ```pyspark``` shell

1.create your build.sbt file
Note that 
- build.sbt    → JAR (just src code, ~KB)
- assembly.sbt    → fat JAR (src code + all dependencies, ~MB)

in this project, because EMR already has setup, we don't need assembly.sbt

2.Build dependencies
```
sbt update
```

3.Run your scala code:
If you're outside of sbt shell,
```
sbt run {file}.scala
```

If you're in the sbt shell,
```
run {file}.scala
```
4.compile test
```
sbt compile
```
this does not create JAR file yet.

5. Unit test
```
sbt test
```
6.compile to JAR
```
sbt package
```


# Cython setup
## Building the Cython shingle hash module

1. Download MurmurHash3 C source:
From root, go to 
```cd /src/cython_minhash```

then,
```bash
   pip download mmh3 --no-binary :all:
   tar xzf mmh3-*.tar.gz
   cp mmh3-*/src/mmh3/murmurhash3.c .
   cp mmh3-*/src/mmh3/murmurhash3.h .
   rm -rf mmh3-*/  mmh3-*.tar.gz
```

2. Build the Cython extension:
```bash
   cd cython_minhash
   python setup.py build_ext --inplace
```
```

And in `.gitignore`:
```
cython_minhash/MurmurHash3.c
cython_minhash/MurmurHash3.h
cython_minhash/*.so
cython_minhash/build/

2.Compile
```
python setup.py build_ext --inplace
```

3.terraform code change
You need to add sudo yum install -y python3-devel in terraform file.
This is because Python itself is installed on EMR, but the .h files needed to compile against Python's C API are a separate package:

python3 package (already on EMR):
  /usr/bin/python3              ← the interpreter
  /usr/lib/python3/             ← .py modules

python3-devel package (not on EMR by default):
  /usr/include/python3.9/Python.h    ← C headers
  /usr/include/python3.9/object.h
  /usr/include/python3.9/pymem.h
  /usr/lib/libpython3.9.so           ← shared library for linking