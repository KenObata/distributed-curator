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


one-off
```
python3 -m venv venv
```

Every time
```
deactivate
source venv/bin/activate
```
You should see (venv) after actuvation.
# Install PySpark locally
```
pip install -r requirements_emr.txt
```
# Run Spark locally with 4GB RAM

Note that development is an argument.
Choose from development, validation, production_proof, scale_proof
```
spark-submit --driver-memory 4g --executor-memory 4g \
--packages graphframes:graphframes:0.8.3-spark3.5-s_2.12 \
--py-files ../../src/spark_partition_aware_deduplicattion_v2.py,../../src/spark_utils.py ../../test/spark_deduplication_test.py development
```
- spark_deduplication.py - Complete implementation for web-scale deduplication
- common_crawl_explorer.py: PoC

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

Next, run partition aware text de-duplication
```
spark-submit \
  --master yarn \
  --deploy-mode cluster \
  --driver-memory 8g \
  --executor-memory 16g \
  --num-executors 10 \
  --conf spark.sql.shuffle.partitions=1000 \
  spark_partition_aware_deduplicattion_v2.py.py
```
Or run it on a single machine
```
spark-submit --driver-memory 4g --executor-memory 4g test/spark_deduplication_test.py
```


# How to unit/integration test

## Unit Test
```
pytest --log-cli-level=INFO test/spark_partition_aware_deduplicattion_v2_unit_test.py::TestDocumentSimilarity -v
```

## Integration Test

Run only a sample
```
pytest --log-cli-level=INFO test/spark_partition_aware_deduplicattion_v2_integration_test.py::test_integration_small_samples -s
```

Run only a specific test
```
pytest --log-cli-level=INFO test/spark_partition_aware_deduplicattion_v2_integration_test.py::test_integration_commoncrawl_sample -s
```

## local UI monitoring
http://192.168.100.130:4040/

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
terraform apply -var="instance_strategy=on-demand" -var="wet_file_scale=1k"
```

Note - you need to create your own terraform.tfvars file looks like this:
```
cluster_name   = "" # EMR cluster name
subnet_id      = "subnet-xxxxxx"          # Your subnet ID
vpc_id         = "vpc-xxxxxx"             # Your VPC ID
scripts_bucket = ""        # Your S3 bucket name
```

upon succes, you should see terminal output:
cluster_id = "j-xxxx"
master_public_dns = "ec2-xx-xxx-xxx-xx.compute-1.amazonaws.com"
private_key_file = "./your-emr-key.pem"
private_key_pem = <sensitive>
spark_submit_example = "spark-submit --master yarn --deploy-mode cluster s3://text-deduplication/scripts/deduplication_benchmark.py"
spark_ui_url = "http://ec2-34-203-229-18.compute-1.amazonaws.com:4040"
ssh_command = "ssh -i ./your-emr-key.pem hadoop@ec2-34-203-229-18.compute-1.amazonaws.com"
yarn_ui_url = "http://ec2-34-203-229-18.compute-1.amazonaws.com:8088"

Step2:
then run these
```
terraform output master_public_dns
```

Step3: upload your requirements.txt to S3:

Then from SSH session on the master node:

```
ssh -i ./your-emr-key.pem hadoop@<master-public-dns>
aws s3 cp s3://your-scripts-bucket/scripts/requirements_emr.txt .
sudo pip3 install --ignore-installed --no-cache-dir --no-deps -r requirements_emr.txt
```


Step4: Exit ssh, and on your macbook, install YARN(8088), Spark UI (4040)
Find YAN host name - run
```
hostname -f
```
Static port forwarding
```
ssh -i ./your-emr-key.pem \
  -L 8088:localhost:8088 \
  -L 18080:localhost:18080 \
  -L 8042:localhost:8042 \
  -L 19888:localhost:19888 \
  -L 18080:localhost:18080 \
  -L 20888:ip-172-31-38-47.ec2.internal:20888 \
  -L 4040:<EMR driver (master node) hostname>:4040 \
  hadoop@<master-public-dns>
```

where 
- NodeManager: http://localhost:8042
- Job History: http://localhost:19888
- History Server http://localhost:18080/
- YAN Cluster mode's Spark UI
    - How to view UI: http://localhost:20888/proxy/application_xxx/

Dynamic port forwarding
```
ssh -i ./your-emr-key.pem \
  -D 18080 \
  hadoop@<master-public-dns>
```

where The -D flag only takes a port number, not a host:port mapping.

after this, on your macbook, run
```
/Applications/Google\ Chrome.app//Contents/MacOS/Google\ Chrome --proxy-server="socks5://localhost:8080"
```



or

```
yarn application -list
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
zip -r dependencies.zip spark_utils.py spark_partition_aware_deduplicattion_v2.py
aws s3 cp dependencies.zip s3://your-scripts-bucket/scripts/
```

Step 7: Run Your Benchmark
From SSH session:
 use zip file
```
spark-submit \
  --master yarn \
  --py-files s3://your-scripts-bucket/scripts/dependencies.zip \
  --packages graphframes:graphframes:0.8.3-spark3.5-s_2.12 \
  --conf spark.sql.execution.arrow.maxRecordsPerBatch=10000 \
  --num-executors 4 \
  --executor-cores 4 \
  --executor-memory 14g \
  --driver-memory 12g \
  --conf spark.sql.shuffle.partitions=30 \
  --conf spark.memory.offHeap.size=1g \
  --conf spark.hadoop.fs.s3a.signing-algorithm="" \
  --conf spark.hadoop.fs.s3a.aws.credentials.provider=com.amazonaws.auth.DefaultAWSCredentialsProviderChain \
  --deploy-mode client \
  s3://your-scripts-bucket/scripts/spark_deduplication_test.py development
```

For 100 of WET files, increase partition count
```
spark-submit \
  --master yarn \
  --py-files s3://your-scripts-bucket/scripts/dependencies.zip \
  --packages graphframes:graphframes:0.8.3-spark3.5-s_2.12 \
  --conf spark.sql.execution.arrow.maxRecordsPerBatch=10000 \
  --num-executors 8 \
  --executor-cores 4 \
  --executor-memory 28g \
  --driver-memory 12g \
  --conf spark.sql.shuffle.partitions=2000 \
  --conf spark.memory.offHeap.size=1g \
  --conf spark.hadoop.fs.s3a.signing-algorithm="" \
  --conf spark.hadoop.fs.s3a.aws.credentials.provider=com.amazonaws.auth.DefaultAWSCredentialsProviderChain \
  --deploy-mode cluster \
  s3://your-scripts-bucket/scripts/spark_deduplication_test.py validation
```

where --deploy-mode cluster is to run the driver on EMR, not laptop.

For 1000 of WET files, increase partition count
```
spark-submit \
  --master yarn \
  --py-files s3://your-scripts-bucket/scripts/dependencies.zip \
  --packages graphframes:graphframes:0.8.3-spark3.5-s_2.12 \
  --conf spark.sql.execution.arrow.maxRecordsPerBatch=10000 \
  --num-executors 32 \
  --executor-cores 4 \
  --executor-memory 28g \
  --driver-memory 24g \
  --conf spark.sql.shuffle.partitions=3000 \
  --conf spark.memory.offHeap.size=1g \
  --conf spark.yarn.maxAppAttempts=1 \
  --conf spark.hadoop.fs.s3a.signing-algorithm="" \
  --conf spark.hadoop.fs.s3a.aws.credentials.provider=com.amazonaws.auth.DefaultAWSCredentialsProviderChain \
  --deploy-mode cluster \
  s3://your-scripts-bucket/scripts/spark_deduplication_test.py production_proof
```

For 9000 WET files, 
```
spark-submit \
  --master yarn \
  --py-files s3://your-scripts-bucket/scripts/dependencies.zip \
  --packages graphframes:graphframes:0.8.3-spark3.5-s_2.12 \
  --conf spark.sql.execution.arrow.maxRecordsPerBatch=10000 \
  --num-executors 36 \
  --executor-cores 8 \
  --executor-memory 60g \
  --driver-memory 32g \
  --conf spark.sql.shuffle.partitions=10000 \
  --conf spark.default.parallelism=10000 \
  --conf spark.memory.offHeap.size=4g \
  --conf spark.yarn.maxAppAttempts=1 \
  --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
  --conf spark.kryoserializer.buffer.max=2048m \
  --conf spark.hadoop.fs.s3a.signing-algorithm="" \
  --conf spark.hadoop.fs.s3a.aws.credentials.provider=com.amazonaws.auth.Defau
ltAWSCredentialsProviderChain \
  --deploy-mode cluster \
  s3://your-scripts-bucket/scripts/spark_deduplication_test.py
scale_proof
```

How to cleanup:
before running terraform destroy, save your spark UI result.
```
aws s3 cp /var/log/spark/apps/application_* s3://your-scripts-bucket//application_*
```
then exit and in your macbook terminal run:
```
aws s3 cp s3://your-scripts-bucket/application_* ~/Downloads/application_*
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
   s3://your-scripts-bucket/scripts/iceberg_setup_test.py development
```

Step 9: How to monitor

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

How to kill yarn application

```
yarn application -kill {you application_id}
```

## EMR ssh trouble shooting
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
