# How to develop this repo
## create virtual env
one-off
```
python3 -m venv venv
```

Every time
```
source venv/bin/activate
```
You should see (venv) after actuvation.
# Install PySpark locally
```
pip install -r requirements.txt
```
# Run Spark locally with 4GB RAM
```
spark-submit --driver-memory 4g --executor-memory 4g src/spark_deduplication_vanilla.py
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
spark-submit --driver-memory 4g --executor-memory 4g src/spark_partition_aware_deduplicattion_v2.py
```

Or run locally for testing
```
python src/spark_partition_aware_deduplicattion_v2.py
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

```
terraform apply 
```

Note - you need to create your own terraform.tfvars file looks like this:
```
cluster_name   = "" # EMR cluster name
subnet_id      = "subnet-xxxxxx"          # Your subnet ID
vpc_id         = "vpc-xxxxxx"             # Your VPC ID
scripts_bucket = ""        # Your S3 bucket name
```

How to cleanup
```
terraform destroy
```

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