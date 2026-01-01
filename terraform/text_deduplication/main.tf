# EMR Cluster for Text Deduplication Benchmark
# Based on configuration discussed for Spark-based MinHash LSH deduplication

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
    local = {
      source  = "hashicorp/local"
      version = "~> 2.0"
    }
  }
}

provider "aws" {
  region = "us-east-1"  # Same region as Common Crawl for no data transfer costs
}

# Variables
variable "cluster_name" {
  description = "Name of the EMR cluster"
  type        = string
  default     = "text-dedupe-benchmark"
}

variable "key_name" {
  description = "EC2 key pair name for SSH access"
  type        = string
  default     = "emr-dedupe-key"
}

# Generate SSH key pair
resource "tls_private_key" "emr_key" {
  algorithm = "RSA"
  rsa_bits  = 4096
}

resource "aws_key_pair" "emr_key" {
  key_name   = var.key_name
  public_key = tls_private_key.emr_key.public_key_openssh
}

# Save private key locally
resource "local_file" "private_key" {
  content         = tls_private_key.emr_key.private_key_pem
  filename        = "${path.module}/${var.key_name}.pem"
  file_permission = "0400"
}

variable "subnet_id" {
  description = "Subnet ID for the EMR cluster"
  type        = string
}

variable "vpc_id" {
  description = "VPC ID for security groups"
  type        = string
}

variable "scripts_bucket" {
  description = "S3 bucket for scripts and bootstrap actions"
  type        = string
  default     = "text-deduplication" 
}

variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

# Security Group for EMR
resource "aws_security_group" "emr_master" { # master means spark driver
  name        = "${var.cluster_name}-master-sg"
  description = "Security group for EMR master node"
  vpc_id      = var.vpc_id

  # SSH access
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]  # Restrict to your IP in production
  }

  # Spark History Server - commented out due to EMR security restrictions
  # EMR only allows port 22 to have public access (0.0.0.0/0)
  # Use SSH tunneling to access Spark History Server
  #ingress {
  #  from_port   = 18080
  #  to_port     = 18080
  #  protocol    = "tcp"
  #  cidr_blocks = ["0.0.0.0/0"]
  #}

  # Spark UI and YARN commented out because 
  # EMR doesn't allow security groups with public access to ports other than SSH (22).
  # Spark UI
  #ingress {
  #  from_port   = 4040
  #  to_port     = 4040
  #  protocol    = "tcp"
  #  cidr_blocks = ["0.0.0.0/0"]
  #}

  # YARN ResourceManager
  #ingress {
  #  from_port   = 8088
  #  to_port     = 8088
  #  protocol    = "tcp"
  #  cidr_blocks = ["0.0.0.0/0"]
  #}

  # Jupyter
  #ingress {
  #  from_port   = 8888
  #  to_port     = 8888
  #  protocol    = "tcp"
  #  cidr_blocks = ["0.0.0.0/0"]
  #}

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.cluster_name}-master-sg"
  }
}

resource "aws_security_group" "emr_core" { # core means spark executors
  name        = "${var.cluster_name}-core-sg"
  description = "Security group for EMR core nodes"
  vpc_id      = var.vpc_id

  # Allow all traffic within the cluster
  ingress {
    from_port = 0
    to_port   = 0
    protocol  = "-1"
    self      = true
  }

  ingress {
    from_port       = 0
    to_port         = 0
    protocol        = "-1"
    security_groups = [aws_security_group.emr_master.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.cluster_name}-core-sg"
  }
}

# IAM Role for EMR Service
resource "aws_iam_role" "emr_service_role" {
  name = "${var.cluster_name}-service-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "elasticmapreduce.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "emr_service_policy" {
  role       = aws_iam_role.emr_service_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEMRServicePolicy_v2"
}

# Additional EC2 permissions for EMR service role
resource "aws_iam_role_policy_attachment" "emr_service_ec2_full" {
  role       = aws_iam_role.emr_service_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2FullAccess"
}

# IAM Role for EC2 Instances (Instance Profile)
resource "aws_iam_role" "emr_ec2_role" {
  name = "${var.cluster_name}-ec2-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

# Policy for EC2 instances to access S3
resource "aws_iam_role_policy" "emr_ec2_policy" {
  name = "${var.cluster_name}-ec2-policy"
  role = aws_iam_role.emr_ec2_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowCommonCrawlAccess"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::commoncrawl",
          "arn:aws:s3:::commoncrawl/*"
        ]
      },
      {
        Sid    = "AllowScriptsBucketAccess"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::${var.scripts_bucket}",
          "arn:aws:s3:::${var.scripts_bucket}/*"
        ]
      },
      {
        Sid    = "AllowEMRLogging"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::aws-logs-*-${data.aws_caller_identity.current.account_id}",
          "arn:aws:s3:::aws-logs-*-${data.aws_caller_identity.current.account_id}/*"
        ]
      },
      {
        Sid    = "AllowCloudWatchMetrics"
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData"
        ]
        Resource = "*"
      },
      {
        Sid    = "AllowCloudWatchLogs"
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogGroups",
          "logs:DescribeLogStreams"
        ]
        Resource = "*"
      },
      {
        Sid    = "AllowDataBucketAccess"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::${var.scripts_bucket}-data-${data.aws_caller_identity.current.account_id}",
          "arn:aws:s3:::${var.scripts_bucket}-data-${data.aws_caller_identity.current.account_id}/*"
        ]
      }
    ]
  })
}

resource "aws_iam_instance_profile" "emr_ec2_profile" {
  name = "${var.cluster_name}-ec2-profile"
  role = aws_iam_role.emr_ec2_role.name
}

# Get current AWS account ID
data "aws_caller_identity" "current" {}

# Add S3 permissions to EMR_EC2_DefaultRole for bootstrap scripts and Common Crawl
resource "aws_iam_role_policy" "emr_ec2_default_s3_access" {
  name = "${var.cluster_name}-ec2-s3-access"
  role = "EMR_EC2_DefaultRole"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowScriptsBucketAccess"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket",
          "s3:PutObject"
        ]
        Resource = [
          "arn:aws:s3:::${var.scripts_bucket}-${data.aws_caller_identity.current.account_id}",
          "arn:aws:s3:::${var.scripts_bucket}-${data.aws_caller_identity.current.account_id}/*"
        ]
      },
      {
        Sid    = "AllowCommonCrawlAccess"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::commoncrawl",
          "arn:aws:s3:::commoncrawl/*"
        ]
      },
      {
        Sid    = "AllowEMRLogsBucket"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket",
          "s3:PutObject"
        ]
        Resource = [
          "arn:aws:s3:::${var.scripts_bucket}-emr-logs-${data.aws_caller_identity.current.account_id}",
          "arn:aws:s3:::${var.scripts_bucket}-emr-logs-${data.aws_caller_identity.current.account_id}/*"
        ]
      },
      {
        Sid    = "AllowDataBucketAccess"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::${var.scripts_bucket}-data-${data.aws_caller_identity.current.account_id}",
          "arn:aws:s3:::${var.scripts_bucket}-data-${data.aws_caller_identity.current.account_id}/*"
        ]
      }
    ]
  })
}

# S3 bucket for scripts and data (name must be globally unique)
resource "aws_s3_bucket" "scripts_bucket" {
  bucket = "${var.scripts_bucket}-${data.aws_caller_identity.current.account_id}"
  force_destroy = true # destroy including existing S3 files
}

# Wait for S3 bucket to propagate (eventual consistency)
resource "time_sleep" "wait_for_bucket" {
  depends_on      = [aws_s3_bucket.scripts_bucket]
  create_duration = "30s"
}

# S3 bucket for logs
resource "aws_s3_bucket" "emr_logs" {
  bucket = "${var.scripts_bucket}-emr-logs-${data.aws_caller_identity.current.account_id}"
  force_destroy = true
}

# S3 bucket for Iceberg data warehouse
resource "aws_s3_bucket" "data" {
  bucket = "${var.scripts_bucket}-data-${data.aws_caller_identity.current.account_id}"
  force_destroy = true
}

# useful for uploading local files
variable "scripts_source_src_dir" {
  description = "Path to the directory containing Python scripts"
  type        = string
  default     = "../../src"
}

variable "scripts_source_test_dir" {
  description = "Path to the directory containing Python scripts"
  type        = string
  default     = "../../test"
}

# Bootstrap action script
resource "aws_s3_object" "bootstrap_script" {
  bucket  = "${var.scripts_bucket}-${data.aws_caller_identity.current.account_id}"
  key     = "bootstrap/install_dependencies.sh"
  content = <<-EOF
    #!/bin/bash
    set -e
    
    echo "Installing Python dependencies..."
    sudo /usr/bin/pip3 install numpy mmh3 xxhash
    echo "Verifying installation..."
    python3 -c "import numpy; import mmh3; import xxhash; print('All packages installed successfully')"
    
    echo "Bootstrap complete!"
  EOF
}

# Upload requirements.txt to S3
resource "aws_s3_object" "requirements" {
  bucket = aws_s3_bucket.scripts_bucket.id
  key    = "scripts/requirements_emr.txt"
  source = "${path.module}/requirements_emr.txt"  # Local file path
  
  depends_on = [time_sleep.wait_for_bucket]
}

# Upload deduplication script to S3
resource "aws_s3_object" "dedup_script" {
  bucket = aws_s3_bucket.scripts_bucket.id
  key    = "scripts/spark_partition_aware_deduplicattion_v2.py"
  source = "${path.module}/${var.scripts_source_src_dir}/spark_partition_aware_deduplicattion_v2.py"
  
  depends_on = [time_sleep.wait_for_bucket]
}

# Upload integration test script to S3
resource "aws_s3_object" "integration_test_script" {
  bucket = aws_s3_bucket.scripts_bucket.id
  key    = "scripts/spark_deduplication_test.py"
  source = "${path.module}/${var.scripts_source_test_dir}/spark_deduplication_test.py"
  
  depends_on = [time_sleep.wait_for_bucket]
}

# Upload iceberg setup test script to S3
resource "aws_s3_object" "iceberg_setup_test_script" {
  bucket = aws_s3_bucket.scripts_bucket.id
  key    = "scripts/iceberg_setup_test.py"
  source = "${path.module}/${var.scripts_source_test_dir}/iceberg_setup_test.py"
  
  depends_on = [time_sleep.wait_for_bucket]
}

# EMR Cluster
variable "instance_strategy" {
  description = "Instance strategy: 'spot' for mixed spot/on-demand, 'on-demand' for all on-demand"
  type        = string
  default     = "spot"
  
  validation {
    condition     = contains(["spot", "on-demand"], var.instance_strategy)
    error_message = "Instance strategy must be either 'spot' or 'on-demand'."
  }
}

variable "bid_strategy" {
  description = "Bid strategy: 'default' for optimized pricing, 'peak-event' for maximum availability during high-demand periods"
  type        = string
  default     = "default"
  
  validation {
    condition     = contains(["default", "peak-event"], var.bid_strategy)
    error_message = "Bid strategy must be either 'default' or 'peak-event'."
  }
}

resource "aws_emr_cluster" "dedup_cluster" {
  name          = var.cluster_name
  release_label = "emr-7.12.0"
  applications  = ["Spark", "Hadoop", "Hive", "JupyterEnterpriseGateway", "Livy"]

  # service_role = aws_iam_role.emr_service_role.arn # permission error
  service_role = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/EMR_DefaultRole"

  ec2_attributes {
    subnet_id                         = var.subnet_id
    # emr_managed_master_security_group = aws_security_group.emr_master.id # let EMR manage its own security
    # emr_managed_slave_security_group  = aws_security_group.emr_core.id # let EMR manage its own security
    instance_profile                  = "arn:aws:iam::740959772378:instance-profile/EMR_EC2_DefaultRole"
    key_name                          = aws_key_pair.emr_key.key_name
    additional_master_security_groups = aws_security_group.emr_master.id
  }

  # Primary (Master) node - On-Demand
  master_instance_fleet {
    name = "Primary"
    
    target_on_demand_capacity = 1  # On-demand for stability
    target_spot_capacity      = 0  # No spot for master
    
    instance_type_configs {
      instance_type     = "r5.xlarge"
      weighted_capacity = 1
      
      ebs_config {
        size                 = 100
        type                 = "gp3"
        iops                 = 3000
        volumes_per_instance = 1
      }
    }
  }

  # Core nodes - Mixed spot/on-demand for reliability
  core_instance_fleet {
    name = "Core"
    
    target_on_demand_capacity = var.instance_strategy == "on-demand" ? 32 : 8   # Reduced on-demand for better cost
    target_spot_capacity      = var.instance_strategy == "on-demand" ? 0 : 24   # Increased spot for cost savings
    
    # Primary choice - r5a.xlarge (AMD - better spot availability)
    instance_type_configs {
      instance_type     = "r5a.xlarge"
      weighted_capacity = 1
      
      bid_price_as_percentage_of_on_demand_price = var.bid_strategy == "peak-event" ? 100 : 80  # Peak events need 100%
      
      ebs_config {
        size                 = 100
        type                 = "gp3"
        iops                 = 3000
        volumes_per_instance = 1
      }
    }
    
    # Fallback choice - r6g.xlarge (ARM Graviton2 - excellent spot availability)
    instance_type_configs {
      instance_type     = "r6g.xlarge"
      weighted_capacity = 1
      
      bid_price_as_percentage_of_on_demand_price = var.bid_strategy == "peak-event" ? 100 : 75  # Peak events need 100%
      
      ebs_config {
        size                 = 100
        type                 = "gp3"
        iops                 = 3000
        volumes_per_instance = 1
      }
    }
    
    # Additional high-scoring instance types for better spot placement
    instance_type_configs {
      instance_type     = "r6i.xlarge"
      weighted_capacity = 1
      
      bid_price_as_percentage_of_on_demand_price = var.bid_strategy == "peak-event" ? 100 : 80  # Peak events need 100%
      
      ebs_config {
        size                 = 100
        type                 = "gp3"
        iops                 = 3000
        volumes_per_instance = 1
      }
    }
    
    instance_type_configs {
      instance_type     = "r7g.xlarge"
      weighted_capacity = 1
      
      bid_price_as_percentage_of_on_demand_price = var.bid_strategy == "peak-event" ? 100 : 75  # Peak events need 100%
      
      ebs_config {
        size                 = 100
        type                 = "gp3"
        iops                 = 3000
        volumes_per_instance = 1
      }
    }
    
    # Additional option - r5.xlarge (Intel 5th gen - proven available)
    instance_type_configs {
      instance_type     = "r5.xlarge"
      weighted_capacity = 1
      
      bid_price_as_percentage_of_on_demand_price = var.bid_strategy == "peak-event" ? 100 : 85  # Peak events need 100%
      
      ebs_config {
        size                 = 100
        type                 = "gp3"
        iops                 = 3000
        volumes_per_instance = 1
      }
    }
    
  }

  bootstrap_action {
    name = "Install Python dependencies"
    path = "s3://${aws_s3_bucket.scripts_bucket.id}/bootstrap/install_dependencies.sh"
  }


  # Spark and YARN configurations
  configurations_json = jsonencode([
    {
      Classification = "spark-defaults"
      Properties = {
        "spark.default.parallelism"        = "2000"
        "spark.memory.fraction"            = "0.8"
        "spark.memory.storageFraction"     = "0.3"
        "spark.serializer"                 = "org.apache.spark.serializer.KryoSerializer"
        "spark.kryoserializer.buffer.max"  = "1024m"
        "spark.driver.memory"              = "12g"
        "spark.executor.memory"            = "12g"
        "spark.executor.cores"             = "4"
        "spark.dynamicAllocation.enabled"  = "false"
        "spark.sql.adaptive.enabled"       = "true"

        "spark.sql.catalog.glue_catalog": "org.apache.iceberg.spark.SparkCatalog",
        "spark.sql.catalog.glue_catalog.warehouse": "s3://${aws_s3_bucket.data.bucket}/iceberg/",
        "spark.sql.catalog.glue_catalog.catalog-impl": "org.apache.iceberg.aws.glue.GlueCatalog",
        "spark.sql.catalog.glue_catalog.io-impl": "org.apache.iceberg.aws.s3.S3FileIO",
        "spark.sql.extensions": "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
        "spark.eventLog.enabled": "true",
        "spark.eventLog.dir": "hdfs:///var/log/spark/apps",
        "spark.history.fs.logDirectory": "hdfs:///var/log/spark/apps"
      }
    },
    {
      Classification = "yarn-site"
      Properties = {
        "yarn.nodemanager.vmem-check-enabled" = "false"
        "yarn.nodemanager.pmem-check-enabled" = "false"
      }
    },
    {
      Classification = "spark-env"
      Properties = {}
      Configurations = [
        {
          Classification = "export"
          Properties = {
            "PYSPARK_PYTHON" = "/usr/bin/python3"
            "SPARK_DAEMON_MEMORY" = "4g"
          }
        }
      ]
    },
    # ICEBERG setting
    {
      "Classification": "iceberg-defaults",
      "Properties": {
        "iceberg.enabled": "true"
      }
    }
  ])

  log_uri = "s3://${aws_s3_bucket.emr_logs.id}/logs/"

  # Enable CloudWatch logging
  log_encryption_kms_key_id = null
  
  # Step concurrency (optional - allows multiple steps)
  step_concurrency_level = 1
  
  # Keep cluster running (set to true for step execution)
  keep_job_flow_alive_when_no_steps = true

  # Termination protection off for easy cleanup
  termination_protection = false

  tags = {
    Name        = var.cluster_name
    Purpose     = "text-deduplication-benchmark"
    Environment = "development"
  }

  depends_on = [
    aws_s3_object.bootstrap_script
  ]
}

# Outputs
output "cluster_id" {
  description = "EMR Cluster ID"
  value       = aws_emr_cluster.dedup_cluster.id
}

output "master_public_dns" {
  description = "Public DNS of the master node"
  value       = aws_emr_cluster.dedup_cluster.master_public_dns
}

output "spark_ui_url" {
  description = "Spark UI URL (requires SSH tunnel or security group access)"
  value       = "http://${aws_emr_cluster.dedup_cluster.master_public_dns}:4040"
}

output "yarn_ui_url" {
  description = "YARN ResourceManager UI URL"
  value       = "http://${aws_emr_cluster.dedup_cluster.master_public_dns}:8088"
}

output "spark_history_server_tunnel_command" {
  description = "SSH tunnel command to access Spark History Server (EMR security restriction)"
  value       = "ssh -i ${path.module}/${var.key_name}.pem -L 18080:${aws_emr_cluster.dedup_cluster.master_public_dns}:18080 hadoop@${aws_emr_cluster.dedup_cluster.master_public_dns}"
}

output "ssh_command" {
  description = "SSH command to connect to master node"
  value       = "ssh -i ${path.module}/${var.key_name}.pem hadoop@${aws_emr_cluster.dedup_cluster.master_public_dns}"
}

output "private_key_file" {
  description = "Path to the private key file"
  value       = "${path.module}/${var.key_name}.pem"
}

output "private_key_pem" {
  description = "Private key content (save this if needed)"
  value       = tls_private_key.emr_key.private_key_pem
  sensitive   = true
}

output "spark_submit_example" {
  description = "Example spark-submit command"
  value       = "spark-submit --master yarn --deploy-mode cluster s3://${var.scripts_bucket}/scripts/deduplication_benchmark.py"
}




# Glue Catalog
resource "aws_iam_role_policy" "emr_glue_access" {
  name = "emr-glue-iceberg-access"
  role = aws_iam_role.emr_ec2_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "glue:CreateDatabase",
          "glue:GetDatabase",
          "glue:GetDatabases",
          "glue:CreateTable",
          "glue:GetTable",
          "glue:GetTables",
          "glue:UpdateTable",
          "glue:DeleteTable",
          "glue:GetPartitions",
          "glue:BatchCreatePartition",
          "glue:BatchDeletePartition"
        ]
        Resource = [
          "arn:aws:glue:${var.region}:${data.aws_caller_identity.current.account_id}:catalog",
          "arn:aws:glue:${var.region}:${data.aws_caller_identity.current.account_id}:database/*",
          "arn:aws:glue:${var.region}:${data.aws_caller_identity.current.account_id}:table/*"
        ]
      }
    ]
  })
}
# Glue catalog database with lifecycle management
resource "aws_glue_catalog_database" "lineage" {
  name = "lineage"
  
  description = "Provenance tracking for LLM training data"
  
}