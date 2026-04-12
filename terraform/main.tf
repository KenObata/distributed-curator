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
  region = "us-east-1" # Same region as Common Crawl for no data transfer costs
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
  default     = "your-emr-key"
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

variable "subnet_ids" {
  description = "List of Subnet IDs for the EMR cluster (multi-AZ for better availability). Can pass single subnet as list."
  type        = list(string)
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

variable "text_dedupe_benchmark_bucket" {
  description = "S3 bucket for cached data and persisted spark history server"
  type        = string
  default     = "text-dedupe-benchmark"
}

resource "aws_s3_object" "spark_history_dir" {
  bucket       = var.text_dedupe_benchmark_bucket
  key          = "spark-history/"
  content_type = "application/x-directory"
  content      = ""
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
    cidr_blocks = ["0.0.0.0/0"] # Restrict to your IP in production
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
  bucket        = "${var.scripts_bucket}-${data.aws_caller_identity.current.account_id}"
  force_destroy = true # destroy including existing S3 files
}

# Wait for S3 bucket to propagate (eventual consistency)
resource "time_sleep" "wait_for_bucket" {
  depends_on      = [aws_s3_bucket.scripts_bucket]
  create_duration = "30s"
}

# S3 bucket for logs
resource "aws_s3_bucket" "emr_logs" {
  bucket        = "${var.scripts_bucket}-emr-logs-${data.aws_caller_identity.current.account_id}"
  force_destroy = true
}


# useful for uploading local files
variable "scripts_source_src_dir" {
  description = "Path to the directory containing Python scripts"
  type        = string
  default     = "../src"
}

variable "scripts_source_scala_dir" {
  description = "Path to the directory containing Python scripts"
  type        = string
  default     = "../target/scala-2.12"
}

variable "scripts_source_test_dir" {
  description = "Path to the directory containing Python scripts"
  type        = string
  default     = "../test"
}

variable "scripts_source_script_dir" {
  description = "Path to the directory containing Python scripts"
  type        = string
  default     = "../scripts"
}

# Bootstrap action script
resource "aws_s3_object" "bootstrap_script" {
  bucket  = "${var.scripts_bucket}-${data.aws_caller_identity.current.account_id}"
  key     = "bootstrap/install_dependencies.sh"
  content = <<-EOF
    #!/bin/bash
    set -e
    
    echo "Installing JDK dependencies..."
    sudo yum install -y java-17-amazon-corretto-devel

    echo "Installing Python dependencies..."
    sudo yum install -y python3-devel
    sudo /usr/bin/pip3 install numpy mmh3 xxhash cython
    echo "Verifying installation..."
    python3 -c "import numpy; import mmh3; import xxhash; import cython; print('All packages installed successfully')"
    
    echo "Building Cython .so file under cython_minhash/..."
    aws s3 cp s3://${var.scripts_bucket}-${data.aws_caller_identity.current.account_id}/scripts/cython_minhash/ /tmp/cython_minhash/ --recursive
    cd /tmp/cython_minhash
    python3 setup.py build_ext --inplace
    ls shingle_hash*.so || { echo "Cython build failed!"; exit 1; }

    echo "Cython .so generated. Now Install to system Python so all executors can import it"
    SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
    sudo cp shingle_hash*.so $SITE_PACKAGES/
    echo "Bootstrap complete!"
  EOF
}

# Upload requirements.txt to S3
resource "aws_s3_object" "requirements" {
  bucket = aws_s3_bucket.scripts_bucket.id
  key    = "scripts/requirements.txt"
  source = "${path.module}/requirements.txt" # Local file path

  depends_on = [time_sleep.wait_for_bucket]
}

# Upload deduplication script to S3
resource "aws_s3_object" "dedup_script" {
  bucket = aws_s3_bucket.scripts_bucket.id
  key    = "scripts/spark_partition_aware_deduplicattion_v2.py"
  source = "${path.module}/${var.scripts_source_src_dir}/spark_partition_aware_deduplicattion_v2.py"

  depends_on = [time_sleep.wait_for_bucket]
}

# Upload /script files to S3
locals {
  script_files = [
    "translate_driver_diagnostic_logs.py",
    "upload_heapdump_on_shutdown.sh",
  ]
}
resource "aws_s3_object" "diagnostic_script" {
  for_each = toset(local.script_files)

  bucket = aws_s3_bucket.scripts_bucket.id
  key    = "bootstrap/${each.value}"
  source = "${path.module}/${var.scripts_source_script_dir}/${each.value}"
  etag   = filemd5("${path.module}/${var.scripts_source_script_dir}/${each.value}")

  depends_on = [time_sleep.wait_for_bucket]
}

resource "aws_s3_object" "scala_script" {
  bucket = aws_s3_bucket.scripts_bucket.id
  key    = "scripts/minhash-udf_2.12-0.1.jar"
  source = "${path.module}/${var.scripts_source_scala_dir}/minhash-udf_2.12-0.1.jar"

  depends_on = [time_sleep.wait_for_bucket]
}

# Upload integration test script to S3
resource "aws_s3_object" "integration_test_script" {
  bucket = aws_s3_bucket.scripts_bucket.id
  key    = "scripts/spark_deduplication_test.py"
  source = "${path.module}/${var.scripts_source_test_dir}/integration_test/spark_deduplication_test.py"

  depends_on = [time_sleep.wait_for_bucket]
}

# Upload Cython MinHash source files to S3
# These are compiled on EMR nodes via bootstrap action
locals {
  cython_files = [
    "shingle_hash.pyx",
    "murmurhash3.c",
    "murmurhash3.h",
    "setup.py",
  ]
}

resource "aws_s3_object" "cython_files" {
  for_each = toset(local.cython_files)

  bucket = aws_s3_bucket.scripts_bucket.id
  key    = "scripts/cython_minhash/${each.value}"
  source = "${path.module}/${var.scripts_source_src_dir}/cython_minhash/${each.value}"

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

variable "wet_file_scale" {
  description = "WET file processing scale: '1' for validation only, '100' for 100 files, '1k' for 1,000 files, '9k' for 9,000 files, '90k' for 90,000 files"
  type        = string
  default     = "1"

  validation {
    condition     = contains(["1", "100", "1k", "9k", "90k"], var.wet_file_scale)
    error_message = "WET file scale must be '1', '100', '1k', '9k', or '90k'."
  }
}

# Use for cases when these values are computed from other values, not supplied by the user. 
locals {
  # Scale-specific configurations
  scale_configs = {
    "1" = {
      instance_type  = "m5a.xlarge" # 4 vCPU, 16 GB - for validating EMR, not running application
      on_demand_spot = { on_demand = 1, spot = 0 }
      on_demand_only = { on_demand = 1, spot = 0 }
    }
    "100" = {
      instance_type  = "r5.2xlarge"
      on_demand_spot = { on_demand = 2, spot = 2 }
      on_demand_only = { on_demand = 4, spot = 0 }
    }
    "1k" = {
      instance_type  = "r5ad.8xlarge"
      on_demand_spot = { on_demand = 2, spot = 2 }
      on_demand_only = { on_demand = 9, spot = 0 }
    }
    "9k" = {
      instance_type  = "r5ad.8xlarge"              # 32 vCores, 256 GB, 2×600 GB NVMe (most available)
      on_demand_spot = { on_demand = 9, spot = 0 } # Capacity units: 9 instances × 1 unit = 9
      on_demand_only = { on_demand = 9, spot = 0 } # Capacity units: 9 instances × 1 unit = 9
    }
    "90k" = {
      instance_type  = "r5ad.8xlarge"                # 32 vCores, 256 GB, 2×600 GB NVMe (same as 9k for availability)
      on_demand_spot = { on_demand = 64, spot = 62 } # Capacity units: 32 instances × 2 units = 64, 31 instances × 2 = 62
      on_demand_only = { on_demand = 126, spot = 0 } # Capacity units: 63 instances × 2 units = 126 (2,016 vCPU)
    }
  }

  # only for instances has suffix of d such as r5ad, it uses NVMe for heavy shuffles.
  # NVMe has different YARN directories.

  selected_config = local.scale_configs[var.wet_file_scale]
  capacity_config = var.instance_strategy == "on-demand" ? local.selected_config.on_demand_only : local.selected_config.on_demand_spot
  has_nvme        = can(regex("[0-9]a?d\\.", local.selected_config.instance_type)) # contains(["1k", "9k", "90k"], var.wet_file_scale)
  yarn_local_dirs = local.has_nvme ? "/mnt1/yarn,/mnt2/yarn" : "/mnt/yarn"
  yarn_log_dirs   = local.has_nvme ? "/mnt1/yarn/logs,/mnt2/yarn/logs" : "/mnt/yarn/logs"
}

resource "aws_emr_cluster" "dedup_cluster" {
  name          = "${var.cluster_name}-${var.wet_file_scale}"
  release_label = "emr-7.12.0"
  applications  = ["Spark", "Hadoop", "Hive", "JupyterEnterpriseGateway", "Livy"]

  # service_role = aws_iam_role.emr_service_role.arn # permission error
  service_role = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/EMR_DefaultRole"

  ec2_attributes {
    subnet_ids = var.subnet_ids # Multi-AZ support for better capacity availability
    # emr_managed_master_security_group = aws_security_group.emr_master.id # let EMR manage its own security
    # emr_managed_slave_security_group  = aws_security_group.emr_core.id # let EMR manage its own security
    instance_profile                  = "arn:aws:iam::REDACTED_ACCOUNT_ID:instance-profile/EMR_EC2_DefaultRole"
    key_name                          = aws_key_pair.emr_key.key_name
    additional_master_security_groups = aws_security_group.emr_master.id
  }

  # Primary (Master) node - On-Demand
  master_instance_fleet {
    name = "Primary"

    target_on_demand_capacity = 1 # On-demand for stability
    target_spot_capacity      = 0 # No spot for master

    instance_type_configs {
      instance_type     = var.wet_file_scale == "1" ? "m5a.xlarge" : "r5.xlarge" # Cheaper for validation
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

    target_on_demand_capacity = local.capacity_config.on_demand
    target_spot_capacity      = local.capacity_config.spot

    # Primary choice - dynamically selected based on scale
    instance_type_configs {
      instance_type     = local.selected_config.instance_type
      weighted_capacity = var.wet_file_scale == "90k" ? 2 : 1 # 8xlarge: 2 units for 90k, 1 unit for 9k and smaller

      bid_price_as_percentage_of_on_demand_price = var.bid_strategy == "peak-event" ? 100 : 80 # Peak events need 100%

      ebs_config {
        size                 = 100
        type                 = "gp3"
        iops                 = 3000
        volumes_per_instance = 1
      }
    }

    # Fallbacks for 1k scale - maintain 128 cores with proper weighted capacity

    # Larger instance fallbacks - fewer instances for same 128 cores
    dynamic "instance_type_configs" {
      for_each = var.wet_file_scale == "1k" ? [1] : []
      content {
        instance_type     = "r5d.8xlarge"
        weighted_capacity = 1

        bid_price_as_percentage_of_on_demand_price = var.bid_strategy == "peak-event" ? 100 : 80

        ebs_config {
          size                 = 100
          type                 = "gp3"
          iops                 = 3000
          volumes_per_instance = 1
        }
      }
    }

    dynamic "instance_type_configs" {
      for_each = var.wet_file_scale == "1k" ? [1] : []
      content {
        instance_type     = "r6id.8xlarge" # Same as r6i + 1x 1900GB NVMe SSD
        weighted_capacity = 1

        bid_price_as_percentage_of_on_demand_price = var.bid_strategy == "peak-event" ? 100 : 75

        # Small EBS for OS/logs - shuffle data goes to NVMe (/mnt, /mnt1) instead of EBS automatically
        ebs_config {
          size                 = 100
          type                 = "gp3"
          iops                 = 3000
          volumes_per_instance = 1
        }
      }
    }

    # Fallbacks for 100 scale only
    dynamic "instance_type_configs" {
      for_each = var.wet_file_scale == "100" ? [1] : []
      content {
        instance_type     = "r5ad.2xlarge"
        weighted_capacity = 1

        bid_price_as_percentage_of_on_demand_price = var.bid_strategy == "peak-event" ? 100 : 85

        ebs_config {
          size                 = 100
          type                 = "gp3"
          iops                 = 3000
          volumes_per_instance = 1
        }
      }
    }

    dynamic "instance_type_configs" {
      for_each = var.wet_file_scale == "100" ? [1] : []
      content {
        instance_type     = "r6id.2xlarge"
        weighted_capacity = 1

        bid_price_as_percentage_of_on_demand_price = var.bid_strategy == "peak-event" ? 100 : 85

        ebs_config {
          size                 = 100
          type                 = "gp3"
          iops                 = 3000
          volumes_per_instance = 1
        }
      }
    }

    dynamic "instance_type_configs" {
      for_each = var.wet_file_scale == "100" ? [1] : []
      content {
        instance_type     = "r6gd.2xlarge"
        weighted_capacity = 1 # ARM alternative to r5.2xlarge

        bid_price_as_percentage_of_on_demand_price = var.bid_strategy == "peak-event" ? 100 : 75

        ebs_config {
          size                 = 100
          type                 = "gp3"
          iops                 = 3000
          volumes_per_instance = 1
        }
      }
    }

    # For 9k/90k scale - NVMe-only fallbacks prioritizing availability (smaller = more available)
    # 9k:  Primary: 9× r5ad.8xlarge (288 vCPU), 1 unit each
    # 90k: Primary: 63× r5ad.8xlarge (2,016 vCPU), 2 units each
    dynamic "instance_type_configs" {
      for_each = contains(["9k", "90k"], var.wet_file_scale) ? [1] : []
      content {
        instance_type     = "r5d.8xlarge" # 32 vCPU, Intel, 2×600GB NVMe
        weighted_capacity = var.wet_file_scale == "90k" ? 2 : 1

        bid_price_as_percentage_of_on_demand_price = var.bid_strategy == "peak-event" ? 100 : 80

        ebs_config {
          size                 = 100
          type                 = "gp3"
          iops                 = 3000
          volumes_per_instance = 1
        }
      }
    }

    dynamic "instance_type_configs" {
      for_each = contains(["9k", "90k"], var.wet_file_scale) ? [1] : []
      content {
        instance_type     = "r6id.8xlarge" # 32 vCPU, newer Intel, 2×950GB NVMe
        weighted_capacity = var.wet_file_scale == "90k" ? 2 : 1

        bid_price_as_percentage_of_on_demand_price = var.bid_strategy == "peak-event" ? 100 : 80

        ebs_config {
          size                 = 100
          type                 = "gp3"
          iops                 = 3000
          volumes_per_instance = 1
        }
      }
    }

    dynamic "instance_type_configs" {
      for_each = contains(["9k", "90k"], var.wet_file_scale) ? [1] : []
      content {
        instance_type     = "r6gd.8xlarge" # 32 vCPU, Graviton ARM, 1×1900GB NVMe (highly available)
        weighted_capacity = var.wet_file_scale == "90k" ? 2 : 1

        bid_price_as_percentage_of_on_demand_price = var.bid_strategy == "peak-event" ? 100 : 80

        ebs_config {
          size                 = 100
          type                 = "gp3"
          iops                 = 3000
          volumes_per_instance = 1
        }
      }
    }

  }

  bootstrap_action {
    name = "Install Python and JDK17 dependencies"
    path = "s3://${aws_s3_bucket.scripts_bucket.id}/bootstrap/install_dependencies.sh"
  }

  bootstrap_action {
    name = "install-heapdump-upload"
    path = "s3://${aws_s3_bucket.scripts_bucket.id}/bootstrap/upload_heapdump_on_shutdown.sh"
    args = ["s3://${var.text_dedupe_benchmark_bucket}/heapdumps",                             # S3_DEST
      "s3://${aws_s3_bucket.scripts_bucket.id}/bootstrap/translate_driver_diagnostic_logs.py" # DIAG_SCRIPT_S3
    ]
  }


  # Spark and YARN configurations
  configurations_json = jsonencode([
    {
      Classification = "spark-defaults"
      Properties = {
        "spark.default.parallelism"       = "2000"
        "spark.memory.fraction"           = "0.8"
        "spark.memory.storageFraction"    = "0.3"
        "spark.serializer"                = "org.apache.spark.serializer.KryoSerializer"
        "spark.kryoserializer.buffer.max" = "1024m"
        "spark.driver.memory"             = "12g"
        "spark.executor.memory"           = "12g"
        "spark.executor.cores"            = "4"
        "spark.dynamicAllocation.enabled" = "false"
        "spark.sql.adaptive.enabled"      = "true"

        "spark.sql.catalog.glue_catalog" : "org.apache.iceberg.spark.SparkCatalog",

        # "spark.eventLog.dir": "hdfs:///var/log/spark/apps", # moved logs from hdfs to S3
        # "spark.history.fs.logDirectory": "hdfs:///var/log/spark/apps",

        "spark.eventLog.enabled" : "true",
        "spark.eventLog.dir" : "s3://${var.text_dedupe_benchmark_bucket}/spark-history/",
        "spark.eventLog.compress" : "true",
        "spark.eventLog.compression.codec" : "zstd",
        "spark.history.fs.logDirectory" : "s3://${var.text_dedupe_benchmark_bucket}/spark-history/"

        "spark.driver.extraJavaOptions" = join(" ", [
          "-XX:+HeapDumpOnOutOfMemoryError",
          "-XX:HeapDumpPath=/tmp/driver_heap_%p.hprof",
          "-XX:NativeMemoryTracking=summary",                           # remove if this want to remove 10% offheap overhead.
          "-Xlog:gc*:file=/tmp/driver_gc_%p.log:time,uptime,level,tags" # clean up left over jvm diagnostic log
        ])
      }
    },
    {
      Classification = "yarn-site"
      Properties = {
        "yarn.nodemanager.vmem-check-enabled"               = "false"
        "yarn.nodemanager.pmem-check-enabled"               = "false"
        "yarn.nodemanager.aux-services"                     = "mapreduce_shuffle,spark_shuffle"
        "yarn.nodemanager.aux-services.spark_shuffle.class" = "org.apache.spark.network.yarn.YarnShuffleService"
        "yarn.nodemanager.local-dirs"                       = local.yarn_local_dirs # "/mnt1/yarn,/mnt2/yarn"
        "yarn.nodemanager.log-dirs"                         = local.yarn_log_dirs   # "/mnt1/yarn/logs,/mnt2/yarn/logs"
      }
    },
    {
      Classification = "spark-env"
      Properties     = {}
      Configurations = [
        {
          Classification = "export"
          Properties = {
            "PYSPARK_PYTHON"      = "/usr/bin/python3"
            "SPARK_DAEMON_MEMORY" = "8g"
            /* Note for SPARK_DAEMON_CLASSPATH:
              For EMR releases after 6.3 and 5.30, the required JAR file 
              is added by default to /usr/lib/spark/jars
              which covers Spark applications (driver/executors).

              But the History Server daemon doesn't use that classpath by default,
               which is why the explicit SPARK_DAEMON_CLASSPATH is needed.

            */
            "SPARK_DAEMON_CLASSPATH" = "/usr/lib/hadoop/hadoop-aws.jar:/usr/share/aws/aws-java-sdk/aws-java-sdk-bundle-1.12.792.jar:/usr/share/aws/aws-java-sdk-v2/aws-sdk-java-bundle-2.35.5.jar"
            # 
          }
        }
      ]
    },
    # ICEBERG setting
    {
      "Classification" : "iceberg-defaults",
      "Properties" : {
        "iceberg.enabled" : "true"
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
    aws_s3_object.bootstrap_script,
    aws_s3_object.diagnostic_script
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
