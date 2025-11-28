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

# Security Group for EMR
resource "aws_security_group" "emr_master" {
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

  # Spark UI
  ingress {
    from_port   = 4040
    to_port     = 4040
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # YARN ResourceManager
  ingress {
    from_port   = 8088
    to_port     = 8088
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Jupyter
  ingress {
    from_port   = 8888
    to_port     = 8888
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

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

resource "aws_security_group" "emr_core" {
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

# S3 bucket for scripts and data (name must be globally unique)
resource "aws_s3_bucket" "scripts_bucket" {
  bucket = "${var.scripts_bucket}-${data.aws_caller_identity.current.account_id}"
}

# Wait for S3 bucket to propagate (eventual consistency)
resource "time_sleep" "wait_for_bucket" {
  depends_on      = [aws_s3_bucket.scripts_bucket]
  create_duration = "30s"
}

# S3 bucket for logs
resource "aws_s3_bucket" "emr_logs" {
  bucket = "${var.scripts_bucket}-emr-logs-${data.aws_caller_identity.current.account_id}"
}

# Bootstrap action script
resource "aws_s3_object" "bootstrap_script" {
  bucket  = "${var.scripts_bucket}-${data.aws_caller_identity.current.account_id}"
  key     = "bootstrap/install_dependencies.sh"
  content = <<-EOF
    #!/bin/bash
    set -e
    
    echo "Installing Python dependencies..."
    sudo pip3 install --upgrade pip
    sudo pip3 install numpy mmh3 xxhash
    
    echo "Bootstrap complete!"
  EOF
}

# EMR Cluster
resource "aws_emr_cluster" "dedup_cluster" {
  name          = var.cluster_name
  release_label = "emr-7.12.0"
  applications  = ["Spark", "Hadoop", "Hive", "JupyterEnterpriseGateway", "Livy"]

  service_role = aws_iam_role.emr_service_role.arn

  ec2_attributes {
    subnet_id                         = var.subnet_id
    # emr_managed_master_security_group = aws_security_group.emr_master.id # let EMR manage its own security
    # emr_managed_slave_security_group  = aws_security_group.emr_core.id # let EMR manage its own security
    instance_profile                  = aws_iam_instance_profile.emr_ec2_profile.arn
    key_name                          = aws_key_pair.emr_key.key_name
  }

  # Primary (Master) node - On-Demand
  master_instance_group {
    instance_type  = "r8g.xlarge"
    instance_count = 1
    name           = "Primary"

    ebs_config {
      size                 = 100
      type                 = "gp3"
      iops                 = 3000
      throughput           = 125
      volumes_per_instance = 1
    }
  }

  # Core nodes - Spot instances for cost savings
  core_instance_group {
    instance_type  = "r8g.xlarge"
    instance_count = 4
    name           = "Core"

    bid_price = "0.10"  # Spot price (slightly above current $0.088)

    ebs_config {
      size                 = 100
      type                 = "gp3"
      iops                 = 3000
      throughput           = 125
      volumes_per_instance = 1
    }

    # Additional EBS volume for shuffle/spill
    ebs_config {
      size                 = 150
      type                 = "gp3"
      iops                 = 3000
      throughput           = 125
      volumes_per_instance = 1
    }
  }

  # Bootstrap action to install dependencies
  bootstrap_action {
    name = "Install Python dependencies"
    path = "s3://${var.scripts_bucket}/bootstrap/install_dependencies.sh"
  }

  # Spark and YARN configurations
  configurations_json = jsonencode([
    {
      Classification = "spark-defaults"
      Properties = {
        "spark.sql.shuffle.partitions"     = "1000"
        "spark.default.parallelism"        = "1000"
        "spark.memory.fraction"            = "0.8"
        "spark.memory.storageFraction"     = "0.3"
        "spark.serializer"                 = "org.apache.spark.serializer.KryoSerializer"
        "spark.kryoserializer.buffer.max"  = "1024m"
        "spark.driver.memory"              = "8g"
        "spark.executor.memory"            = "24g"
        "spark.executor.cores"             = "4"
        "spark.dynamicAllocation.enabled"  = "false"
        "spark.sql.adaptive.enabled"       = "true"
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
          }
        }
      ]
    }
  ])

  log_uri = "s3://${aws_s3_bucket.emr_logs.id}/logs/"

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
