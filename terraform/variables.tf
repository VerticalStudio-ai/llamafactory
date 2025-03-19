
variable "environment" {
  description = "Environment name (nonprod, prod)"
  type        = string
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.3.0.0/16"
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "eu-central-1"
}

variable "github_org" {
  type        = string
  description = "GitHub organization name"
  default     = "VerticalStudio-ai"
}
