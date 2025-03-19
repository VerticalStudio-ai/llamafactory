terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {}
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Environment = var.environment
      Project     = "Studio"
      ManagedBy   = "Terraform"
      Repository  = "https://github.com/VerticalStudio-ai/llamafactory"
    }
  }
}

data "aws_availability_zones" "available" {
  state = "available"
}