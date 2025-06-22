"""
Lambda function configuration for sentiment analysis deployment.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
from aws_cdk import Duration


@dataclass
class LambdaConfig:
    """Configuration for Lambda function deployment."""
    
    # Basic function settings
    function_name: str = "sentiment-analysis"
    description: str = "Sentiment analysis using LSTM and Verbalizer models"
    
    # Resource allocation
    memory_size: int = 2048  # MB - Reduced from 4096 for cost optimization
    timeout: Duration = field(default_factory=lambda: Duration.minutes(5))  # Reduced from 15 minutes
    
    # Runtime settings
    runtime_platform: str = "linux/amd64"
    architecture: str = "x86_64"
    
    # Environment variables
    environment_vars: Dict[str, str] = field(default_factory=lambda: {
        "PYTHONPATH": "/var/task",
        "MODEL_PATH": "/var/task/models",  # Fixed path issue
        "LOG_LEVEL": "INFO"
    })
    
    # Docker build settings
    docker_build_args: Dict[str, str] = field(default_factory=lambda: {
        "BUILDPLATFORM": "linux/amd64"
    })
    
    # Model paths (relative to Lambda directory)
    models_source_path: str = "../lambda"  # Path to Dockerfile directory
    
    # CloudWatch settings
    log_retention_days: int = 7  # Cost optimization - shorter retention
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.memory_size < 512 or self.memory_size > 10240:
            raise ValueError("Memory size must be between 512 MB and 10240 MB")
        
        if self.timeout.to_seconds() < 1 or self.timeout.to_seconds() > 900:
            raise ValueError("Timeout must be between 1 second and 15 minutes")
        
        if not self.function_name or not self.function_name.replace("-", "").replace("_", "").isalnum():
            raise ValueError("Function name must be alphanumeric with hyphens/underscores only")


@dataclass
class LambdaPermissions:
    """IAM permissions configuration for Lambda function."""
    
    # Basic execution permissions
    basic_execution: bool = True
    
    # CloudWatch permissions
    cloudwatch_logs: bool = True
    
    # Additional permissions (if needed for future features)
    s3_read: Optional[str] = None  # S3 bucket ARN for model artifacts
    dynamodb_read: Optional[str] = None  # DynamoDB table ARN for caching
    
    def get_managed_policies(self) -> list:
        """Get list of AWS managed policies to attach."""
        policies = []
        
        if self.basic_execution:
            policies.append("service-role/AWSLambdaBasicExecutionRole")
        
        return policies
    
    def get_inline_policies(self) -> Dict[str, Dict]:
        """Get inline policies for additional permissions."""
        policies = {}
        
        if self.s3_read:
            policies["S3ReadPolicy"] = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "s3:GetObject",
                            "s3:ListBucket"
                        ],
                        "Resource": [
                            self.s3_read,
                            f"{self.s3_read}/*"
                        ]
                    }
                ]
            }
        
        if self.dynamodb_read:
            policies["DynamoDBReadPolicy"] = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "dynamodb:GetItem",
                            "dynamodb:Query",
                            "dynamodb:Scan"
                        ],
                        "Resource": self.dynamodb_read
                    }
                ]
            }
        
        return policies
