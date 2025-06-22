"""
Frontend configuration for the Sentiment Analysis React app.

This module provides configuration classes for deploying the React frontend
to AWS using S3 and CloudFront.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import aws_cdk as cdk


@dataclass
class FrontendConfig:
    """Configuration for the React frontend deployment."""
    
    # S3 Configuration
    bucket_name_prefix: str = "sentiment-analysis-frontend"
    enable_versioning: bool = True
    enable_public_read_access: bool = False  # CloudFront will access via OAI
    
    # CloudFront Configuration
    enable_cloudfront: bool = True
    price_class: str = "PriceClass_100"  # Use only North America and Europe
    enable_compression: bool = True
    default_root_object: str = "index.html"
    
    # Build Configuration
    build_command: str = "npm run build"
    build_directory: str = "build"
    source_directory: str = "../frontend"
    
    # Cache Configuration
    default_cache_behavior: Dict = None
    additional_cache_behaviors: List[Dict] = None
    
    # Environment Variables for React App
    environment_variables: Dict[str, str] = None
    
    # Custom Domain (Optional)
    domain_name: Optional[str] = None
    certificate_arn: Optional[str] = None
    hosted_zone_id: Optional[str] = None
    
    def __post_init__(self):
        """Set default values after initialization."""
        if self.default_cache_behavior is None:
            self.default_cache_behavior = {
                "allowed_methods": ["GET", "HEAD", "OPTIONS"],
                "cached_methods": ["GET", "HEAD"],
                "compress": True,
                "cache_policy_id": "4135ea2d-6df8-44a3-9df3-4b5a84be39ad",  # CachingDisabled
                "origin_request_policy_id": "88a5eaf4-2fd4-4709-b370-b4c650ea3fcf",  # CORS-S3Origin
                "response_headers_policy_id": "67f7725c-6f97-4210-82d7-5512b31e9d03"  # SecurityHeaders
            }
        
        if self.additional_cache_behaviors is None:
            self.additional_cache_behaviors = [
                {
                    "path_pattern": "/static/*",
                    "allowed_methods": ["GET", "HEAD"],
                    "cached_methods": ["GET", "HEAD"],
                    "compress": True,
                    "cache_policy_id": "658327ea-f89d-4fab-a63d-7e88639e58f6",  # CachingOptimized
                    "ttl": cdk.Duration.days(365)
                }
            ]
        
        if self.environment_variables is None:
            self.environment_variables = {}
    
    def validate(self) -> None:
        """Validate the configuration."""
        if not self.bucket_name_prefix:
            raise ValueError("bucket_name_prefix cannot be empty")
        
        if not self.build_command:
            raise ValueError("build_command cannot be empty")
        
        if not self.build_directory:
            raise ValueError("build_directory cannot be empty")
        
        if not self.source_directory:
            raise ValueError("source_directory cannot be empty")
        
        if self.domain_name and not self.certificate_arn:
            raise ValueError("certificate_arn is required when domain_name is specified")


@dataclass
class BuildConfig:
    """Configuration for building the React application."""
    
    # Node.js Configuration
    node_version: str = "18"
    npm_registry: Optional[str] = None
    
    # Build Process
    install_command: str = "npm ci"
    build_command: str = "npm run build"
    build_timeout: cdk.Duration = cdk.Duration.minutes(10)
    
    # Build Environment
    build_environment: Dict[str, str] = None
    
    # Output Configuration
    output_directory: str = "build"
    include_source_maps: bool = False
    
    def __post_init__(self):
        """Set default values after initialization."""
        if self.build_environment is None:
            self.build_environment = {
                "NODE_ENV": "production",
                "GENERATE_SOURCEMAP": "false" if not self.include_source_maps else "true"
            }
    
    def validate(self) -> None:
        """Validate the build configuration."""
        if not self.install_command:
            raise ValueError("install_command cannot be empty")
        
        if not self.build_command:
            raise ValueError("build_command cannot be empty")
        
        if not self.output_directory:
            raise ValueError("output_directory cannot be empty")


@dataclass
class DeploymentConfig:
    """Configuration for deployment process."""
    
    # Deployment Strategy
    enable_blue_green: bool = False
    enable_rollback: bool = True
    
    # Cache Invalidation
    invalidate_cache_on_deploy: bool = True
    invalidation_paths: List[str] = None
    
    # Monitoring
    enable_monitoring: bool = True
    enable_alarms: bool = True
    
    # Security
    enable_waf: bool = False
    enable_security_headers: bool = True
    
    def __post_init__(self):
        """Set default values after initialization."""
        if self.invalidation_paths is None:
            self.invalidation_paths = ["/*"]
    
    def validate(self) -> None:
        """Validate the deployment configuration."""
        if self.invalidate_cache_on_deploy and not self.invalidation_paths:
            raise ValueError("invalidation_paths cannot be empty when cache invalidation is enabled")


# Predefined configurations for different environments
def get_frontend_config(environment: str = "prod") -> FrontendConfig:
    """Get frontend configuration for the specified environment."""
    
    base_config = FrontendConfig()
    
    if environment == "dev":
        base_config.price_class = "PriceClass_All"
        base_config.enable_versioning = False
        base_config.default_cache_behavior["cache_policy_id"] = "4135ea2d-6df8-44a3-9df3-4b5a84be39ad"  # CachingDisabled
    
    elif environment == "staging":
        base_config.price_class = "PriceClass_200"
        base_config.enable_versioning = True
    
    elif environment == "prod":
        base_config.price_class = "PriceClass_100"
        base_config.enable_versioning = True
        base_config.enable_compression = True
    
    return base_config


def get_build_config(environment: str = "prod") -> BuildConfig:
    """Get build configuration for the specified environment."""
    
    base_config = BuildConfig()
    
    if environment == "dev":
        base_config.include_source_maps = True
        base_config.build_environment["NODE_ENV"] = "development"
    
    elif environment == "staging":
        base_config.include_source_maps = True
        base_config.build_environment["NODE_ENV"] = "production"
    
    elif environment == "prod":
        base_config.include_source_maps = False
        base_config.build_environment["NODE_ENV"] = "production"
    
    return base_config


def get_deployment_config(environment: str = "prod") -> DeploymentConfig:
    """Get deployment configuration for the specified environment."""
    
    base_config = DeploymentConfig()
    
    if environment == "dev":
        base_config.enable_alarms = False
        base_config.enable_waf = False
    
    elif environment == "staging":
        base_config.enable_alarms = True
        base_config.enable_waf = False
    
    elif environment == "prod":
        base_config.enable_alarms = True
        base_config.enable_waf = True
    
    return base_config
