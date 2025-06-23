"""
Simplified CDK Construct for deploying React frontend to S3 and CloudFront.

This construct handles:
- Building the React application with API configuration at build time
- Deploying to S3 bucket
- Setting up CloudFront distribution
- Configuring proper caching and security headers

No custom resources needed - all configuration is done at build time.
"""

import os
import time
from pathlib import Path
from typing import Dict, Optional

import aws_cdk as cdk
from aws_cdk import (
    aws_s3 as s3,
    aws_cloudfront as cloudfront,
    aws_cloudfront_origins as origins,
    aws_s3_deployment as s3deploy
)
from constructs import Construct

from config.frontend_config import FrontendConfig, BuildConfig, DeploymentConfig


class SimplifiedSentimentFrontendConstruct(Construct):
    """
    Simplified CDK Construct for deploying the React frontend.
    
    This construct creates:
    - S3 bucket for hosting static files
    - CloudFront distribution for global CDN
    - Build process for React application with API configuration
    - Proper caching and security configurations
    
    All API configuration is injected at build time, eliminating the need
    for custom resources and runtime configuration updates.
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        api_url: str,
        api_key_id: str,
        frontend_config: FrontendConfig,
        build_config: BuildConfig,
        deployment_config: DeploymentConfig,
        environment: str = "prod",
        **kwargs
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)
        
        self.api_url = api_url
        self.api_key_id = api_key_id
        self.frontend_config = frontend_config
        self.build_config = build_config
        self.deployment_config = deployment_config
        self.environment = environment
        
        # Validate configurations
        self._validate_configs()
        
        # Create S3 bucket
        self._create_s3_bucket()
        
        # Create CloudFront distribution
        self._create_cloudfront_distribution()
        
        # Deploy to S3 with build-time configuration
        self._deploy_to_s3()
        
        # Set up monitoring (if enabled)
        if self.deployment_config.enable_monitoring:
            self._setup_monitoring()
    
    def _validate_configs(self) -> None:
        """Validate all configurations."""
        self.frontend_config.validate()
        self.build_config.validate()
        self.deployment_config.validate()
        
        if not self.api_url:
            raise ValueError("api_url is required")
        
        if not self.api_key_id:
            raise ValueError("api_key_id is required")
    
    def _create_s3_bucket(self) -> None:
        """Create S3 bucket for hosting the frontend."""
        
        # Generate unique bucket name with timestamp to avoid conflicts
        bucket_name = f"{self.frontend_config.bucket_name_prefix}-{self.environment}-{cdk.Aws.ACCOUNT_ID}-{int(time.time())}"
        
        # Create S3 bucket
        self.bucket = s3.Bucket(
            self,
            "FrontendBucket",
            bucket_name=bucket_name,
            versioned=self.frontend_config.enable_versioning,
            removal_policy=cdk.RemovalPolicy.DESTROY if self.environment == "dev" else cdk.RemovalPolicy.RETAIN,
            auto_delete_objects=self.environment == "dev",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,  # CloudFront will access via OAC
            encryption=s3.BucketEncryption.S3_MANAGED,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="DeleteOldVersions",
                    enabled=True,
                    noncurrent_version_expiration=cdk.Duration.days(30)
                )
            ] if self.frontend_config.enable_versioning else None
        )
        
        # Add tags
        cdk.Tags.of(self.bucket).add("Environment", self.environment)
        cdk.Tags.of(self.bucket).add("Service", "sentiment-analysis-frontend")
        cdk.Tags.of(self.bucket).add("Component", "static-hosting")
    
    def _create_cloudfront_distribution(self) -> None:
        """Create CloudFront distribution."""
        
        if not self.frontend_config.enable_cloudfront:
            return
        
        # Create CloudFront distribution with Origin Access Control
        self.distribution = cloudfront.Distribution(
            self,
            "FrontendDistribution",
            default_behavior=cloudfront.BehaviorOptions(
                origin=origins.S3BucketOrigin.with_origin_access_control(self.bucket),
                allowed_methods=cloudfront.AllowedMethods.ALLOW_GET_HEAD_OPTIONS,
                cached_methods=cloudfront.CachedMethods.CACHE_GET_HEAD_OPTIONS,
                compress=self.frontend_config.enable_compression,
                cache_policy=cloudfront.CachePolicy.CACHING_DISABLED,  # For SPA
                origin_request_policy=cloudfront.OriginRequestPolicy.CORS_S3_ORIGIN,
                response_headers_policy=cloudfront.ResponseHeadersPolicy.SECURITY_HEADERS,
                viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS
            ),
            additional_behaviors={
                "/static/*": cloudfront.BehaviorOptions(
                    origin=origins.S3BucketOrigin.with_origin_access_control(self.bucket),
                    allowed_methods=cloudfront.AllowedMethods.ALLOW_GET_HEAD,
                    cached_methods=cloudfront.CachedMethods.CACHE_GET_HEAD,
                    compress=True,
                    cache_policy=cloudfront.CachePolicy.CACHING_OPTIMIZED,
                    viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS
                )
            },
            default_root_object=self.frontend_config.default_root_object,
            error_responses=[
                cloudfront.ErrorResponse(
                    http_status=404,
                    response_http_status=200,
                    response_page_path="/index.html",
                    ttl=cdk.Duration.minutes(5)
                ),
                cloudfront.ErrorResponse(
                    http_status=403,
                    response_http_status=200,
                    response_page_path="/index.html",
                    ttl=cdk.Duration.minutes(5)
                )
            ],
            price_class=cloudfront.PriceClass.PRICE_CLASS_100,
            enabled=True,
            comment=f"Sentiment Analysis Frontend - {self.environment}"
        )
        
        # Add tags
        cdk.Tags.of(self.distribution).add("Environment", self.environment)
        cdk.Tags.of(self.distribution).add("Service", "sentiment-analysis-frontend")
        cdk.Tags.of(self.distribution).add("Component", "cdn")
    
    def _deploy_to_s3(self) -> None:
        """Deploy the built React app to S3 with API configuration."""
        
        # Get the frontend source directory
        frontend_dir = Path(__file__).parent.parent / self.frontend_config.source_directory
        
        if not frontend_dir.exists():
            raise ValueError(f"Frontend directory not found: {frontend_dir}")
        
        # Prepare build environment with API configuration
        build_env = {
            **self.build_config.build_environment,
            "REACT_APP_API_URL": self.api_url,
            "REACT_APP_API_KEY_ID": self.api_key_id,
            **self.frontend_config.environment_variables
        }
        
        print(f"Frontend will be built from: {frontend_dir}")
        print(f"API URL will be: {self.api_url}")
        
        # Deploy with build-time configuration
        self.deployment = s3deploy.BucketDeployment(
            self,
            "FrontendDeployment",
            sources=[s3deploy.Source.asset(
                str(frontend_dir),
                exclude=[
                    "node_modules",
                    "build",
                    ".git",
                    "*.log",
                    ".env.local",
                    ".env.development.local", 
                    ".env.test.local",
                    "coverage",
                    ".DS_Store"
                ],
                bundling=cdk.BundlingOptions(
                    image=cdk.DockerImage.from_registry("node:20-alpine"),  # Updated to Node 20
                    environment=build_env,
                    command=[
                        "sh", "-c",
                        """
                        cd /asset-input &&
                        echo "Installing dependencies..." &&
                        rm -rf node_modules/.cache || true &&
                        rm -rf node_modules || true &&
                        npm install &&
                        echo "Building React application..." &&
                        echo "API URL: $REACT_APP_API_URL" &&
                        echo "API Key ID: $REACT_APP_API_KEY_ID" &&
                        npm run build &&
                        echo "Build completed successfully" &&
                        cp -r build/* /asset-output/
                        """
                    ],
                    user="root"
                )
            )],
            destination_bucket=self.bucket,
            distribution=self.distribution if hasattr(self, 'distribution') else None,
            distribution_paths=self.deployment_config.invalidation_paths if self.deployment_config.invalidate_cache_on_deploy else None
        )
        
        # Ensure deployment happens after distribution is created
        if hasattr(self, 'distribution'):
            self.deployment.node.add_dependency(self.distribution)
    
    def _setup_monitoring(self) -> None:
        """Set up monitoring for the frontend."""
        # This would include CloudWatch alarms for CloudFront metrics
        # Implementation would depend on specific monitoring requirements
        pass
    
    # Public properties
    @property
    def bucket_name(self) -> str:
        """Get the S3 bucket name."""
        return self.bucket.bucket_name
    
    @property
    def bucket_arn(self) -> str:
        """Get the S3 bucket ARN."""
        return self.bucket.bucket_arn
    
    @property
    def distribution_id(self) -> Optional[str]:
        """Get the CloudFront distribution ID."""
        return self.distribution.distribution_id if hasattr(self, 'distribution') else None
    
    @property
    def distribution_domain_name(self) -> Optional[str]:
        """Get the CloudFront distribution domain name."""
        return self.distribution.distribution_domain_name if hasattr(self, 'distribution') else None
    
    @property
    def website_url(self) -> str:
        """Get the website URL."""
        if hasattr(self, 'distribution'):
            return f"https://{self.distribution.distribution_domain_name}"
        else:
            return f"https://{self.bucket.bucket_website_domain_name}"
    
    def get_deployment_info(self) -> Dict[str, str]:
        """Get deployment information."""
        return {
            "bucket_name": self.bucket_name,
            "bucket_arn": self.bucket_arn,
            "distribution_id": self.distribution_id or "N/A",
            "website_url": self.website_url,
            "environment": self.environment,
            "api_url": self.api_url,
            "configuration_method": "build-time"
        }
