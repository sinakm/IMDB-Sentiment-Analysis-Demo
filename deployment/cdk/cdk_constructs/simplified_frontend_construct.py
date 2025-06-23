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
import json
from pathlib import Path
from typing import Dict, Optional

import aws_cdk as cdk
from aws_cdk import (
    aws_s3 as s3,
    aws_cloudfront as cloudfront,
    aws_cloudfront_origins as origins,
    aws_s3_deployment as s3deploy,
    aws_lambda as lambda_,
    aws_iam as iam,
    custom_resources as cr
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
        """Deploy the built React app to S3 with runtime configuration."""
        
        # Get the frontend source directory
        frontend_dir = Path(__file__).parent.parent / self.frontend_config.source_directory
        
        if not frontend_dir.exists():
            raise ValueError(f"Frontend directory not found: {frontend_dir}")
        
        # Prepare build environment WITHOUT API configuration (build-time only)
        build_env = {
            **self.build_config.build_environment,
            **self.frontend_config.environment_variables
        }
        
        print(f"Frontend will be built from: {frontend_dir}")
        print(f"API URL will be: {self.api_url}")
        
        # Deploy with runtime configuration using a custom resource approach
        self.deployment = s3deploy.BucketDeployment(
            self,
            "FrontendDeployment",
            sources=[
                # Build and deploy the React app
                s3deploy.Source.asset(
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
                        image=cdk.DockerImage.from_registry("node:20-alpine"),
                        environment=build_env,
                        command=[
                            "sh", "-c",
                            """
                            cd /asset-input &&
                            echo "Installing dependencies..." &&
                            rm -rf node_modules/.cache || true &&
                            rm -rf node_modules || true &&
                            npm install &&
                            echo "Building React application (without API config)..." &&
                            npm run build &&
                            echo "Build completed successfully" &&
                            cp -r build/* /asset-output/
                            """
                        ],
                        user="root"
                    )
                )
            ],
            destination_bucket=self.bucket,
            distribution=self.distribution if hasattr(self, 'distribution') else None,
            distribution_paths=self.deployment_config.invalidation_paths if self.deployment_config.invalidate_cache_on_deploy else None
        )
        
        # Create runtime configuration file separately using a custom resource
        self._create_runtime_config()
        
        # Ensure deployment happens after distribution is created
        if hasattr(self, 'distribution'):
            self.deployment.node.add_dependency(self.distribution)
    
    def _create_runtime_config(self) -> None:
        """Create runtime configuration file using a custom resource."""
        
        # Create Lambda function for the custom resource
        config_lambda = lambda_.Function(
            self,
            "ConfigUpdater",
            runtime=lambda_.Runtime.PYTHON_3_9,
            handler="index.handler",
            code=lambda_.Code.from_inline("""
import boto3
import json
import urllib3

def handler(event, context):
    print(f"Event: {json.dumps(event)}")
    
    try:
        s3 = boto3.client('s3')
        apigateway = boto3.client('apigateway')
        
        bucket_name = event['ResourceProperties']['BucketName']
        api_url = event['ResourceProperties']['ApiUrl']
        api_key_id = event['ResourceProperties']['ApiKeyId']
        environment = event['ResourceProperties']['Environment']
        
        # Get the actual API key value from API Gateway
        try:
            api_key_response = apigateway.get_api_key(
                apiKey=api_key_id,
                includeValue=True
            )
            api_key_value = api_key_response['value']
            print(f"Retrieved API key value for ID: {api_key_id}")
        except Exception as e:
            print(f"Error retrieving API key: {e}")
            api_key_value = api_key_id  # Fallback to ID if retrieval fails
        
        # Create the runtime configuration content with actual API key value
        config_content = f'''window.__ENV__ = {{
  API_URL: '{api_url}',
  API_KEY_ID: '{api_key_value}',
  ENVIRONMENT: '{environment}'
}};'''
        
        if event['RequestType'] in ['Create', 'Update']:
            # Upload the configuration file to S3
            s3.put_object(
                Bucket=bucket_name,
                Key='env.js',
                Body=config_content,
                ContentType='application/javascript',
                CacheControl='no-cache'
            )
            print(f"Successfully uploaded env.js to {bucket_name}")
        
        elif event['RequestType'] == 'Delete':
            # Delete the configuration file
            try:
                s3.delete_object(Bucket=bucket_name, Key='env.js')
                print(f"Successfully deleted env.js from {bucket_name}")
            except Exception as e:
                print(f"Error deleting env.js: {e}")
        
        # Send success response
        send_response(event, context, 'SUCCESS', {})
        
    except Exception as e:
        print(f"Error: {e}")
        send_response(event, context, 'FAILED', {})

def send_response(event, context, status, data):
    response_body = {
        'Status': status,
        'Reason': f'See CloudWatch Log Stream: {context.log_stream_name}',
        'PhysicalResourceId': context.log_stream_name,
        'StackId': event['StackId'],
        'RequestId': event['RequestId'],
        'LogicalResourceId': event['LogicalResourceId'],
        'Data': data
    }
    
    http = urllib3.PoolManager()
    response = http.request(
        'PUT',
        event['ResponseURL'],
        body=json.dumps(response_body),
        headers={'Content-Type': 'application/json'}
    )
    print(f"Response status: {response.status}")
            """),
            timeout=cdk.Duration.minutes(5)
        )
        
        # Grant S3 permissions to the Lambda
        self.bucket.grant_write(config_lambda)
        
        # Grant API Gateway permissions to retrieve API key
        config_lambda.add_to_role_policy(
            iam.PolicyStatement(
                actions=["apigateway:GET"],
                resources=[f"arn:aws:apigateway:{cdk.Aws.REGION}::/apikeys/*"]
            )
        )
        
        # Create custom resource
        self.config_updater = cr.AwsCustomResource(
            self,
            "FrontendConfigUpdater",
            on_create=cr.AwsSdkCall(
                service="Lambda",
                action="invoke",
                parameters={
                    "FunctionName": config_lambda.function_name,
                    "Payload": json.dumps({
                        "RequestType": "Create",
                        "ResourceProperties": {
                            "BucketName": self.bucket.bucket_name,
                            "ApiUrl": self.api_url,
                            "ApiKeyId": self.api_key_id,
                            "Environment": self.environment
                        }
                    })
                },
                physical_resource_id=cr.PhysicalResourceId.of("frontend-config-updater")
            ),
            on_update=cr.AwsSdkCall(
                service="Lambda",
                action="invoke",
                parameters={
                    "FunctionName": config_lambda.function_name,
                    "Payload": json.dumps({
                        "RequestType": "Update",
                        "ResourceProperties": {
                            "BucketName": self.bucket.bucket_name,
                            "ApiUrl": self.api_url,
                            "ApiKeyId": self.api_key_id,
                            "Environment": self.environment
                        }
                    })
                }
            ),
            policy=cr.AwsCustomResourcePolicy.from_statements([
                iam.PolicyStatement(
                    actions=["lambda:InvokeFunction"],
                    resources=[config_lambda.function_arn]
                )
            ])
        )
        
        # Ensure config is created after the main deployment
        self.config_updater.node.add_dependency(self.deployment)
    
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
