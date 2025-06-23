"""
Dedicated CDK Stack for Sentiment Analysis Frontend.

This stack handles the React frontend deployment independently from the API stack,
eliminating dependency issues and simplifying the deployment process.
"""

import aws_cdk as cdk
from aws_cdk import Stack
from constructs import Construct

# Import our simplified frontend construct
from cdk_constructs.simplified_frontend_construct import SimplifiedSentimentFrontendConstruct

# Import configuration classes
from config.frontend_config import (
    get_frontend_config,
    get_build_config,
    get_deployment_config
)

# Import utilities
from utils.common import get_tags


class SentimentAnalysisFrontendStack(Stack):
    """
    Dedicated CDK Stack for the React Frontend.
    
    This stack creates:
    - S3 bucket for hosting static files
    - CloudFront distribution for global CDN
    - Build process for React application
    - Proper caching and security configurations
    
    It depends on the API stack outputs for API URL and API Key.
    """

    def __init__(
        self, 
        scope: Construct, 
        construct_id: str,
        environment: str = "prod",
        api_stack_name: str = "SentimentAnalysisStack",
        **kwargs
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)
        
        self.env_name = environment
        self.api_stack_name = api_stack_name
        
        # Import API details from the API stack
        self._import_api_details()
        
        # Get frontend configurations
        self._create_configurations()
        
        # Create frontend construct
        self._create_frontend_construct()
        
        # Create stack outputs
        self._create_outputs()
        
        # Add stack tags
        self._add_stack_tags()
    
    def _import_api_details(self) -> None:
        """Import API URL and API Key from the API stack."""
        
        # Import API URL from the API stack export (with environment suffix)
        self.api_url = cdk.Fn.import_value(
            f"SentimentAnalysis-{self.env_name}-ApiUrl"
        )
        
        # Import API Key ID from the API stack export (with environment suffix)
        self.api_key_id = cdk.Fn.import_value(
            f"SentimentAnalysis-{self.env_name}-ApiKeyId"
        )
        
        print(f"Frontend stack will use API from: {self.api_stack_name}")
    
    def _create_configurations(self) -> None:
        """Create configuration objects for the frontend."""
        
        # Get frontend configurations
        self.frontend_config = get_frontend_config(self.env_name)
        self.build_config = get_build_config(self.env_name)
        self.deployment_config = get_deployment_config(self.env_name)
        
        # Validate all configurations
        for config in [self.frontend_config, self.build_config, self.deployment_config]:
            if hasattr(config, 'validate'):
                config.validate()
    
    def _create_frontend_construct(self) -> None:
        """Create the frontend construct."""
        
        self.frontend_construct = SimplifiedSentimentFrontendConstruct(
            self,
            "FrontendConstruct",
            api_url=self.api_url,
            api_key_id=self.api_key_id,
            frontend_config=self.frontend_config,
            build_config=self.build_config,
            deployment_config=self.deployment_config,
            environment=self.env_name
        )
    
    def _create_outputs(self) -> None:
        """Create CloudFormation outputs."""
        
        # Frontend URL
        cdk.CfnOutput(
            self,
            "FrontendUrl",
            value=self.frontend_construct.website_url,
            description="Frontend website URL",
            export_name=f"SentimentAnalysis-{self.env_name}-FrontendUrl"
        )
        
        # S3 Bucket
        cdk.CfnOutput(
            self,
            "FrontendBucket",
            value=self.frontend_construct.bucket_name,
            description="S3 bucket hosting the frontend",
            export_name=f"SentimentAnalysis-{self.env_name}-FrontendBucket"
        )
        
        # CloudFront Distribution (if enabled)
        if self.frontend_construct.distribution_id:
            cdk.CfnOutput(
                self,
                "CloudFrontDistributionId",
                value=self.frontend_construct.distribution_id,
                description="CloudFront distribution ID",
                export_name=f"SentimentAnalysis-{self.env_name}-CloudFrontDistributionId"
            )
            
            cdk.CfnOutput(
                self,
                "CloudFrontDomainName",
                value=self.frontend_construct.distribution_domain_name,
                description="CloudFront distribution domain name"
            )
        
        # Deployment information
        cdk.CfnOutput(
            self,
            "DeploymentInfo",
            value=f"""
Frontend deployed successfully!
- Website URL: {self.frontend_construct.website_url}
- S3 Bucket: {self.frontend_construct.bucket_name}
- Environment: {self.env_name}
- API Integration: Configured via runtime injection (env.js)
            """.strip(),
            description="Frontend deployment summary"
        )
    
    def _add_stack_tags(self) -> None:
        """Add tags to the entire stack."""
        tags = get_tags(
            environment=self.env_name,
            additional_tags={
                "StackName": self.stack_name,
                "Service": "sentiment-analysis-frontend",
                "Component": "react-spa",
                "DeploymentMethod": "CDK"
            }
        )
        
        for key, value in tags.items():
            cdk.Tags.of(self).add(key, value)
    
    # Public properties
    @property
    def frontend_construct_ref(self) -> SimplifiedSentimentFrontendConstruct:
        """Get the frontend construct."""
        return self.frontend_construct
    
    @property
    def website_url(self) -> str:
        """Get the website URL."""
        return self.frontend_construct.website_url
    
    @property
    def bucket_name(self) -> str:
        """Get the S3 bucket name."""
        return self.frontend_construct.bucket_name
    
    def get_deployment_summary(self) -> dict:
        """Get a summary of the deployed frontend resources."""
        return {
            "environment": self.env_name,
            "website_url": self.website_url,
            "bucket_name": self.bucket_name,
            "distribution_id": self.frontend_construct.distribution_id,
            "cloudfront_enabled": hasattr(self.frontend_construct, 'distribution'),
            "api_integration": "Build-time configuration"
        }
