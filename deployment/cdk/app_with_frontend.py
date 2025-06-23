"""
CDK App for deploying both API and Frontend stacks.

This app creates two separate stacks:
1. SentimentAnalysisStack - API, Lambda, and monitoring
2. SentimentAnalysisFrontendStack - React frontend with S3 and CloudFront

The frontend stack depends on the API stack outputs.
"""

import aws_cdk as cdk
from sentiment_stack import SentimentAnalysisStack
from frontend_stack import SentimentAnalysisFrontendStack


def main():
    """Main function to create and deploy the CDK app."""
    
    app = cdk.App()
    
    # Get environment from context or default to 'prod'
    environment = app.node.try_get_context("environment") or "prod"
    alarm_email = app.node.try_get_context("alarm_email")
    
    # Create the API stack first
    api_stack = SentimentAnalysisStack(
        app,
        f"SentimentAnalysisStack-{environment}",
        environment=environment,
        alarm_email=alarm_email,
        env=cdk.Environment(
            account=app.node.try_get_context("account"),
            region=app.node.try_get_context("region") or "us-east-1"
        ),
        description=f"Sentiment Analysis API Stack - {environment}"
    )
    
    # Create the frontend stack that depends on the API stack
    frontend_stack = SentimentAnalysisFrontendStack(
        app,
        f"SentimentAnalysisFrontendStack-{environment}",
        environment=environment,
        api_stack_name=api_stack.stack_name,
        env=cdk.Environment(
            account=app.node.try_get_context("account"),
            region=app.node.try_get_context("region") or "us-east-1"
        ),
        description=f"Sentiment Analysis Frontend Stack - {environment}"
    )
    
    # Add dependency to ensure API stack deploys first
    frontend_stack.add_dependency(api_stack)
    
    # Add stack-level tags
    cdk.Tags.of(api_stack).add("Project", "SentimentAnalysis")
    cdk.Tags.of(api_stack).add("Component", "API")
    cdk.Tags.of(frontend_stack).add("Project", "SentimentAnalysis")
    cdk.Tags.of(frontend_stack).add("Component", "Frontend")
    
    app.synth()


if __name__ == "__main__":
    main()
