"""
Main CDK Stack for Sentiment Analysis API.

This stack orchestrates all the constructs to create a complete
sentiment analysis API deployment with Lambda, API Gateway, and monitoring.
"""

import aws_cdk as cdk
from aws_cdk import Stack
from constructs import Construct

# Import our modular constructs
from cdk_constructs import (
    SentimentLambdaConstruct,
    SentimentApiGatewayConstruct,
    SentimentMonitoringConstruct
)

# Import configuration classes
from config import (
    LambdaConfig,
    ApiConfig,
    MonitoringConfig,
    LambdaPermissions
)
from config.api_config import RequestValidation, ApiSecurity
from config.monitoring_config import AlarmConfig, DashboardConfig

# Import utilities
from utils.common import get_tags, get_environment_config, merge_configs


class SentimentAnalysisStack(Stack):
    """
    Main CDK Stack for Sentiment Analysis API.
    
    This stack creates:
    - Lambda function with Docker container (LSTM + Verbalizer models)
    - API Gateway with API Key authentication
    - CloudWatch monitoring and alarms
    - SNS notifications (optional)
    - CloudWatch dashboard
    """

    def __init__(
        self, 
        scope: Construct, 
        construct_id: str,
        environment: str = "prod",
        alarm_email: str = None,
        **kwargs
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)
        
        self.env_name = environment
        self.alarm_email = alarm_email
        
        # Get environment-specific configuration
        env_config = get_environment_config(environment)
        
        # Create configurations
        self._create_configurations(env_config)
        
        # Create constructs
        self._create_lambda_construct()
        self._create_api_gateway_construct()
        self._create_monitoring_construct()
        
        # Create stack outputs
        self._create_outputs()
        
        # Add stack tags
        self._add_stack_tags()
    
    def _create_configurations(self, env_config: dict) -> None:
        """Create configuration objects for all constructs."""
        
        # Lambda configuration
        self.lambda_config = LambdaConfig(
            function_name="sentiment-analysis",
            memory_size=env_config.get("memory_size", 2048),
            timeout=cdk.Duration.minutes(env_config.get("timeout_minutes", 5)),
            log_retention_days=env_config.get("log_retention_days", 7),
            environment_vars={
                "PYTHONPATH": "/var/task",
                "MODEL_PATH": "/var/task/models",  # Fixed path issue
                "LOG_LEVEL": "INFO",
                "ENVIRONMENT": self.env_name
            }
        )
        
        # Lambda permissions
        self.lambda_permissions = LambdaPermissions(
            basic_execution=True,
            cloudwatch_logs=True
        )
        
        # API Gateway configuration
        self.api_config = ApiConfig(
            api_name="Sentiment Analysis API",
            throttle_rate_limit=env_config.get("throttle_rate_limit", 100),
            quota_limit=env_config.get("quota_limit", 10000),
            cors_allow_origins=["*"] if self.env_name == "dev" else ["*"]  # Restrict in production
        )
        
        # API validation and security
        self.api_validation = RequestValidation()
        self.api_security = ApiSecurity()
        
        # Monitoring configuration
        self.monitoring_config = MonitoringConfig(
            enable_alarms=env_config.get("enable_alarms", True),
            enable_dashboard=env_config.get("enable_dashboard", True),
            alarm_email=self.alarm_email
        )
        
        # Alarm and dashboard configurations
        self.alarm_config = AlarmConfig()
        self.dashboard_config = DashboardConfig()
        
        # Validate all configurations
        for config in [self.lambda_config, self.api_config, self.monitoring_config]:
            if hasattr(config, 'validate'):
                config.validate()
    
    def _create_lambda_construct(self) -> None:
        """Create the Lambda construct."""
        self.lambda_construct = SentimentLambdaConstruct(
            self,
            "LambdaConstruct",
            config=self.lambda_config,
            permissions=self.lambda_permissions,
            environment=self.env_name
        )
        
        # Store reference to Lambda function for other constructs
        self.lambda_function = self.lambda_construct.function
    
    def _create_api_gateway_construct(self) -> None:
        """Create the API Gateway construct."""
        self.api_gateway_construct = SentimentApiGatewayConstruct(
            self,
            "ApiGatewayConstruct",
            lambda_function=self.lambda_function,
            config=self.api_config,
            validation=self.api_validation,
            security=self.api_security,
            environment=self.env_name
        )
        
        # Store reference to API Gateway for other constructs
        self.api_gateway = self.api_gateway_construct.rest_api
    
    def _create_monitoring_construct(self) -> None:
        """Create the monitoring construct."""
        if self.monitoring_config.enable_alarms or self.monitoring_config.enable_dashboard:
            self.monitoring_construct = SentimentMonitoringConstruct(
                self,
                "MonitoringConstruct",
                lambda_function=self.lambda_function,
                api_gateway=self.api_gateway,
                config=self.monitoring_config,
                alarm_config=self.alarm_config,
                dashboard_config=self.dashboard_config,
                environment=self.env_name
            )
        else:
            self.monitoring_construct = None
    
    def _create_outputs(self) -> None:
        """Create CloudFormation outputs."""
        
        # API Gateway outputs
        cdk.CfnOutput(
            self,
            "ApiUrl",
            value=self.api_gateway_construct.api_url,
            description="API Gateway URL",
            export_name=f"SentimentAnalysis-{self.env_name}-ApiUrl"
        )
        
        cdk.CfnOutput(
            self,
            "PredictEndpoint",
            value=self.api_gateway_construct.predict_endpoint_url,
            description="Prediction endpoint URL",
            export_name=f"SentimentAnalysis-{self.env_name}-PredictEndpoint"
        )
        
        cdk.CfnOutput(
            self,
            "HealthEndpoint",
            value=self.api_gateway_construct.health_endpoint_url,
            description="Health check endpoint URL",
            export_name=f"SentimentAnalysis-{self.env_name}-HealthEndpoint"
        )
        
        # API Key outputs
        cdk.CfnOutput(
            self,
            "ApiKeyId",
            value=self.api_gateway_construct.api_key_id,
            description="API Key ID (use AWS CLI to get the actual key value)",
            export_name=f"SentimentAnalysis-{self.env_name}-ApiKeyId"
        )
        
        cdk.CfnOutput(
            self,
            "ApiKeyCommand",
            value=f"aws apigateway get-api-key --api-key {self.api_gateway_construct.api_key_id} --include-value --query 'value' --output text",
            description="Command to retrieve API key value"
        )
        
        # Lambda outputs
        cdk.CfnOutput(
            self,
            "LambdaFunctionName",
            value=self.lambda_construct.function_name,
            description="Lambda function name",
            export_name=f"SentimentAnalysis-{self.env_name}-LambdaFunctionName"
        )
        
        cdk.CfnOutput(
            self,
            "LambdaFunctionArn",
            value=self.lambda_construct.function_arn,
            description="Lambda function ARN",
            export_name=f"SentimentAnalysis-{self.env_name}-LambdaFunctionArn"
        )
        
        # Monitoring outputs
        if self.monitoring_construct:
            if hasattr(self.monitoring_construct, 'dashboard'):
                cdk.CfnOutput(
                    self,
                    "DashboardUrl",
                    value=self.monitoring_construct.dashboard_url or "Dashboard not created",
                    description="CloudWatch Dashboard URL"
                )
            
            if hasattr(self.monitoring_construct, 'sns_topic') and self.monitoring_construct.sns_topic:
                cdk.CfnOutput(
                    self,
                    "AlertsTopicArn",
                    value=self.monitoring_construct.sns_topic.topic_arn,
                    description="SNS Topic ARN for alerts"
                )
        
        # Usage instructions
        cdk.CfnOutput(
            self,
            "UsageInstructions",
            value=f"""
Test your API:
1. Get API key: aws apigateway get-api-key --api-key {self.api_gateway_construct.api_key_id} --include-value --query 'value' --output text
2. Test health: curl {self.api_gateway_construct.health_endpoint_url}
3. Test prediction: curl -X POST {self.api_gateway_construct.predict_endpoint_url} -H "Content-Type: application/json" -H "x-api-key: YOUR_API_KEY" -d '{{"text": "This movie was great!", "model": "both"}}'
            """.strip(),
            description="Instructions for testing the deployed API"
        )
    
    def _add_stack_tags(self) -> None:
        """Add tags to the entire stack."""
        tags = get_tags(
            environment=self.env_name,
            additional_tags={
                "StackName": self.stack_name,
                "Service": "sentiment-analysis-api",
                "ModelTypes": "LSTM,Verbalizer",
                "DeploymentMethod": "CDK"
            }
        )
        
        for key, value in tags.items():
            cdk.Tags.of(self).add(key, value)
    
    # Public properties for accessing constructs
    @property
    def lambda_function_construct(self) -> SentimentLambdaConstruct:
        """Get the Lambda construct."""
        return self.lambda_construct
    
    @property
    def api_gateway_construct_ref(self) -> SentimentApiGatewayConstruct:
        """Get the API Gateway construct."""
        return self.api_gateway_construct
    
    @property
    def monitoring_construct_ref(self) -> SentimentMonitoringConstruct:
        """Get the monitoring construct."""
        return self.monitoring_construct
    
    # Convenience methods
    def get_api_key_value_command(self) -> str:
        """Get the AWS CLI command to retrieve the API key value."""
        return f"aws apigateway get-api-key --api-key {self.api_gateway_construct.api_key_id} --include-value --query 'value' --output text"
    
    def get_test_commands(self) -> dict:
        """Get test commands for the deployed API."""
        return {
            "health_check": f"curl {self.api_gateway_construct.health_endpoint_url}",
            "prediction_test": f"""curl -X POST {self.api_gateway_construct.predict_endpoint_url} \\
  -H "Content-Type: application/json" \\
  -H "x-api-key: YOUR_API_KEY" \\
  -d '{{"text": "This movie was absolutely fantastic!", "model": "both"}}'""",
            "get_api_key": self.get_api_key_value_command()
        }
    
    def get_deployment_summary(self) -> dict:
        """Get a summary of the deployed resources."""
        return {
            "environment": self.env_name,
            "lambda_function": self.lambda_construct.function_name,
            "api_url": self.api_gateway_construct.api_url,
            "api_key_id": self.api_gateway_construct.api_key_id,
            "monitoring_enabled": self.monitoring_construct is not None,
            "dashboard_url": self.monitoring_construct.dashboard_url if self.monitoring_construct else None,
            "models": ["LSTM", "Verbalizer"],
            "endpoints": {
                "predict": self.api_gateway_construct.predict_endpoint_url,
                "health": self.api_gateway_construct.health_endpoint_url
            }
        }
