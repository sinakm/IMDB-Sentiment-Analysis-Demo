"""
Lambda construct for sentiment analysis deployment.
"""

from constructs import Construct
from aws_cdk import (
    aws_lambda as _lambda,
    aws_iam as iam,
    aws_logs as logs,
    RemovalPolicy
)
from typing import Optional

from config.lambda_config import LambdaConfig, LambdaPermissions
from utils.common import get_resource_name, get_tags, validate_config


class SentimentLambdaConstruct(Construct):
    """CDK construct for creating the sentiment analysis Lambda function."""
    
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        config: LambdaConfig,
        permissions: Optional[LambdaPermissions] = None,
        environment: str = "prod",
        **kwargs
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)
        
        # Validate configuration
        validate_config(config)
        
        self.config = config
        self.permissions = permissions or LambdaPermissions()
        self.environment = environment
        
        # Create resources
        self._create_execution_role()
        self._create_log_group()
        self._create_lambda_function()
    
    def _create_execution_role(self) -> None:
        """Create IAM execution role for Lambda function."""
        role_name = get_resource_name(
            base_name="sentiment-analysis",
            resource_type="lambda-role",
            environment=self.environment
        )
        
        # Create the execution role
        self.execution_role = iam.Role(
            self,
            "ExecutionRole",
            role_name=role_name,
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            description="Execution role for sentiment analysis Lambda function",
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(policy)
                for policy in self.permissions.get_managed_policies()
            ]
        )
        
        # Add inline policies if any
        inline_policies = self.permissions.get_inline_policies()
        for policy_name, policy_document in inline_policies.items():
            self.execution_role.add_to_policy(
                iam.PolicyStatement.from_json(policy_document["Statement"][0])
            )
        
        # Add tags
        tags = get_tags(
            environment=self.environment,
            additional_tags={"ResourceType": "IAM-Role"}
        )
        for key, value in tags.items():
            self.execution_role.add_to_policy(
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    actions=["tag:GetResources"],
                    resources=["*"]
                )
            )
    
    def _create_log_group(self) -> None:
        """Create CloudWatch log group for Lambda function."""
        log_group_name = get_resource_name(
            base_name="sentiment-analysis",
            resource_type="lambda-logs",
            environment=self.environment
        )
        
        self.log_group = logs.LogGroup(
            self,
            "LogGroup",
            log_group_name=f"/aws/lambda/{self.config.function_name}",
            retention=getattr(logs.RetentionDays, f"_{self.config.log_retention_days}_DAYS", logs.RetentionDays.ONE_WEEK),
            removal_policy=RemovalPolicy.DESTROY
        )
    
    def _create_lambda_function(self) -> None:
        """Create the Lambda function."""
        function_name = get_resource_name(
            base_name="sentiment-analysis",
            resource_type="function",
            environment=self.environment
        )
        
        # Create Lambda function from Docker image
        self.lambda_function = _lambda.DockerImageFunction(
            self,
            "Function",
            function_name=function_name,
            code=_lambda.DockerImageCode.from_image_asset(
                directory=self.config.models_source_path,
                build_args=self.config.docker_build_args
            ),
            memory_size=self.config.memory_size,
            timeout=self.config.timeout,
            role=self.execution_role,
            log_group=self.log_group,
            description=self.config.description,
            environment=self.config.environment_vars,
            architecture=_lambda.Architecture.X86_64
        )
        
        # Add tags to Lambda function
        tags = get_tags(
            environment=self.environment,
            additional_tags={
                "ResourceType": "Lambda-Function",
                "ModelTypes": "LSTM,Verbalizer"
            }
        )
        for key, value in tags.items():
            self.lambda_function.add_environment(f"TAG_{key.upper()}", value)
    
    @property
    def function(self) -> _lambda.DockerImageFunction:
        """Get the Lambda function."""
        return self.lambda_function
    
    @property
    def function_arn(self) -> str:
        """Get the Lambda function ARN."""
        return self.lambda_function.function_arn
    
    @property
    def function_name(self) -> str:
        """Get the Lambda function name."""
        return self.lambda_function.function_name
    
    def add_environment_variable(self, key: str, value: str) -> None:
        """Add an environment variable to the Lambda function."""
        self.lambda_function.add_environment(key, value)
    
    def grant_invoke(self, principal: iam.IPrincipal) -> iam.Grant:
        """Grant invoke permission to a principal."""
        return self.lambda_function.grant_invoke(principal)
    
    def add_permission(self, id: str, principal: iam.IPrincipal, action: str = "lambda:InvokeFunction") -> None:
        """Add a resource-based permission to the Lambda function."""
        self.lambda_function.add_permission(
            id,
            principal=principal,
            action=action
        )
    
    def get_function_url(self) -> Optional[str]:
        """Get function URL if configured."""
        # This would be implemented if function URLs are needed
        return None
    
    def add_event_source(self, source: _lambda.IEventSource) -> None:
        """Add an event source to the Lambda function."""
        self.lambda_function.add_event_source(source)
    
    def add_layers(self, *layers: _lambda.ILayerVersion) -> None:
        """Add Lambda layers to the function."""
        # Note: Docker-based functions don't support layers
        # This is here for interface compatibility
        pass
    
    def get_metrics(self) -> dict:
        """Get CloudWatch metrics for the Lambda function."""
        return {
            "invocations": self.lambda_function.metric_invocations(),
            "errors": self.lambda_function.metric_errors(),
            "duration": self.lambda_function.metric_duration(),
            "throttles": self.lambda_function.metric_throttles()
        }
