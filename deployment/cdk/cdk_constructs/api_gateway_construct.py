"""
API Gateway construct for sentiment analysis deployment.
"""

from constructs import Construct
from aws_cdk import (
    aws_apigateway as apigateway,
    aws_lambda as _lambda,
    aws_iam as iam
)
from typing import Dict, Any

from config.api_config import ApiConfig, RequestValidation, ApiSecurity
from utils.common import get_resource_name, get_tags, validate_config


class SentimentApiGatewayConstruct(Construct):
    """CDK construct for creating the sentiment analysis API Gateway."""
    
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        lambda_function: _lambda.Function,
        config: ApiConfig,
        validation: RequestValidation = None,
        security: ApiSecurity = None,
        environment: str = "prod",
        **kwargs
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)
        
        # Validate configuration
        validate_config(config)
        
        self.config = config
        self.validation = validation or RequestValidation()
        self.security = security or ApiSecurity()
        self.environment = environment
        self.lambda_function = lambda_function
        
        # Create resources
        self._create_api_gateway()
        self._create_lambda_integration()
        self._create_request_models()
        self._create_endpoints()
        self._create_api_key_and_usage_plan()
    
    def _create_api_gateway(self) -> None:
        """Create the REST API Gateway."""
        api_name = get_resource_name(
            base_name="sentiment-analysis",
            resource_type="api",
            environment=self.environment
        )
        
        self.api = apigateway.RestApi(
            self,
            "RestApi",
            rest_api_name=api_name,
            description=self.config.description,
            default_cors_preflight_options=self.config.get_cors_options(),
            deploy_options=apigateway.StageOptions(
                stage_name=self.config.stage_name,
                throttling_rate_limit=self.config.throttle_rate_limit,
                throttling_burst_limit=self.config.throttle_burst_limit,
                logging_level=apigateway.MethodLoggingLevel.INFO,
                data_trace_enabled=True,
                metrics_enabled=True
            ),
            cloud_watch_role=True,
            endpoint_configuration=apigateway.EndpointConfiguration(
                types=[apigateway.EndpointType.REGIONAL]
            )
        )
        
        # Add tags
        tags = get_tags(
            environment=self.environment,
            additional_tags={"ResourceType": "API-Gateway"}
        )
        for key, value in tags.items():
            self.api.node.add_metadata(key, value)
    
    def _create_lambda_integration(self) -> None:
        """Create Lambda integration for API Gateway."""
        self.lambda_integration = apigateway.LambdaIntegration(
            self.lambda_function,
            request_templates={
                "application/json": '{"statusCode": "200"}'
            },
            integration_responses=[
                apigateway.IntegrationResponse(
                    status_code="200",
                    response_templates={
                        "application/json": ""
                    }
                ),
                apigateway.IntegrationResponse(
                    status_code="400",
                    selection_pattern=".*Bad Request.*",
                    response_templates={
                        "application/json": '{"error": "Bad Request"}'
                    }
                ),
                apigateway.IntegrationResponse(
                    status_code="500",
                    selection_pattern=".*Internal Server Error.*",
                    response_templates={
                        "application/json": '{"error": "Internal Server Error"}'
                    }
                )
            ],
            proxy=True,
            allow_test_invoke=True
        )
    
    def _create_request_models(self) -> None:
        """Create request/response models for validation."""
        # Predict request model
        self.predict_request_model = self.api.add_model(
            "PredictRequestModel",
            content_type="application/json",
            model_name="PredictRequest",
            schema=apigateway.JsonSchema(
                schema=apigateway.JsonSchemaVersion.DRAFT4,
                title="Predict Request",
                type=apigateway.JsonSchemaType.OBJECT,
                properties={
                    "text": apigateway.JsonSchema(
                        type=apigateway.JsonSchemaType.STRING,
                        min_length=1,
                        max_length=5000,
                        description="Text to analyze for sentiment"
                    ),
                    "model": apigateway.JsonSchema(
                        type=apigateway.JsonSchemaType.STRING,
                        enum=["lstm", "verbalizer", "both"],
                        description="Model to use for analysis"
                    )
                },
                required=["text"],
                additional_properties=False
            )
        )
        
        # Success response model
        self.success_response_model = self.api.add_model(
            "SuccessResponseModel",
            content_type="application/json",
            model_name="SuccessResponse",
            schema=apigateway.JsonSchema(
                schema=apigateway.JsonSchemaVersion.DRAFT4,
                title="Success Response",
                type=apigateway.JsonSchemaType.OBJECT,
                properties={
                    "text": apigateway.JsonSchema(type=apigateway.JsonSchemaType.STRING),
                    "predictions": apigateway.JsonSchema(type=apigateway.JsonSchemaType.OBJECT),
                    "consensus": apigateway.JsonSchema(type=apigateway.JsonSchemaType.OBJECT),
                    "total_processing_time_ms": apigateway.JsonSchema(type=apigateway.JsonSchemaType.NUMBER),
                    "models_available": apigateway.JsonSchema(type=apigateway.JsonSchemaType.NUMBER),
                    "version": apigateway.JsonSchema(type=apigateway.JsonSchemaType.STRING)
                }
            )
        )
        
        # Error response model
        self.error_response_model = self.api.add_model(
            "ErrorResponseModel",
            content_type="application/json",
            model_name="ErrorResponse",
            schema=apigateway.JsonSchema(
                schema=apigateway.JsonSchemaVersion.DRAFT4,
                title="Error Response",
                type=apigateway.JsonSchemaType.OBJECT,
                properties={
                    "error": apigateway.JsonSchema(type=apigateway.JsonSchemaType.STRING),
                    "message": apigateway.JsonSchema(type=apigateway.JsonSchemaType.STRING),
                    "request_id": apigateway.JsonSchema(type=apigateway.JsonSchemaType.STRING)
                }
            )
        )
    
    def _create_request_validator(self) -> apigateway.RequestValidator:
        """Create request validator."""
        return self.api.add_request_validator(
            "RequestValidator",
            validate_request_body=True,
            validate_request_parameters=True
        )
    
    def _create_endpoints(self) -> None:
        """Create API endpoints."""
        request_validator = self._create_request_validator()
        
        # Create /predict endpoint
        predict_resource = self.api.root.add_resource("predict")
        predict_method = predict_resource.add_method(
            "POST",
            self.lambda_integration,
            api_key_required=self.config.endpoints["predict"]["api_key_required"],
            request_validator=request_validator if self.config.endpoints["predict"]["request_validator"] else None,
            request_models={
                "application/json": self.predict_request_model
            },
            method_responses=[
                apigateway.MethodResponse(
                    status_code="200",
                    response_models={
                        "application/json": self.success_response_model
                    },
                    response_parameters={
                        "method.response.header.Access-Control-Allow-Origin": True,
                        "method.response.header.Access-Control-Allow-Headers": True,
                        "method.response.header.Access-Control-Allow-Methods": True
                    }
                ),
                apigateway.MethodResponse(
                    status_code="400",
                    response_models={
                        "application/json": self.error_response_model
                    }
                ),
                apigateway.MethodResponse(
                    status_code="500",
                    response_models={
                        "application/json": self.error_response_model
                    }
                )
            ]
        )
        
        # Create /health endpoint
        health_resource = self.api.root.add_resource("health")
        health_method = health_resource.add_method(
            "GET",
            self.lambda_integration,
            api_key_required=self.config.endpoints["health"]["api_key_required"],
            method_responses=[
                apigateway.MethodResponse(
                    status_code="200",
                    response_models={
                        "application/json": apigateway.Model.EMPTY_MODEL
                    }
                ),
                apigateway.MethodResponse(
                    status_code="503",
                    response_models={
                        "application/json": self.error_response_model
                    }
                )
            ]
        )
        
        self.predict_method = predict_method
        self.health_method = health_method
    
    def _create_api_key_and_usage_plan(self) -> None:
        """Create API key and usage plan."""
        # Create API Key
        self.api_key = self.api.add_api_key(
            "ApiKey",
            api_key_name=get_resource_name(
                base_name="sentiment-analysis",
                resource_type="api-key",
                environment=self.environment
            ),
            description=self.config.api_key_description
        )
        
        # Create Usage Plan
        usage_plan_name = get_resource_name(
            base_name="sentiment-analysis",
            resource_type="usage-plan",
            environment=self.environment
        )
        
        self.usage_plan = self.api.add_usage_plan(
            "UsagePlan",
            name=usage_plan_name,
            description=f"Usage plan for {self.config.api_name}",
            throttle=self.config.get_throttle_settings(),
            quota=self.config.get_quota_settings(),
            api_stages=[
                apigateway.UsagePlanPerApiStage(
                    api=self.api,
                    stage=self.api.deployment_stage
                )
            ]
        )
        
        # Associate API key with usage plan
        self.usage_plan.add_api_key(self.api_key)
    
    @property
    def rest_api(self) -> apigateway.RestApi:
        """Get the REST API."""
        return self.api
    
    @property
    def api_url(self) -> str:
        """Get the API URL."""
        return self.api.url
    
    @property
    def api_key_id(self) -> str:
        """Get the API key ID."""
        return self.api_key.key_id
    
    @property
    def predict_endpoint_url(self) -> str:
        """Get the predict endpoint URL."""
        return f"{self.api.url}predict"
    
    @property
    def health_endpoint_url(self) -> str:
        """Get the health endpoint URL."""
        return f"{self.api.url}health"
    
    def add_cors_options(self, resource: apigateway.Resource) -> None:
        """Add CORS options to a resource."""
        resource.add_cors_preflight(
            allow_origins=self.config.cors_allow_origins,
            allow_methods=self.config.cors_allow_methods,
            allow_headers=self.config.cors_allow_headers
        )
    
    def get_api_metrics(self) -> Dict[str, Any]:
        """Get CloudWatch metrics for the API."""
        return {
            "count": self.api.metric_count(),
            "latency": self.api.metric_latency(),
            "client_error": self.api.metric_client_error(),
            "server_error": self.api.metric_server_error()
        }
