"""
API Gateway configuration for sentiment analysis deployment.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from aws_cdk import aws_apigateway as apigateway


@dataclass
class ApiConfig:
    """Configuration for API Gateway deployment."""
    
    # Basic API settings
    api_name: str = "Sentiment Analysis API"
    description: str = "API for sentiment analysis using multiple ML models"
    stage_name: str = "prod"
    
    # CORS configuration
    cors_allow_origins: List[str] = field(default_factory=lambda: ["*"])  # Restrict in production
    cors_allow_methods: List[str] = field(default_factory=lambda: ["GET", "POST", "OPTIONS"])
    cors_allow_headers: List[str] = field(default_factory=lambda: [
        "Content-Type", 
        "X-Amz-Date", 
        "Authorization", 
        "X-Api-Key", 
        "X-Amz-Security-Token"
    ])
    
    # Throttling configuration
    throttle_rate_limit: int = 100  # requests per second
    throttle_burst_limit: int = 200  # burst capacity
    
    # Usage plan configuration
    quota_limit: int = 10000  # requests per month
    quota_period: apigateway.Period = apigateway.Period.MONTH
    
    # API Key configuration
    api_key_name: str = "sentiment-analysis-key"
    api_key_description: str = "API key for sentiment analysis service"
    
    # Endpoint configuration
    endpoints: Dict[str, Dict] = field(default_factory=lambda: {
        "predict": {
            "path": "predict",
            "method": "POST",
            "api_key_required": True,
            "description": "Run sentiment analysis on text",
            "request_validator": True
        },
        "health": {
            "path": "health", 
            "method": "GET",
            "api_key_required": False,
            "description": "Health check endpoint",
            "request_validator": False
        }
    })
    
    def get_cors_options(self) -> apigateway.CorsOptions:
        """Get CORS configuration for API Gateway."""
        return apigateway.CorsOptions(
            allow_origins=self.cors_allow_origins,
            allow_methods=self.cors_allow_methods,
            allow_headers=self.cors_allow_headers
        )
    
    def get_throttle_settings(self) -> apigateway.ThrottleSettings:
        """Get throttling configuration."""
        return apigateway.ThrottleSettings(
            rate_limit=self.throttle_rate_limit,
            burst_limit=self.throttle_burst_limit
        )
    
    def get_quota_settings(self) -> apigateway.QuotaSettings:
        """Get quota configuration."""
        return apigateway.QuotaSettings(
            limit=self.quota_limit,
            period=self.quota_period
        )
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.throttle_rate_limit <= 0:
            raise ValueError("Throttle rate limit must be positive")
        
        if self.throttle_burst_limit < self.throttle_rate_limit:
            raise ValueError("Burst limit must be >= rate limit")
        
        if self.quota_limit <= 0:
            raise ValueError("Quota limit must be positive")
        
        if not self.api_name or len(self.api_name.strip()) == 0:
            raise ValueError("API name cannot be empty")


@dataclass
class RequestValidation:
    """Request validation configuration for API endpoints."""
    
    # Predict endpoint validation
    predict_request_schema: Dict = field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "minLength": 1,
                "maxLength": 5000,
                "description": "Text to analyze for sentiment"
            },
            "model": {
                "type": "string",
                "enum": ["lstm", "verbalizer", "both"],
                "description": "Model to use for analysis"
            }
        },
        "required": ["text"],
        "additionalProperties": False
    })
    
    # Response schemas
    success_response_schema: Dict = field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "predictions": {"type": "object"},
            "consensus": {"type": "object"},
            "total_processing_time_ms": {"type": "number"},
            "models_available": {"type": "number"},
            "version": {"type": "string"}
        }
    })
    
    error_response_schema: Dict = field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "error": {"type": "string"},
            "message": {"type": "string"},
            "request_id": {"type": "string"}
        }
    })
    
    def get_request_validator_options(self) -> Dict[str, bool]:
        """Get request validator configuration."""
        return {
            "validate_request_body": True,
            "validate_request_parameters": True
        }


@dataclass
class ApiSecurity:
    """Security configuration for API Gateway."""
    
    # API Key settings
    enable_api_key: bool = True
    api_key_source: apigateway.ApiKeySourceType = apigateway.ApiKeySourceType.HEADER
    
    # Rate limiting
    enable_throttling: bool = True
    
    # Request size limits
    max_request_size: int = 1024 * 1024  # 1MB
    
    # Additional security headers
    security_headers: Dict[str, str] = field(default_factory=lambda: {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains"
    })
    
    def validate(self) -> None:
        """Validate security configuration."""
        if self.max_request_size <= 0:
            raise ValueError("Max request size must be positive")
        
        if self.max_request_size > 10 * 1024 * 1024:  # 10MB Lambda limit
            raise ValueError("Max request size cannot exceed 10MB")
