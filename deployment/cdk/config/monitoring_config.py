"""
Monitoring configuration for sentiment analysis deployment.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from aws_cdk import aws_logs as logs, aws_cloudwatch as cloudwatch, Duration


@dataclass
class MonitoringConfig:
    """Configuration for CloudWatch monitoring and logging."""
    
    # Log group settings
    log_group_name: str = "/aws/lambda/sentiment-analysis"
    log_retention: logs.RetentionDays = logs.RetentionDays.ONE_WEEK
    
    # Metrics configuration
    enable_custom_metrics: bool = True
    metrics_namespace: str = "SentimentAnalysis"
    
    # Alarm configuration
    enable_alarms: bool = True
    alarm_email: Optional[str] = None  # SNS topic email for notifications
    
    # Dashboard configuration
    enable_dashboard: bool = True
    dashboard_name: str = "SentimentAnalysisDashboard"
    
    def validate(self) -> None:
        """Validate monitoring configuration."""
        if not self.log_group_name.startswith("/aws/lambda/"):
            raise ValueError("Log group name must start with '/aws/lambda/'")
        
        if self.alarm_email and "@" not in self.alarm_email:
            raise ValueError("Invalid email format for alarm notifications")


@dataclass
class AlarmConfig:
    """Configuration for CloudWatch alarms."""
    
    # Error rate alarm
    error_rate_threshold: float = 5.0  # Percentage
    error_rate_evaluation_periods: int = 2
    error_rate_datapoints_to_alarm: int = 2
    
    # Duration alarm
    duration_threshold_ms: float = 30000  # 30 seconds
    duration_evaluation_periods: int = 3
    duration_datapoints_to_alarm: int = 2
    
    # Memory utilization alarm
    memory_utilization_threshold: float = 80.0  # Percentage
    memory_evaluation_periods: int = 3
    memory_datapoints_to_alarm: int = 2
    
    # Throttle alarm
    throttle_threshold: float = 1.0  # Any throttling
    throttle_evaluation_periods: int = 1
    throttle_datapoints_to_alarm: int = 1
    
    # Invocation count alarm (for cost monitoring)
    invocation_threshold: float = 1000.0  # Daily invocations
    invocation_evaluation_periods: int = 1
    invocation_datapoints_to_alarm: int = 1
    
    def get_alarm_configs(self) -> List[Dict]:
        """Get list of alarm configurations."""
        return [
            {
                "name": "ErrorRate",
                "description": "High error rate in sentiment analysis function",
                "metric_name": "Errors",
                "statistic": cloudwatch.Statistic.SUM,
                "threshold": self.error_rate_threshold,
                "comparison_operator": cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
                "evaluation_periods": self.error_rate_evaluation_periods,
                "datapoints_to_alarm": self.error_rate_datapoints_to_alarm,
                "period": Duration.minutes(5),
                "treat_missing_data": cloudwatch.TreatMissingData.NOT_BREACHING
            },
            {
                "name": "Duration",
                "description": "High execution duration in sentiment analysis function",
                "metric_name": "Duration",
                "statistic": cloudwatch.Statistic.AVERAGE,
                "threshold": self.duration_threshold_ms,
                "comparison_operator": cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
                "evaluation_periods": self.duration_evaluation_periods,
                "datapoints_to_alarm": self.duration_datapoints_to_alarm,
                "period": Duration.minutes(5),
                "treat_missing_data": cloudwatch.TreatMissingData.NOT_BREACHING
            },
            {
                "name": "Throttles",
                "description": "Function throttling detected",
                "metric_name": "Throttles",
                "statistic": cloudwatch.Statistic.SUM,
                "threshold": self.throttle_threshold,
                "comparison_operator": cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
                "evaluation_periods": self.throttle_evaluation_periods,
                "datapoints_to_alarm": self.throttle_datapoints_to_alarm,
                "period": Duration.minutes(1),
                "treat_missing_data": cloudwatch.TreatMissingData.NOT_BREACHING
            }
        ]


@dataclass
class DashboardConfig:
    """Configuration for CloudWatch dashboard."""
    
    # Widget dimensions
    widget_width: int = 12
    widget_height: int = 6
    
    # Time range
    default_period: Duration = Duration.minutes(5)
    
    # Metrics to display
    lambda_metrics: List[str] = field(default_factory=lambda: [
        "Invocations",
        "Duration", 
        "Errors",
        "Throttles",
        "ConcurrentExecutions"
    ])
    
    api_gateway_metrics: List[str] = field(default_factory=lambda: [
        "Count",
        "Latency",
        "4XXError",
        "5XXError"
    ])
    
    custom_metrics: List[str] = field(default_factory=lambda: [
        "ModelInferenceTime",
        "ModelAccuracy",
        "RequestSize"
    ])
    
    def get_widget_configs(self) -> List[Dict]:
        """Get dashboard widget configurations."""
        widgets = []
        
        # Lambda metrics widget
        widgets.append({
            "type": "metric",
            "title": "Lambda Function Metrics",
            "metrics": self.lambda_metrics,
            "width": self.widget_width,
            "height": self.widget_height,
            "period": self.default_period
        })
        
        # API Gateway metrics widget
        widgets.append({
            "type": "metric", 
            "title": "API Gateway Metrics",
            "metrics": self.api_gateway_metrics,
            "width": self.widget_width,
            "height": self.widget_height,
            "period": self.default_period
        })
        
        # Custom metrics widget
        if self.custom_metrics:
            widgets.append({
                "type": "metric",
                "title": "Custom Application Metrics", 
                "metrics": self.custom_metrics,
                "width": self.widget_width,
                "height": self.widget_height,
                "period": self.default_period
            })
        
        # Log insights widget
        widgets.append({
            "type": "log",
            "title": "Recent Errors",
            "query": """
                fields @timestamp, @message
                | filter @message like /ERROR/
                | sort @timestamp desc
                | limit 20
            """,
            "width": self.widget_width * 2,
            "height": self.widget_height,
            "region": "us-east-1"
        })
        
        return widgets


@dataclass
class LoggingConfig:
    """Configuration for application logging."""
    
    # Log levels
    default_log_level: str = "INFO"
    lambda_log_level: str = "INFO"
    
    # Log format
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Structured logging
    enable_structured_logging: bool = True
    
    # Log sampling (for high-volume applications)
    enable_log_sampling: bool = False
    sample_rate: float = 0.1  # 10% of logs
    
    # Custom log fields
    custom_fields: Dict[str, str] = field(default_factory=lambda: {
        "service": "sentiment-analysis",
        "version": "2.0.0",
        "environment": "production"
    })
    
    def get_log_config(self) -> Dict:
        """Get logging configuration dictionary."""
        config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": self.log_format
                }
            },
            "handlers": {
                "default": {
                    "level": self.default_log_level,
                    "formatter": "standard",
                    "class": "logging.StreamHandler"
                }
            },
            "loggers": {
                "": {
                    "handlers": ["default"],
                    "level": self.default_log_level,
                    "propagate": False
                }
            }
        }
        
        if self.enable_structured_logging:
            config["formatters"]["structured"] = {
                "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
                "class": "pythonjsonlogger.jsonlogger.JsonFormatter"
            }
            config["handlers"]["default"]["formatter"] = "structured"
        
        return config
