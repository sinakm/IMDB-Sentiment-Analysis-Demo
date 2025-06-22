"""
Monitoring construct for sentiment analysis deployment.
"""

from constructs import Construct
from aws_cdk import (
    aws_cloudwatch as cloudwatch,
    aws_logs as logs,
    aws_sns as sns,
    aws_lambda as _lambda,
    aws_apigateway as apigateway,
    Duration
)
from typing import Dict, List, Optional

from config.monitoring_config import MonitoringConfig, AlarmConfig, DashboardConfig
from utils.common import get_resource_name, get_tags, validate_config


class SentimentMonitoringConstruct(Construct):
    """CDK construct for creating monitoring and alerting for sentiment analysis."""
    
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        lambda_function: _lambda.Function,
        api_gateway: apigateway.RestApi,
        config: MonitoringConfig,
        alarm_config: AlarmConfig = None,
        dashboard_config: DashboardConfig = None,
        environment: str = "prod",
        **kwargs
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)
        
        # Validate configuration
        validate_config(config)
        
        self.config = config
        self.alarm_config = alarm_config or AlarmConfig()
        self.dashboard_config = dashboard_config or DashboardConfig()
        self.environment = environment
        self.lambda_function = lambda_function
        self.api_gateway = api_gateway
        
        # Create resources
        self._create_sns_topic()
        self._create_alarms()
        if self.config.enable_dashboard:
            self._create_dashboard()
    
    def _create_sns_topic(self) -> None:
        """Create SNS topic for alarm notifications."""
        if self.config.alarm_email:
            topic_name = get_resource_name(
                base_name="sentiment-analysis",
                resource_type="alerts",
                environment=self.environment
            )
            
            self.sns_topic = sns.Topic(
                self,
                "AlertsTopic",
                topic_name=topic_name,
                display_name="Sentiment Analysis Alerts"
            )
            
            # Add email subscription
            self.sns_topic.add_subscription(
                sns.EmailSubscription(self.config.alarm_email)
            )
        else:
            self.sns_topic = None
    
    def _create_alarms(self) -> None:
        """Create CloudWatch alarms."""
        if not self.config.enable_alarms:
            return
        
        self.alarms = {}
        alarm_configs = self.alarm_config.get_alarm_configs()
        
        for alarm_config in alarm_configs:
            alarm_name = get_resource_name(
                base_name="sentiment-analysis",
                resource_type=f"alarm-{alarm_config['name'].lower()}",
                environment=self.environment
            )
            
            # Create metric based on the alarm type
            if alarm_config["metric_name"] in ["Errors", "Duration", "Throttles", "Invocations"]:
                metric = getattr(self.lambda_function, f"metric_{alarm_config['metric_name'].lower()}")()
            else:
                # Custom metric
                metric = cloudwatch.Metric(
                    namespace=self.config.metrics_namespace,
                    metric_name=alarm_config["metric_name"],
                    statistic=alarm_config["statistic"]
                )
            
            # Create alarm
            alarm = cloudwatch.Alarm(
                self,
                f"Alarm{alarm_config['name']}",
                alarm_name=alarm_name,
                alarm_description=alarm_config["description"],
                metric=metric,
                threshold=alarm_config["threshold"],
                comparison_operator=alarm_config["comparison_operator"],
                evaluation_periods=alarm_config["evaluation_periods"],
                datapoints_to_alarm=alarm_config["datapoints_to_alarm"],
                treat_missing_data=alarm_config["treat_missing_data"]
            )
            
            # Add SNS action if topic exists
            if self.sns_topic:
                alarm.add_alarm_action(
                    cloudwatch.SnsAction(self.sns_topic)
                )
            
            self.alarms[alarm_config["name"]] = alarm
        
        # API Gateway specific alarms
        self._create_api_gateway_alarms()
    
    def _create_api_gateway_alarms(self) -> None:
        """Create API Gateway specific alarms."""
        # 4XX Error Rate Alarm
        api_4xx_alarm = cloudwatch.Alarm(
            self,
            "Api4XXErrorAlarm",
            alarm_name=get_resource_name(
                base_name="sentiment-analysis",
                resource_type="alarm-api-4xx",
                environment=self.environment
            ),
            alarm_description="High 4XX error rate in API Gateway",
            metric=self.api_gateway.metric_client_error(),
            threshold=10,  # 10 4XX errors in 5 minutes
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
            evaluation_periods=2,
            datapoints_to_alarm=2,
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING
        )
        
        # 5XX Error Rate Alarm
        api_5xx_alarm = cloudwatch.Alarm(
            self,
            "Api5XXErrorAlarm",
            alarm_name=get_resource_name(
                base_name="sentiment-analysis",
                resource_type="alarm-api-5xx",
                environment=self.environment
            ),
            alarm_description="High 5XX error rate in API Gateway",
            metric=self.api_gateway.metric_server_error(),
            threshold=5,  # 5 5XX errors in 5 minutes
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
            evaluation_periods=1,
            datapoints_to_alarm=1,
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING
        )
        
        # High Latency Alarm
        api_latency_alarm = cloudwatch.Alarm(
            self,
            "ApiLatencyAlarm",
            alarm_name=get_resource_name(
                base_name="sentiment-analysis",
                resource_type="alarm-api-latency",
                environment=self.environment
            ),
            alarm_description="High latency in API Gateway",
            metric=self.api_gateway.metric_latency(),
            threshold=10000,  # 10 seconds
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
            evaluation_periods=3,
            datapoints_to_alarm=2,
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING
        )
        
        # Add SNS actions if topic exists
        if self.sns_topic:
            for alarm in [api_4xx_alarm, api_5xx_alarm, api_latency_alarm]:
                alarm.add_alarm_action(cloudwatch.SnsAction(self.sns_topic))
        
        # Store API alarms
        self.alarms.update({
            "Api4XXError": api_4xx_alarm,
            "Api5XXError": api_5xx_alarm,
            "ApiLatency": api_latency_alarm
        })
    
    def _create_dashboard(self) -> None:
        """Create CloudWatch dashboard."""
        dashboard_name = get_resource_name(
            base_name="sentiment-analysis",
            resource_type="dashboard",
            environment=self.environment
        )
        
        self.dashboard = cloudwatch.Dashboard(
            self,
            "Dashboard",
            dashboard_name=dashboard_name
        )
        
        # Lambda metrics widget
        lambda_widget = cloudwatch.GraphWidget(
            title="Lambda Function Metrics",
            left=[
                self.lambda_function.metric_invocations(),
                self.lambda_function.metric_errors(),
                self.lambda_function.metric_throttles()
            ],
            right=[
                self.lambda_function.metric_duration()
            ],
            width=12,
            height=6
        )
        
        # API Gateway metrics widget
        api_widget = cloudwatch.GraphWidget(
            title="API Gateway Metrics",
            left=[
                self.api_gateway.metric_count(),
                self.api_gateway.metric_client_error(),
                self.api_gateway.metric_server_error()
            ],
            right=[
                self.api_gateway.metric_latency()
            ],
            width=12,
            height=6
        )
        
        # Custom metrics widget (if any)
        custom_metrics = []
        for metric_name in self.dashboard_config.custom_metrics:
            custom_metrics.append(
                cloudwatch.Metric(
                    namespace=self.config.metrics_namespace,
                    metric_name=metric_name
                )
            )
        
        if custom_metrics:
            custom_widget = cloudwatch.GraphWidget(
                title="Custom Application Metrics",
                left=custom_metrics,
                width=12,
                height=6
            )
        else:
            custom_widget = None
        
        # Add widgets to dashboard
        self.dashboard.add_widgets(lambda_widget, api_widget)
        if custom_widget:
            self.dashboard.add_widgets(custom_widget)
        
        # Add alarm status widget
        alarm_widget = cloudwatch.AlarmStatusWidget(
            title="Alarm Status",
            alarms=list(self.alarms.values()),
            width=24,
            height=4
        )
        self.dashboard.add_widgets(alarm_widget)
    
    def add_custom_metric(self, metric_name: str, namespace: str = None) -> cloudwatch.Metric:
        """Add a custom metric."""
        return cloudwatch.Metric(
            namespace=namespace or self.config.metrics_namespace,
            metric_name=metric_name
        )
    
    def create_custom_alarm(
        self,
        alarm_id: str,
        alarm_name: str,
        metric: cloudwatch.Metric,
        threshold: float,
        comparison_operator: cloudwatch.ComparisonOperator,
        evaluation_periods: int = 2,
        datapoints_to_alarm: int = 2
    ) -> cloudwatch.Alarm:
        """Create a custom alarm."""
        alarm = cloudwatch.Alarm(
            self,
            alarm_id,
            alarm_name=alarm_name,
            metric=metric,
            threshold=threshold,
            comparison_operator=comparison_operator,
            evaluation_periods=evaluation_periods,
            datapoints_to_alarm=datapoints_to_alarm
        )
        
        if self.sns_topic:
            alarm.add_alarm_action(cloudwatch.SnsAction(self.sns_topic))
        
        return alarm
    
    @property
    def dashboard_url(self) -> Optional[str]:
        """Get dashboard URL."""
        if hasattr(self, 'dashboard'):
            return f"https://console.aws.amazon.com/cloudwatch/home?region=us-east-1#dashboards:name={self.dashboard.dashboard_name}"
        return None
    
    def get_alarm_arns(self) -> Dict[str, str]:
        """Get ARNs of all alarms."""
        return {name: alarm.alarm_arn for name, alarm in self.alarms.items()}
    
    def get_metrics_summary(self) -> Dict[str, cloudwatch.Metric]:
        """Get summary of all metrics being monitored."""
        return {
            "lambda_invocations": self.lambda_function.metric_invocations(),
            "lambda_errors": self.lambda_function.metric_errors(),
            "lambda_duration": self.lambda_function.metric_duration(),
            "lambda_throttles": self.lambda_function.metric_throttles(),
            "api_count": self.api_gateway.metric_count(),
            "api_latency": self.api_gateway.metric_latency(),
            "api_4xx_errors": self.api_gateway.metric_client_error(),
            "api_5xx_errors": self.api_gateway.metric_server_error()
        }
