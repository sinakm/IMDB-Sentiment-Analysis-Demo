"""
Common utility functions for CDK deployment.
"""

from typing import Dict, Any, Optional
import re


def get_resource_name(base_name: str, resource_type: str, environment: str = "prod") -> str:
    """
    Generate a consistent resource name following naming conventions.
    
    Args:
        base_name: Base name for the resource
        resource_type: Type of AWS resource (lambda, api, role, etc.)
        environment: Environment name (dev, staging, prod)
    
    Returns:
        Formatted resource name
    """
    # Sanitize base name
    clean_base = re.sub(r'[^a-zA-Z0-9-]', '-', base_name.lower())
    clean_base = re.sub(r'-+', '-', clean_base).strip('-')
    
    # Sanitize resource type
    clean_type = re.sub(r'[^a-zA-Z0-9-]', '-', resource_type.lower())
    
    # Sanitize environment
    clean_env = re.sub(r'[^a-zA-Z0-9-]', '-', environment.lower())
    
    return f"{clean_base}-{clean_type}-{clean_env}"


def sanitize_tag_value(value: str) -> str:
    """
    Sanitize tag value to meet AWS requirements.
    Valid characters are [a-zA-Z+-=._:/]
    """
    # Replace invalid characters with hyphens
    sanitized = re.sub(r'[^a-zA-Z0-9+\-=._:/]', '-', str(value))
    # Remove multiple consecutive hyphens
    sanitized = re.sub(r'-+', '-', sanitized)
    # Trim to max 256 characters
    return sanitized[:256].strip('-')


def get_tags(
    project_name: str = "sentiment-analysis",
    environment: str = "prod",
    owner: str = "ml-team",
    cost_center: Optional[str] = None,
    additional_tags: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """
    Generate standard tags for AWS resources.
    
    Args:
        project_name: Name of the project
        environment: Environment (dev, staging, prod)
        owner: Team or person responsible
        cost_center: Cost center for billing
        additional_tags: Additional custom tags
    
    Returns:
        Dictionary of tags with sanitized values
    """
    tags = {
        "Project": sanitize_tag_value(project_name),
        "Environment": sanitize_tag_value(environment),
        "Owner": sanitize_tag_value(owner),
        "ManagedBy": "CDK",
        "Service": "sentiment-analysis-api"
    }
    
    if cost_center:
        tags["CostCenter"] = sanitize_tag_value(cost_center)
    
    if additional_tags:
        # Sanitize additional tags
        for key, value in additional_tags.items():
            # Sanitize both key and value
            clean_key = re.sub(r'[^a-zA-Z0-9+\-=._:/]', '-', str(key))[:128]
            clean_value = sanitize_tag_value(value)
            tags[clean_key] = clean_value
    
    return tags


def validate_config(config: Any) -> bool:
    """
    Validate configuration object by calling its validate method if available.
    
    Args:
        config: Configuration object to validate
    
    Returns:
        True if validation passes
    
    Raises:
        ValueError: If validation fails
    """
    if hasattr(config, 'validate') and callable(getattr(config, 'validate')):
        config.validate()
        return True
    
    return True


def sanitize_name(name: str, max_length: int = 64) -> str:
    """
    Sanitize a name for AWS resource naming requirements.
    
    Args:
        name: Original name
        max_length: Maximum allowed length
    
    Returns:
        Sanitized name
    """
    # Convert to lowercase and replace invalid characters
    sanitized = re.sub(r'[^a-zA-Z0-9-]', '-', name.lower())
    
    # Remove multiple consecutive hyphens
    sanitized = re.sub(r'-+', '-', sanitized)
    
    # Remove leading/trailing hyphens
    sanitized = sanitized.strip('-')
    
    # Truncate if too long
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length].rstrip('-')
    
    # Ensure it doesn't start with a number (for some AWS resources)
    if sanitized and sanitized[0].isdigit():
        sanitized = f"r-{sanitized}"
    
    return sanitized


def get_environment_config(environment: str) -> Dict[str, Any]:
    """
    Get environment-specific configuration.
    
    Args:
        environment: Environment name
    
    Returns:
        Environment configuration dictionary
    """
    configs = {
        "dev": {
            "memory_size": 1024,
            "timeout_minutes": 3,
            "log_retention_days": 3,
            "throttle_rate_limit": 10,
            "quota_limit": 1000,
            "enable_alarms": False,
            "enable_dashboard": False
        },
        "staging": {
            "memory_size": 1536,
            "timeout_minutes": 5,
            "log_retention_days": 7,
            "throttle_rate_limit": 50,
            "quota_limit": 5000,
            "enable_alarms": True,
            "enable_dashboard": True
        },
        "prod": {
            "memory_size": 2048,
            "timeout_minutes": 5,
            "log_retention_days": 7,
            "throttle_rate_limit": 100,
            "quota_limit": 10000,
            "enable_alarms": True,
            "enable_dashboard": True
        }
    }
    
    return configs.get(environment, configs["prod"])


def format_arn(
    service: str,
    resource: str,
    region: str = "us-east-1",
    account_id: str = "*",
    resource_type: Optional[str] = None
) -> str:
    """
    Format an AWS ARN.
    
    Args:
        service: AWS service name
        resource: Resource name
        region: AWS region
        account_id: AWS account ID
        resource_type: Resource type (optional)
    
    Returns:
        Formatted ARN
    """
    if resource_type:
        return f"arn:aws:{service}:{region}:{account_id}:{resource_type}/{resource}"
    else:
        return f"arn:aws:{service}:{region}:{account_id}:{resource}"


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.
    Later configs override earlier ones.
    
    Args:
        *configs: Configuration dictionaries to merge
    
    Returns:
        Merged configuration dictionary
    """
    result = {}
    
    for config in configs:
        if config:
            result.update(config)
    
    return result


def validate_aws_name(name: str, resource_type: str) -> bool:
    """
    Validate AWS resource name according to service-specific rules.
    
    Args:
        name: Resource name to validate
        resource_type: Type of AWS resource
    
    Returns:
        True if name is valid
    
    Raises:
        ValueError: If name is invalid
    """
    if not name:
        raise ValueError(f"{resource_type} name cannot be empty")
    
    # Common rules for most AWS resources
    if resource_type in ["lambda", "api-gateway", "log-group"]:
        if len(name) > 64:
            raise ValueError(f"{resource_type} name cannot exceed 64 characters")
        
        if not re.match(r'^[a-zA-Z0-9-_]+$', name):
            raise ValueError(f"{resource_type} name can only contain alphanumeric characters, hyphens, and underscores")
    
    # Lambda-specific rules
    if resource_type == "lambda":
        if name.startswith('-') or name.endswith('-'):
            raise ValueError("Lambda function name cannot start or end with hyphen")
    
    # API Gateway-specific rules
    if resource_type == "api-gateway":
        if len(name) > 1024:
            raise ValueError("API Gateway name cannot exceed 1024 characters")
    
    return True


def get_stack_outputs(stack_name: str) -> Dict[str, str]:
    """
    Get outputs from a deployed CDK stack.
    
    Args:
        stack_name: Name of the CDK stack
    
    Returns:
        Dictionary of stack outputs
    
    Note:
        This is a placeholder for actual implementation that would
        use boto3 to query CloudFormation stack outputs.
    """
    # This would be implemented with boto3 in a real scenario
    # For now, return empty dict as placeholder
    return {}
