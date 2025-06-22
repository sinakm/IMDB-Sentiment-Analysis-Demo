"""
Utility functions for CDK deployment.

This package contains common utilities and helper functions used across constructs.
"""

from .common import get_resource_name, get_tags, validate_config

__all__ = [
    "get_resource_name",
    "get_tags", 
    "validate_config"
]
