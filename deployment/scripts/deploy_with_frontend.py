#!/usr/bin/env python3
"""
Deployment script for the complete Sentiment Analysis stack with frontend.

This script deploys:
1. Lambda function with LSTM and Verbalizer models
2. API Gateway with API key authentication
3. React frontend to S3 + CloudFront
4. Monitoring and alarms

Usage:
    python deploy_with_frontend.py [--environment prod|dev|staging] [--email your@email.com]
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def run_command(command: str, cwd: str = None) -> bool:
    """Run a shell command and return success status."""
    print(f"Running: {command}")
    if cwd:
        print(f"In directory: {cwd}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            check=True,
            capture_output=False
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        return False


def check_prerequisites():
    """Check if all prerequisites are installed."""
    print("🔍 Checking prerequisites...")
    
    # Check AWS CLI
    if not run_command("aws --version"):
        print("❌ AWS CLI not found. Please install AWS CLI.")
        return False
    
    # Check CDK
    if not run_command("cdk --version"):
        print("❌ AWS CDK not found. Please install AWS CDK.")
        return False
    
    # Check Node.js
    if not run_command("node --version"):
        print("❌ Node.js not found. Please install Node.js.")
        return False
    
    # Check npm
    if not run_command("npm --version"):
        print("❌ npm not found. Please install npm.")
        return False
    
    # Check Python
    if not run_command("python --version"):
        print("❌ Python not found. Please install Python.")
        return False
    
    print("✅ All prerequisites found!")
    return True


def install_dependencies(project_root: Path):
    """Install Python and Node.js dependencies."""
    print("📦 Installing dependencies...")
    
    # Install CDK dependencies
    cdk_dir = project_root / "deployment" / "cdk"
    print(f"Installing CDK dependencies in {cdk_dir}")
    if not run_command("pip install -r requirements.txt", str(cdk_dir)):
        return False
    
    # Install frontend dependencies
    frontend_dir = project_root / "deployment" / "frontend"
    print(f"Installing frontend dependencies in {frontend_dir}")
    if not run_command("npm install", str(frontend_dir)):
        return False
    
    print("✅ Dependencies installed!")
    return True


def bootstrap_cdk():
    """Bootstrap CDK if needed."""
    print("🚀 Bootstrapping CDK...")
    
    # Check if already bootstrapped
    result = subprocess.run(
        "aws cloudformation describe-stacks --stack-name CDKToolkit",
        shell=True,
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✅ CDK already bootstrapped!")
        return True
    
    # Bootstrap CDK
    if not run_command("cdk bootstrap"):
        print("❌ CDK bootstrap failed!")
        return False
    
    print("✅ CDK bootstrapped!")
    return True


def deploy_stack(environment: str, email: str = None, project_root: Path = None):
    """Deploy the complete stack."""
    print(f"🚀 Deploying stack for environment: {environment}")
    
    cdk_dir = project_root / "deployment" / "cdk"
    
    # Prepare deployment command
    deploy_cmd = f"cdk deploy --require-approval never"
    
    # Add context variables
    deploy_cmd += f" -c environment={environment}"
    deploy_cmd += f" -c deploy_frontend=true"
    
    if email:
        deploy_cmd += f" -c alarm_email={email}"
    
    # Run deployment
    if not run_command(deploy_cmd, str(cdk_dir)):
        print("❌ Deployment failed!")
        return False
    
    print("✅ Deployment completed!")
    return True


def get_stack_outputs(environment: str, project_root: Path):
    """Get and display stack outputs."""
    print("📋 Getting stack outputs...")
    
    cdk_dir = project_root / "deployment" / "cdk"
    stack_name = f"SentimentAnalysisStack-{environment}"
    
    # Get stack outputs
    cmd = f"aws cloudformation describe-stacks --stack-name {stack_name} --query 'Stacks[0].Outputs' --output table"
    
    print(f"Stack outputs for {stack_name}:")
    run_command(cmd, str(cdk_dir))


def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description="Deploy Sentiment Analysis stack with frontend")
    parser.add_argument(
        "--environment", "-e",
        choices=["dev", "staging", "prod"],
        default="prod",
        help="Deployment environment"
    )
    parser.add_argument(
        "--email",
        help="Email address for CloudWatch alarms"
    )
    parser.add_argument(
        "--skip-deps",
        action="store_true",
        help="Skip dependency installation"
    )
    parser.add_argument(
        "--skip-bootstrap",
        action="store_true",
        help="Skip CDK bootstrap"
    )
    
    args = parser.parse_args()
    
    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    print("🎯 Sentiment Analysis Stack Deployment")
    print(f"Environment: {args.environment}")
    print(f"Project root: {project_root}")
    if args.email:
        print(f"Alarm email: {args.email}")
    print("-" * 50)
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Install dependencies
    if not args.skip_deps:
        if not install_dependencies(project_root):
            sys.exit(1)
    
    # Bootstrap CDK
    if not args.skip_bootstrap:
        if not bootstrap_cdk():
            sys.exit(1)
    
    # Deploy stack
    if not deploy_stack(args.environment, args.email, project_root):
        sys.exit(1)
    
    # Get outputs
    get_stack_outputs(args.environment, project_root)
    
    print("\n🎉 Deployment completed successfully!")
    print("\nNext steps:")
    print("1. Get your API key using the command shown in the outputs")
    print("2. Test the API endpoints")
    print("3. Access the frontend URL to use the web interface")
    print("4. Monitor the CloudWatch dashboard for metrics")


if __name__ == "__main__":
    main()
