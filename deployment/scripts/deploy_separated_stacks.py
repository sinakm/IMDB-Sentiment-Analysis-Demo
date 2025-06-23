"""
Deployment script for separated API and Frontend stacks.

This script deploys the sentiment analysis infrastructure in two separate stacks:
1. API Stack (Lambda, API Gateway, Monitoring)
2. Frontend Stack (React app, S3, CloudFront)

The frontend stack depends on the API stack outputs.
"""

import subprocess
import sys
import time
import json
from pathlib import Path
from typing import Dict, Optional


class StackDeployer:
    """Handles deployment of CDK stacks with proper dependency management."""
    
    def __init__(self, environment: str = "prod", region: str = "us-east-1"):
        self.environment = environment
        self.region = region
        self.cdk_dir = Path(__file__).parent.parent / "cdk"
        
    def run_command(self, command: list, cwd: Optional[Path] = None) -> Dict:
        """Run a command and return the result."""
        if cwd is None:
            cwd = self.cdk_dir
            
        print(f"Running: {' '.join(command)}")
        print(f"Working directory: {cwd}")
        
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                capture_output=True,
                text=True,
                check=True
            )
            return {
                "success": True,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except subprocess.CalledProcessError as e:
            return {
                "success": False,
                "stdout": e.stdout,
                "stderr": e.stderr,
                "returncode": e.returncode,
                "error": str(e)
            }
    
    def check_cdk_bootstrap(self) -> bool:
        """Check if CDK is bootstrapped in the target region."""
        print(f"Checking CDK bootstrap status for region {self.region}...")
        
        result = self.run_command([
            "aws", "cloudformation", "describe-stacks",
            "--stack-name", "CDKToolkit",
            "--region", self.region
        ])
        
        if result["success"]:
            print("✅ CDK is bootstrapped")
            return True
        else:
            print("❌ CDK is not bootstrapped")
            return False
    
    def bootstrap_cdk(self) -> bool:
        """Bootstrap CDK in the target region."""
        print(f"Bootstrapping CDK in region {self.region}...")
        
        result = self.run_command([
            "cdk", "bootstrap",
            f"aws://unknown-account/{self.region}",
            "--app", "python app_with_frontend.py"
        ])
        
        if result["success"]:
            print("✅ CDK bootstrap completed")
            return True
        else:
            print(f"❌ CDK bootstrap failed: {result['stderr']}")
            return False
    
    def deploy_api_stack(self) -> bool:
        """Deploy the API stack first."""
        stack_name = f"SentimentAnalysisStack-{self.environment}"
        print(f"\n🚀 Deploying API stack: {stack_name}")
        
        result = self.run_command([
            "cdk", "deploy", stack_name,
            "--app", "python app_with_frontend.py",
            "--require-approval", "never",
            "--context", f"environment={self.environment}",
            "--context", f"region={self.region}"
        ])
        
        if result["success"]:
            print("✅ API stack deployed successfully")
            print("Stack outputs:")
            print(result["stdout"])
            return True
        else:
            print(f"❌ API stack deployment failed:")
            print(f"Error: {result['stderr']}")
            return False
    
    def deploy_frontend_stack(self) -> bool:
        """Deploy the frontend stack after API stack."""
        stack_name = f"SentimentAnalysisFrontendStack-{self.environment}"
        print(f"\n🚀 Deploying Frontend stack: {stack_name}")
        
        result = self.run_command([
            "cdk", "deploy", stack_name,
            "--app", "python app_with_frontend.py",
            "--require-approval", "never",
            "--context", f"environment={self.environment}",
            "--context", f"region={self.region}"
        ])
        
        if result["success"]:
            print("✅ Frontend stack deployed successfully")
            print("Stack outputs:")
            print(result["stdout"])
            return True
        else:
            print(f"❌ Frontend stack deployment failed:")
            print(f"Error: {result['stderr']}")
            return False
    
    def deploy_both_stacks(self) -> bool:
        """Deploy both stacks in the correct order."""
        print(f"\n🎯 Starting deployment for environment: {self.environment}")
        print(f"📍 Target region: {self.region}")
        
        # Check CDK bootstrap
        if not self.check_cdk_bootstrap():
            if not self.bootstrap_cdk():
                return False
        
        # Deploy API stack first
        if not self.deploy_api_stack():
            print("\n❌ Deployment failed at API stack")
            return False
        
        # Wait a moment for stack outputs to be available
        print("\n⏳ Waiting for API stack outputs to be available...")
        time.sleep(10)
        
        # Deploy frontend stack
        if not self.deploy_frontend_stack():
            print("\n❌ Deployment failed at Frontend stack")
            return False
        
        print("\n🎉 All stacks deployed successfully!")
        return True
    
    def get_stack_outputs(self, stack_name: str) -> Dict:
        """Get outputs from a deployed stack."""
        result = self.run_command([
            "aws", "cloudformation", "describe-stacks",
            "--stack-name", stack_name,
            "--region", self.region,
            "--query", "Stacks[0].Outputs",
            "--output", "json"
        ])
        
        if result["success"]:
            try:
                return json.loads(result["stdout"])
            except json.JSONDecodeError:
                return {}
        return {}
    
    def print_deployment_summary(self):
        """Print a summary of the deployed resources."""
        api_stack_name = f"SentimentAnalysisStack-{self.environment}"
        frontend_stack_name = f"SentimentAnalysisFrontendStack-{self.environment}"
        
        print("\n" + "="*60)
        print("🎯 DEPLOYMENT SUMMARY")
        print("="*60)
        
        # Get API stack outputs
        api_outputs = self.get_stack_outputs(api_stack_name)
        if api_outputs:
            print("\n📡 API Stack Resources:")
            for output in api_outputs:
                key = output.get("OutputKey", "")
                value = output.get("OutputValue", "")
                description = output.get("Description", "")
                print(f"  • {key}: {value}")
                if description:
                    print(f"    {description}")
        
        # Get frontend stack outputs
        frontend_outputs = self.get_stack_outputs(frontend_stack_name)
        if frontend_outputs:
            print("\n🌐 Frontend Stack Resources:")
            for output in frontend_outputs:
                key = output.get("OutputKey", "")
                value = output.get("OutputValue", "")
                description = output.get("Description", "")
                print(f"  • {key}: {value}")
                if description:
                    print(f"    {description}")
        
        print("\n" + "="*60)


def main():
    """Main deployment function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy Sentiment Analysis stacks")
    parser.add_argument(
        "--environment", "-e",
        default="prod",
        choices=["dev", "staging", "prod"],
        help="Deployment environment"
    )
    parser.add_argument(
        "--region", "-r",
        default="us-east-1",
        help="AWS region"
    )
    parser.add_argument(
        "--api-only",
        action="store_true",
        help="Deploy only the API stack"
    )
    parser.add_argument(
        "--frontend-only",
        action="store_true",
        help="Deploy only the frontend stack"
    )
    
    args = parser.parse_args()
    
    deployer = StackDeployer(
        environment=args.environment,
        region=args.region
    )
    
    success = False
    
    if args.api_only:
        success = deployer.deploy_api_stack()
    elif args.frontend_only:
        success = deployer.deploy_frontend_stack()
    else:
        success = deployer.deploy_both_stacks()
    
    if success:
        deployer.print_deployment_summary()
        print("\n✅ Deployment completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Deployment failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
