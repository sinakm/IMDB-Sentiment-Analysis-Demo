"""
Validate deployment setup before CDK deploy.

This script checks that all required files are in place and the Lambda function
can be built and tested locally.
"""

import sys
import os
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple


def check_file_exists(file_path: Path, description: str) -> bool:
    """Check if a file exists and report the result."""
    if file_path.exists():
        size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"  ✅ {description}: {file_path} ({size_mb:.1f} MB)")
        return True
    else:
        print(f"  ❌ {description}: {file_path} (missing)")
        return False


def validate_model_artifacts() -> bool:
    """Validate that model artifacts are properly exported."""
    print("🔍 Validating model artifacts...")
    
    project_root = Path(__file__).parent.parent.parent
    artifacts_dir = project_root / "artifacts"
    
    if not artifacts_dir.exists():
        print("  ❌ Artifacts directory not found. Run export_models.py first.")
        return False
    
    success = True
    
    # Check LSTM model files
    lstm_dir = artifacts_dir / "lstm"
    success &= check_file_exists(lstm_dir / "model.pt", "LSTM model")
    success &= check_file_exists(lstm_dir / "config.json", "LSTM config")
    success &= check_file_exists(lstm_dir / "vocab.json", "LSTM vocabulary")
    
    # Check Verbalizer model files
    verbalizer_dir = artifacts_dir / "verbalizer"
    success &= check_file_exists(verbalizer_dir / "model.pt", "Verbalizer model")
    success &= check_file_exists(verbalizer_dir / "config.json", "Verbalizer config")
    
    # Check deployment manifest
    success &= check_file_exists(artifacts_dir / "deployment_manifest.json", "Deployment manifest")
    
    return success


def validate_lambda_models() -> bool:
    """Validate that models are copied to Lambda directory."""
    print("🔍 Validating Lambda models...")
    
    project_root = Path(__file__).parent.parent.parent
    lambda_models_dir = project_root / "deployment" / "lambda" / "models"
    
    if not lambda_models_dir.exists():
        print("  ❌ Lambda models directory not found. Run copy_models script first.")
        return False
    
    success = True
    
    # Check LSTM model files
    lstm_dir = lambda_models_dir / "lstm"
    success &= check_file_exists(lstm_dir / "model.pt", "Lambda LSTM model")
    success &= check_file_exists(lstm_dir / "config.json", "Lambda LSTM config")
    success &= check_file_exists(lstm_dir / "vocab.json", "Lambda LSTM vocabulary")
    
    # Check Verbalizer model files
    verbalizer_dir = lambda_models_dir / "verbalizer"
    success &= check_file_exists(verbalizer_dir / "model.pt", "Lambda Verbalizer model")
    success &= check_file_exists(verbalizer_dir / "config.json", "Lambda Verbalizer config")
    
    return success


def validate_lambda_code() -> bool:
    """Validate Lambda function code."""
    print("🔍 Validating Lambda code...")
    
    project_root = Path(__file__).parent.parent.parent
    lambda_dir = project_root / "deployment" / "lambda"
    
    success = True
    
    # Check required files
    success &= check_file_exists(lambda_dir / "lambda_function.py", "Lambda function")
    success &= check_file_exists(lambda_dir / "inference_engine.py", "Inference engine")
    success &= check_file_exists(lambda_dir / "requirements.txt", "Requirements file")
    success &= check_file_exists(lambda_dir / "Dockerfile", "Dockerfile")
    
    return success


def validate_cdk_setup() -> bool:
    """Validate CDK setup."""
    print("🔍 Validating CDK setup...")
    
    project_root = Path(__file__).parent.parent.parent
    cdk_dir = project_root / "deployment" / "cdk"
    
    success = True
    
    # Check CDK files
    success &= check_file_exists(cdk_dir / "app.py", "CDK app")
    success &= check_file_exists(cdk_dir / "sentiment_stack.py", "CDK stack")
    success &= check_file_exists(cdk_dir / "requirements.txt", "CDK requirements")
    
    # Check constructs
    constructs_dir = cdk_dir / "constructs"
    success &= check_file_exists(constructs_dir / "__init__.py", "Constructs init")
    success &= check_file_exists(constructs_dir / "lambda_construct.py", "Lambda construct")
    success &= check_file_exists(constructs_dir / "api_gateway_construct.py", "API Gateway construct")
    success &= check_file_exists(constructs_dir / "monitoring_construct.py", "Monitoring construct")
    
    # Check config
    config_dir = cdk_dir / "config"
    success &= check_file_exists(config_dir / "__init__.py", "Config init")
    success &= check_file_exists(config_dir / "lambda_config.py", "Lambda config")
    success &= check_file_exists(config_dir / "api_config.py", "API config")
    success &= check_file_exists(config_dir / "monitoring_config.py", "Monitoring config")
    
    return success


def check_aws_cli() -> bool:
    """Check if AWS CLI is installed and configured."""
    print("🔍 Checking AWS CLI...")
    
    try:
        # Check if AWS CLI is installed
        result = subprocess.run(["aws", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✅ AWS CLI installed: {result.stdout.strip()}")
        else:
            print("  ❌ AWS CLI not found")
            return False
        
        # Check if AWS credentials are configured
        result = subprocess.run(["aws", "sts", "get-caller-identity"], capture_output=True, text=True)
        if result.returncode == 0:
            identity = json.loads(result.stdout)
            print(f"  ✅ AWS credentials configured for account: {identity.get('Account', 'unknown')}")
            return True
        else:
            print("  ❌ AWS credentials not configured")
            return False
            
    except FileNotFoundError:
        print("  ❌ AWS CLI not found")
        return False


def check_docker() -> bool:
    """Check if Docker is installed and running."""
    print("🔍 Checking Docker...")
    
    try:
        # Check if Docker is installed
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✅ Docker installed: {result.stdout.strip()}")
        else:
            print("  ❌ Docker not found")
            return False
        
        # Check if Docker daemon is running
        result = subprocess.run(["docker", "info"], capture_output=True, text=True)
        if result.returncode == 0:
            print("  ✅ Docker daemon is running")
            return True
        else:
            print("  ❌ Docker daemon not running")
            return False
            
    except FileNotFoundError:
        print("  ❌ Docker not found")
        return False


def check_cdk_cli() -> bool:
    """Check if CDK CLI is installed."""
    print("🔍 Checking CDK CLI...")
    
    try:
        result = subprocess.run(["cdk", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✅ CDK CLI installed: {result.stdout.strip()}")
            return True
        else:
            print("  ❌ CDK CLI not found")
            return False
            
    except FileNotFoundError:
        print("  ❌ CDK CLI not found. Install with: npm install -g aws-cdk")
        return False


def test_lambda_locally() -> bool:
    """Test Lambda function locally if possible."""
    print("🔍 Testing Lambda function locally...")
    
    project_root = Path(__file__).parent.parent.parent
    lambda_dir = project_root / "deployment" / "lambda"
    
    # Change to lambda directory
    original_cwd = os.getcwd()
    
    try:
        os.chdir(lambda_dir)
        
        # Try to import and test the lambda function
        sys.path.insert(0, str(lambda_dir))
        
        try:
            from lambda_function import lambda_handler
            
            # Test health check
            health_event = {
                'httpMethod': 'GET',
                'path': '/health'
            }
            
            class MockContext:
                aws_request_id = 'test-request-123'
            
            result = lambda_handler(health_event, MockContext())
            
            if result['statusCode'] == 200:
                print("  ✅ Health check test passed")
                
                # Test prediction (if models are available)
                predict_event = {
                    'body': json.dumps({
                        'text': 'This is a test message.',
                        'model': 'both'
                    })
                }
                
                result = lambda_handler(predict_event, MockContext())
                
                if result['statusCode'] == 200:
                    print("  ✅ Prediction test passed")
                    return True
                else:
                    print(f"  ⚠️ Prediction test failed with status {result['statusCode']}")
                    return False
            else:
                print(f"  ❌ Health check failed with status {result['statusCode']}")
                return False
                
        except Exception as e:
            print(f"  ⚠️ Local test failed: {e}")
            return False
            
    finally:
        os.chdir(original_cwd)
        if str(lambda_dir) in sys.path:
            sys.path.remove(str(lambda_dir))


def generate_deployment_summary() -> Dict:
    """Generate a summary of the deployment readiness."""
    project_root = Path(__file__).parent.parent.parent
    
    summary = {
        "timestamp": "2025-06-22T13:32:00Z",
        "project_root": str(project_root),
        "models_ready": False,
        "lambda_ready": False,
        "cdk_ready": False,
        "prerequisites_ready": False,
        "next_steps": []
    }
    
    # Check each component
    if validate_model_artifacts() and validate_lambda_models():
        summary["models_ready"] = True
    else:
        summary["next_steps"].append("Run export_models.py and copy_models script")
    
    if validate_lambda_code():
        summary["lambda_ready"] = True
    else:
        summary["next_steps"].append("Fix Lambda function code issues")
    
    if validate_cdk_setup():
        summary["cdk_ready"] = True
    else:
        summary["next_steps"].append("Fix CDK setup issues")
    
    if check_aws_cli() and check_docker() and check_cdk_cli():
        summary["prerequisites_ready"] = True
    else:
        summary["next_steps"].append("Install and configure prerequisites (AWS CLI, Docker, CDK)")
    
    if not summary["next_steps"]:
        summary["next_steps"].append("Ready to deploy! Run: cd deployment/cdk && cdk deploy")
    
    return summary


def main():
    """Main validation pipeline."""
    print("=" * 60)
    print("🔍 DEPLOYMENT VALIDATION")
    print("=" * 60)
    print("Checking deployment readiness...")
    print()
    
    # Run all validations
    validations = [
        ("Model Artifacts", validate_model_artifacts),
        ("Lambda Models", validate_lambda_models),
        ("Lambda Code", validate_lambda_code),
        ("CDK Setup", validate_cdk_setup),
        ("AWS CLI", check_aws_cli),
        ("Docker", check_docker),
        ("CDK CLI", check_cdk_cli)
    ]
    
    results = {}
    
    for name, validation_func in validations:
        print(f"\n{name}:")
        results[name] = validation_func()
        print()
    
    # Test Lambda locally
    print("Local Testing:")
    local_test_result = test_lambda_locally()
    print()
    
    # Generate summary
    summary = generate_deployment_summary()
    
    print("=" * 60)
    print("📋 VALIDATION SUMMARY")
    print("=" * 60)
    
    all_passed = all(results.values())
    
    if all_passed and local_test_result:
        print("🎉 All validations passed! Ready to deploy.")
        print("\nNext steps:")
        print("1. cd deployment/cdk")
        print("2. cdk bootstrap (first time only)")
        print("3. cdk deploy")
    else:
        print("❌ Some validations failed. Please fix the issues above.")
        print("\nNext steps:")
        for step in summary["next_steps"]:
            print(f"- {step}")
    
    print(f"\nValidation results:")
    for name, result in results.items():
        status = "✅" if result else "❌"
        print(f"  {status} {name}")
    
    local_status = "✅" if local_test_result else "⚠️"
    print(f"  {local_status} Local Testing")
    
    return all_passed and local_test_result


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
