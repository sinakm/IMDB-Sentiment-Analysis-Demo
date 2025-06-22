# 🚀 AWS Lambda Deployment Guide

Complete guide to deploy your LSTM vs Verbalizer sentiment analysis API to AWS Lambda using CDK.

## 📋 Overview

This deployment creates:

- **AWS Lambda Function** (Docker container with both models)
- **API Gateway** (REST API with authentication)
- **CloudWatch** (Logging and monitoring)

### Models Deployed

- **LSTM**: Traditional approach with learned embeddings (~150MB)
- **Verbalizer**: Modern approach with pre-computed ModernBERT embeddings (~600MB)

## 🔧 Prerequisites

### Required Software

- ✅ **AWS CLI** configured with appropriate permissions
- ✅ **Docker** installed and running
- ✅ **Node.js** (v16+) and **AWS CDK**: `npm install -g aws-cdk`
- ✅ **Python 3.9+** environment

### AWS Permissions Required

Your AWS credentials need permissions for:

- Lambda (create/update functions)
- API Gateway (create/manage APIs)
- IAM (create roles)
- CloudFormation (deploy stacks)
- ECR (push Docker images)

### Verify Prerequisites

```bash
# Check AWS CLI
aws sts get-caller-identity

# Check Docker
docker --version

# Check CDK
cdk --version

# Check Python
python --version
```

## 🎯 Step-by-Step Deployment

### Step 1: Train Your Models

First, ensure you have trained both models:

```bash
cd buildops

# Quick training (for testing)
python compare_all_models.py --quick

# OR full training (recommended for production)
python compare_all_models.py --full
```

**Expected output:**

```
✅ LSTM model trained and saved to checkpoints/lstm/
✅ Verbalizer model trained and saved to checkpoints/verbalizer/
```

### Step 2: Export Models for Deployment

Export the trained models to deployment-ready artifacts:

```bash
python deployment/scripts/export_models.py
```

**This script will:**

- ✅ Find best model checkpoints in `checkpoints/`
- ✅ Copy models to `artifacts/` directory
- ✅ Create configuration files
- ✅ Generate deployment manifest
- ✅ Create copy scripts for next step

**Expected output:**

```
📦 MODEL EXPORT PIPELINE FOR LAMBDA DEPLOYMENT
✅ LSTM model exported
✅ Verbalizer model exported
✅ Deployment manifest created
🎉 MODEL EXPORT COMPLETED SUCCESSFULLY!
```

### Step 3: Copy Models to Lambda Directory

Copy the exported artifacts to the Lambda function directory:

**Windows:**

```cmd
copy_models.bat
```

**macOS/Linux:**

```bash
./copy_models.sh
```

**Manual copy (if scripts fail):**

```bash
# Remove existing models
rm -rf deployment/lambda/models

# Copy artifacts
cp -r artifacts deployment/lambda/models
```

**Verify the copy:**

```bash
ls deployment/lambda/models/
# Should show: lstm/ verbalizer/ deployment_manifest.json
```

### Step 4: Deploy with CDK

Navigate to CDK directory and deploy:

```bash
cd deployment/cdk

# Install CDK dependencies
pip install -r requirements.txt

# Bootstrap CDK (first time only)
cdk bootstrap

# Deploy the stack
cdk deploy
```

**CDK will:**

- 🐳 Build Docker image with your models
- 📤 Push image to AWS ECR
- 🚀 Create Lambda function
- 🌐 Set up API Gateway
- 🔑 Configure API key authentication
- 📊 Set up CloudWatch logging

**Expected output:**

```
✅ SentimentAnalysisStack

Outputs:
SentimentAnalysisStack.ApiEndpoint = https://abc123.execute-api.us-east-1.amazonaws.com/prod
SentimentAnalysisStack.ApiKey = your-api-key-here
```

## 🧪 Testing Your Deployment

### Health Check

```bash
curl https://your-api-url/health
```

**Expected response:**

```json
{
  "status": "healthy",
  "models_available": 2,
  "available_models": ["lstm", "verbalizer"],
  "version": "2.0.0"
}
```

### Test Both Models

```bash
curl -X POST https://your-api-url/predict \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-api-key" \
  -d '{
    "text": "This movie was absolutely fantastic!",
    "model": "both"
  }'
```

**Expected response:**

```json
{
  "text": "This movie was absolutely fantastic!",
  "predictions": {
    "lstm": {
      "prediction": "positive",
      "confidence": 0.78,
      "processing_time_ms": 45
    },
    "verbalizer": {
      "prediction": "positive",
      "confidence": 0.87,
      "processing_time_ms": 25
    }
  },
  "consensus": {
    "prediction": "positive",
    "agreement": true,
    "avg_confidence": 0.825,
    "models_count": 2
  },
  "total_processing_time_ms": 70,
  "models_available": 2
}
```

### Test Single Model

```bash
curl -X POST https://your-api-url/predict \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-api-key" \
  -d '{
    "text": "This movie was terrible and boring.",
    "model": "verbalizer"
  }'
```

## 📊 API Reference

### POST /predict

**Request:**

```json
{
  "text": "Text to analyze (required)",
  "model": "both|lstm|verbalizer (optional, default: both)"
}
```

**Response:**

```json
{
  "text": "Input text",
  "predictions": {
    "lstm": {"prediction": "positive|negative", "confidence": 0.0-1.0, "processing_time_ms": 0},
    "verbalizer": {"prediction": "positive|negative", "confidence": 0.0-1.0, "processing_time_ms": 0}
  },
  "consensus": {
    "prediction": "positive|negative",
    "agreement": true|false,
    "avg_confidence": 0.0-1.0,
    "models_count": 1-2
  },
  "total_processing_time_ms": 0,
  "models_available": 1-2,
  "version": "2.0.0"
}
```

### GET /health

**Response:**

```json
{
  "status": "healthy|unhealthy",
  "models_available": 0-2,
  "available_models": ["lstm", "verbalizer"],
  "version": "2.0.0"
}
```

## 🔧 Configuration & Customization

### Lambda Configuration

Current settings (in CDK stack):

- **Memory**: 2048 MB
- **Timeout**: 300 seconds (5 minutes)
- **Runtime**: Python 3.9
- **Architecture**: x86_64

### Model Selection

You can configure which models to load by modifying the inference engine:

```python
# In deployment/lambda/inference_engine.py
# Comment out models you don't want to deploy
```

### API Key Management

```bash
# Get API key
aws apigateway get-api-keys --query 'items[0].value' --output text

# Create new API key
aws apigateway create-api-key --name "SentimentAnalysis-NewKey"
```

## 📈 Monitoring & Logs

### CloudWatch Logs

```bash
# View logs
aws logs describe-log-groups --log-group-name-prefix "/aws/lambda/SentimentAnalysis"

# Tail logs
aws logs tail /aws/lambda/SentimentAnalysisFunction --follow
```

### Performance Metrics

Monitor in AWS Console:

- **Invocations**: Number of API calls
- **Duration**: Response time per request
- **Errors**: Failed requests
- **Memory Usage**: Peak memory consumption

## 🛠️ Troubleshooting

### Common Issues

**1. Model files not found**

```
❌ Failed to initialize LSTM: [Errno 2] No such file or directory
```

**Solution:** Ensure you ran `copy_models.bat/sh` after exporting models.

**2. Memory limit exceeded**

```
❌ Task timed out after 300.00 seconds
```

**Solution:** Increase Lambda memory in CDK stack (max 10GB).

**3. API Gateway 403 Forbidden**

```
❌ {"message":"Forbidden"}
```

**Solution:** Include `x-api-key` header in your requests.

**4. Docker build fails**

```
❌ Error building Docker image
```

**Solution:** Ensure Docker is running and you have sufficient disk space.

### Debug Mode

Enable debug logging by setting environment variable in CDK:

```python
environment={
    "LOG_LEVEL": "DEBUG"
}
```

### Local Testing

Test Lambda function locally:

```bash
cd deployment/lambda
python lambda_function.py
```

## 🧹 Cleanup

### Remove Deployment

```bash
cd deployment/cdk
cdk destroy
```

**This will remove:**

- Lambda function
- API Gateway
- CloudWatch logs
- IAM roles
- ECR repository

### Clean Local Artifacts

```bash
# Remove exported models
rm -rf artifacts/

# Remove copied models
rm -rf deployment/lambda/models/

# Remove copy scripts
rm copy_models.bat copy_models.sh
```

## 💰 Cost Estimation

### AWS Lambda Costs

- **Requests**: $0.20 per 1M requests
- **Duration**: $0.0000166667 per GB-second
- **Example**: 1000 requests/day ≈ $2-5/month

### API Gateway Costs

- **Requests**: $3.50 per 1M requests
- **Example**: 1000 requests/day ≈ $0.10/month

### Total Estimated Cost

- **Light usage** (100 requests/day): ~$1/month
- **Medium usage** (1000 requests/day): ~$5/month
- **Heavy usage** (10,000 requests/day): ~$50/month

## 🔒 Security Best Practices

1. **API Key Rotation**: Rotate API keys regularly
2. **CORS Configuration**: Restrict origins in production
3. **Rate Limiting**: Implement throttling for production use
4. **VPC**: Deploy Lambda in VPC for enhanced security
5. **Encryption**: Enable encryption at rest and in transit

## 📚 Additional Resources

- [AWS Lambda Documentation](https://docs.aws.amazon.com/lambda/)
- [AWS CDK Documentation](https://docs.aws.amazon.com/cdk/)
- [API Gateway Documentation](https://docs.aws.amazon.com/apigateway/)

---

## 🎉 Success!

Your LSTM vs Verbalizer sentiment analysis API is now live on AWS!

**Key Benefits:**

- ⚡ **Serverless**: No infrastructure management
- 🔄 **Auto-scaling**: Handles traffic spikes automatically
- 💰 **Cost-effective**: Pay only for what you use
- 🌍 **Global**: Available worldwide via CloudFront
- 📊 **Monitored**: Full observability with CloudWatch

**Next Steps:**

- Integrate with your applications
- Monitor performance and costs
- Scale based on usage patterns
- Consider adding more models or features

Happy analyzing! 🚀
