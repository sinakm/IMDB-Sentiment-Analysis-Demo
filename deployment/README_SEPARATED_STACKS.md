# Separated Stack Deployment Guide

This guide explains the separated stack architecture for the Sentiment Analysis application with runtime configuration.

## 🌟 [**Live Demo**](https://d354nm7rm3934r.cloudfront.net/)

See the deployed application in action!

## Architecture Overview

The application is now deployed using **two separate CDK stacks**:

### 1. API Stack (`SentimentAnalysisStack`)

- **Lambda Function** with sentiment analysis models
- **API Gateway** with endpoints and API key authentication
- **CloudWatch Monitoring** and alarms
- **SNS Notifications** (optional)

### 2. Frontend Stack (`SentimentAnalysisFrontendStack`)

- **React Application** with runtime configuration
- **S3 Bucket** for static hosting
- **CloudFront Distribution** for global CDN
- **Runtime Configuration**: Dynamic API key injection via custom resource

## Benefits of Separated Stacks

✅ **Runtime Configuration**: API keys injected at deployment time, not build time  
✅ **Independent Deployments**: Update frontend without touching API infrastructure  
✅ **Cleaner Dependencies**: Frontend imports API details via CloudFormation exports  
✅ **Better Error Isolation**: Issues in one stack don't affect the other  
✅ **Faster Iterations**: Frontend changes deploy quickly  
✅ **CORS Support**: Proper CORS headers for all API responses (200, 400, 500)

## Deployment Options

### Option 1: Deploy Both Stacks (Recommended)

```bash
cd buildops/deployment/cdk
python ../scripts/deploy_separated_stacks.py --environment prod
```

### Option 2: Deploy API Stack Only

```bash
cd buildops/deployment/cdk
python ../scripts/deploy_separated_stacks.py --api-only --environment prod
```

### Option 3: Deploy Frontend Stack Only (after API is deployed)

```bash
cd buildops/deployment/cdk
python ../scripts/deploy_separated_stacks.py --frontend-only --environment prod
```

### Option 4: Manual CDK Deployment

```bash
cd buildops/deployment/cdk

# Deploy API stack first
cdk deploy SentimentAnalysisStack-prod --app "python app_with_frontend.py"

# Deploy frontend stack second
cdk deploy SentimentAnalysisFrontendStack-prod --app "python app_with_frontend.py"
```

## Environment Configuration

The deployment script supports multiple environments:

- **dev**: Development environment with relaxed security
- **staging**: Staging environment for testing
- **prod**: Production environment with full security

```bash
# Deploy to development
python ../scripts/deploy_separated_stacks.py --environment dev

# Deploy to staging
python ../scripts/deploy_separated_stacks.py --environment staging

# Deploy to production (default)
python ../scripts/deploy_separated_stacks.py --environment prod
```

## How It Works

### 1. API Stack Deployment

1. Creates Lambda function with sentiment models
2. Sets up API Gateway with endpoints
3. Configures monitoring and alarms
4. **Exports** API URL and API Key ID for frontend use

### 2. Frontend Stack Deployment

1. **Imports** API URL and API Key ID from API stack
2. Builds React app without API configuration
3. Deploys to S3 and CloudFront
4. **Creates runtime configuration** via custom resource Lambda

### 3. Runtime Configuration

The frontend gets API details injected at deployment time via `env.js`:

```javascript
window.__ENV__ = {
  API_URL: "https://api-gateway-url.com/prod",
  API_KEY_ID: "actual-api-key-value", // Retrieved dynamically
  ENVIRONMENT: "prod",
};
```

The custom resource Lambda:

- Retrieves the actual API key value (not just ID)
- Creates `env.js` file with runtime configuration
- Uploads to S3 with no-cache headers

## Key Changes Made

### 1. Runtime Configuration Approach

- Custom resource Lambda retrieves actual API key value
- Frontend uses `window.__ENV__` instead of `process.env`
- API key value injected at deployment time, not build time
- Updated to Node 20 (from Node 18)

### 2. CORS Fixes

- Added CORS headers to all API Gateway method responses (200, 400, 500)
- Fixed integration response headers
- Proper error handling with CORS support

### 2. Stack Separation

- API stack exports required values
- Frontend stack imports via CloudFormation
- Proper dependency management

### 3. Improved Deployment Process

- Automated deployment script
- Better error handling and logging
- Support for partial deployments

## Troubleshooting

### If API Stack Deployment Fails

1. Check AWS credentials and permissions
2. Verify CDK is bootstrapped: `cdk bootstrap`
3. Check CloudWatch logs for Lambda function issues

### If Frontend Stack Deployment Fails

1. Ensure API stack is deployed first
2. Verify API stack exports are available
3. Check that frontend source directory exists
4. Verify Node.js dependencies in package.json

### Common Issues

**Issue**: Frontend can't find API URL  
**Solution**: Ensure API stack is deployed and exports are available

**Issue**: Build fails during Docker bundling  
**Solution**: Check frontend package.json and ensure all dependencies are listed

**Issue**: CloudFront distribution creation fails  
**Solution**: Check AWS service limits and region availability

## Stack Outputs

### API Stack Outputs

- `ApiUrl`: The API Gateway URL
- `ApiKeyId`: The API key identifier
- `PredictEndpoint`: Direct prediction endpoint URL
- `HealthEndpoint`: Health check endpoint URL
- `LambdaFunctionName`: Lambda function name

### Frontend Stack Outputs

- `FrontendUrl`: The website URL (CloudFront or S3)
- `FrontendBucket`: S3 bucket name
- `CloudFrontDistributionId`: CloudFront distribution ID

## Testing the Deployment

After successful deployment:

1. **Test API Health**:

   ```bash
   curl https://your-api-url/prod/health
   ```

2. **Test API Prediction**:

   ```bash
   # Get API key first
   aws apigateway get-api-key --api-key YOUR_API_KEY_ID --include-value --query 'value' --output text

   # Test prediction
   curl -X POST https://your-api-url/prod/predict \
     -H "Content-Type: application/json" \
     -H "x-api-key: YOUR_API_KEY" \
     -d '{"text": "This movie was great!", "model": "both"}'
   ```

3. **Test Frontend**:
   Open the frontend URL in your browser and test the sentiment analysis interface.

## Migration from Old Architecture

If you have the old combined stack deployed:

1. **Destroy the old stack** (optional, but recommended):

   ```bash
   cdk destroy SentimentAnalysisStack --app "python app.py"
   ```

2. **Deploy the new separated stacks**:
   ```bash
   python ../scripts/deploy_separated_stacks.py --environment prod
   ```

## Next Steps

1. **Deploy the API stack first** to resolve your current deployment issue
2. **Test the API endpoints** to ensure they work correctly
3. **Deploy the frontend stack** to get the complete application
4. **Update your CI/CD pipelines** to use the new deployment approach

This new architecture eliminates the custom resource issues you were experiencing and provides a much more maintainable and scalable deployment process.
