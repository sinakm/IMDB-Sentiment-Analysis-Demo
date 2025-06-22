#!/bin/bash
echo "Copying model artifacts to Lambda directory..."

rm -rf deployment/lambda/models
mkdir -p deployment/lambda/models
cp -r artifacts/* deployment/lambda/models/

echo "SUCCESS: Models copied successfully!"
echo "Next step: cd deployment/cdk && cdk deploy"
