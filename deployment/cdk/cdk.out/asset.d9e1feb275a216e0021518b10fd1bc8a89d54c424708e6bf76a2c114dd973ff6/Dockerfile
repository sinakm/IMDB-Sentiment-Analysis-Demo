# Use AWS Lambda Python runtime
FROM public.ecr.aws/lambda/python:3.13

# Set working directory
WORKDIR ${LAMBDA_TASK_ROOT}

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy Lambda function code
COPY lambda_function.py .
COPY inference_engine.py .

# Copy model artifacts (ensure the directory structure is preserved)
COPY models/ ./models/

# Copy deployment manifest if it exists (this will be copied with the models/ directory above)
# COPY models/deployment_manifest.json ./models/ 2>/dev/null || echo "No deployment manifest found"

# Verify model files are in place
RUN ls -la ./models/ && \
    ls -la ./models/lstm/ 2>/dev/null || echo "LSTM models not found" && \
    ls -la ./models/verbalizer/ 2>/dev/null || echo "Verbalizer models not found"

# Set the CMD to your handler
CMD ["lambda_function.lambda_handler"]
