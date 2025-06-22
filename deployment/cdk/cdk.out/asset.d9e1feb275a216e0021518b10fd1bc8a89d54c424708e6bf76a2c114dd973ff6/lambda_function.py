"""
AWS Lambda function for sentiment analysis inference.

This function provides a REST API endpoint that runs inference on both
LSTM and Verbalizer models and returns comprehensive results.
"""

import json
import logging
import traceback
from typing import Dict, Any
from inference_engine import SentimentInferenceEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global inference engine (initialized once per container)
inference_engine = None

def initialize_models():
    """Initialize the inference engine (called once per container)."""
    global inference_engine
    
    if inference_engine is None:
        logger.info("🚀 Initializing sentiment analysis models...")
        try:
            inference_engine = SentimentInferenceEngine()
            logger.info("✅ Models initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize models: {e}")
            logger.error(traceback.format_exc())
            raise e
    
    return inference_engine

def validate_request(event: Dict[str, Any]) -> tuple[str, str, str]:
    """Validate the incoming request and extract text and model preference."""
    
    # Handle different event sources (API Gateway, direct invoke, etc.)
    if 'body' in event:
        # API Gateway request
        try:
            if isinstance(event['body'], str):
                body = json.loads(event['body'])
            else:
                body = event['body']
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in request body: {e}")
    else:
        # Direct Lambda invoke
        body = event
    
    # Extract text
    if 'text' not in body:
        raise ValueError("Missing 'text' field in request")
    
    text = body['text']
    if not isinstance(text, str):
        raise ValueError("'text' field must be a string")
    
    if len(text.strip()) == 0:
        raise ValueError("'text' field cannot be empty")
    
    if len(text) > 5000:  # Reasonable limit
        raise ValueError("'text' field too long (max 5000 characters)")
    
    # Extract model preference (optional)
    model = body.get('model', 'both')  # Default to both models
    if model not in ['lstm', 'verbalizer', 'both']:
        raise ValueError("'model' field must be 'lstm', 'verbalizer', or 'both'")
    
    return text.strip(), model, body

def create_response(status_code: int, body: Dict[str, Any], headers: Dict[str, str] = None) -> Dict[str, Any]:
    """Create a properly formatted API Gateway response."""
    
    default_headers = {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',  # Enable CORS
        'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
        'Access-Control-Allow-Methods': 'POST,GET,OPTIONS'
    }
    
    if headers:
        default_headers.update(headers)
    
    return {
        'statusCode': status_code,
        'headers': default_headers,
        'body': json.dumps(body, ensure_ascii=False)
    }

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Main Lambda handler function.
    
    Expected input:
    {
        "text": "This movie was absolutely fantastic!",
        "model": "both"  // Optional: "lstm", "verbalizer", or "both"
    }
    
    Returns:
    {
        "text": "This movie was absolutely fantastic!",
        "predictions": {
            "lstm": {"prediction": "positive", "confidence": 0.78, "processing_time_ms": 45},
            "verbalizer": {"prediction": "positive", "confidence": 0.87, "processing_time_ms": 25}
        },
        "consensus": {
            "prediction": "positive",
            "agreement": true,
            "avg_confidence": 0.825,
            "models_count": 2
        },
        "total_processing_time_ms": 70,
        "models_available": 2,
        "models": ["lstm", "verbalizer"]
    }
    """
    
    # Log the incoming event (without sensitive data)
    logger.info(f"Received event: {json.dumps({k: v for k, v in event.items() if k != 'body'})}")
    
    try:
        # Handle OPTIONS request for CORS
        if event.get('httpMethod') == 'OPTIONS':
            return create_response(200, {'message': 'CORS preflight'})
        
        # Handle health check
        if event.get('path') == '/health' or event.get('httpMethod') == 'GET':
            return health_check(event, context)
        
        # Validate request
        text, model_preference, request_body = validate_request(event)
        logger.info(f"Processing text: {text[:100]}{'...' if len(text) > 100 else ''}")
        logger.info(f"Model preference: {model_preference}")
        
        # Initialize models if needed
        engine = initialize_models()
        
        # Run inference based on model preference
        if model_preference == 'both':
            results = engine.predict_all(text)
        else:
            # Single model prediction
            single_result = engine.predict_single(text, model_preference)
            results = {
                "text": text,
                "predictions": {model_preference: single_result},
                "consensus": {
                    "prediction": single_result.get("prediction", "error"),
                    "agreement": True,
                    "avg_confidence": single_result.get("confidence", 0.0),
                    "models_count": 1 if single_result.get("prediction") != "error" else 0
                },
                "total_processing_time_ms": single_result.get("processing_time_ms", 0),
                "models_available": 1 if single_result.get("prediction") != "error" else 0,
                "models": [model_preference] if single_result.get("prediction") != "error" else []
            }
        
        # Add metadata
        results['lambda_request_id'] = context.aws_request_id if context else 'local'
        results['version'] = '2.0.0'  # Updated for two-model approach
        results['model_preference'] = model_preference
        
        logger.info(f"Inference completed in {results['total_processing_time_ms']}ms")
        logger.info(f"Consensus: {results['consensus']['prediction']} ({results['consensus']['avg_confidence']:.3f})")
        
        return create_response(200, results)
        
    except ValueError as e:
        # Client error (bad request)
        logger.warning(f"Client error: {e}")
        return create_response(400, {
            'error': 'Bad Request',
            'message': str(e),
            'request_id': context.aws_request_id if context else 'local'
        })
        
    except Exception as e:
        # Server error
        logger.error(f"Server error: {e}")
        logger.error(traceback.format_exc())
        return create_response(500, {
            'error': 'Internal Server Error',
            'message': 'An unexpected error occurred during inference',
            'request_id': context.aws_request_id if context else 'local'
        })

def health_check(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Health check endpoint."""
    try:
        engine = initialize_models()
        
        # Get model info
        model_info = engine.get_model_info()
        
        # Quick test inference
        test_result = engine.predict_all("This is a test.")
        
        return create_response(200, {
            'status': 'healthy',
            'models_available': model_info['total_models'],
            'available_models': model_info['available_models'],
            'model_types': model_info['model_types'],
            'version': '2.0.0',
            'test_processing_time_ms': test_result.get('total_processing_time_ms', 0),
            'timestamp': test_result.get('total_processing_time_ms', 0)
        })
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return create_response(503, {
            'status': 'unhealthy',
            'error': str(e),
            'version': '2.0.0'
        })

# For local testing
if __name__ == "__main__":
    # Test the function locally
    test_events = [
        # Test both models
        {
            'body': json.dumps({
                'text': 'This movie was absolutely fantastic! Great acting and amazing plot.',
                'model': 'both'
            })
        },
        # Test single model
        {
            'body': json.dumps({
                'text': 'This movie was terrible and boring.',
                'model': 'verbalizer'
            })
        },
        # Test health check
        {
            'httpMethod': 'GET',
            'path': '/health'
        }
    ]
    
    class MockContext:
        aws_request_id = 'test-request-123'
    
    for i, test_event in enumerate(test_events):
        print(f"\n{'='*50}")
        print(f"Test {i+1}: {test_event.get('body', 'Health Check')}")
        print(f"{'='*50}")
        
        result = lambda_handler(test_event, MockContext())
        print(f"Status: {result['statusCode']}")
        print(json.dumps(json.loads(result['body']), indent=2))
