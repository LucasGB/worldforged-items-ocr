import json
import base64
import boto3
import logging
import tempfile
import os
from typing import Dict, Any
from pathlib import Path

# Import your main processor class
from game_item_ocr_processor import GameItemOCRProcessor

import os

os.environ["TMPDIR"] = "/tmp"
os.environ["PADDLEX_CACHE_DIR"] = "/tmp/paddlex_cache"


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global processor instance (reused across invocations)
processor = None
s3_client = boto3.client('s3')

def get_processor():
    """Get or create the OCR processor instance"""
    global processor
    if processor is None:
        logger.info("Initializing OCR processor...")
        processor = GameItemOCRProcessor()
        logger.info("OCR processor initialized")
    return processor

def lambda_handler(event, context):
    """
    Lambda handler for OCR processing
    Expected event formats:
    
    1. Direct image data:
    {
        "image_data": "base64_encoded_image",
        "image_format": "png"  # optional
    }
    
    2. S3 reference:
    {
        "s3_bucket": "my-bucket",
        "s3_key": "path/to/image.png"
    }
    
    3. API Gateway (image in body as base64):
    {
        "body": "base64_encoded_image",
        "isBase64Encoded": true,
        "headers": {"content-type": "image/png"}
    }
    """

    print(event)
    
    try:
        logger.info(f"Processing event: {json.dumps(event, default=str)[:500]}...")
        os.makedirs("/tmp/paddlex_cache", exist_ok=True)
        
        # Parse the event based on source 
        image_path = None
        
        if 'httpMethod' in event:
            # API Gateway request
            image_path = handle_api_gateway_request(event)
        elif 's3_bucket' in event and 's3_key' in event:
            # S3 reference
            image_path = handle_s3_request(event)
        elif 'image_data' in event:
            # Direct base64 image data
            image_path = handle_direct_image_request(event)
        else:
            raise ValueError("Invalid event format. Expected image_data, S3 reference, or API Gateway request")
        
        # Process the image
        ocr_processor = get_processor()
        result = ocr_processor.process_item_image(image_path)
        
        # Clean up temporary file
        if image_path and os.path.exists(image_path):
            os.unlink(image_path)
        
        # Format response based on source
        if 'httpMethod' in event:
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'POST, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type'
                },
                'body': json.dumps({
                    'success': True,
                    'data': result['item_stats'],
                    'metadata': result['processing_metadata']
                })
            }
        else:
            return {
                'success': True,
                'data': result['item_stats'],
                'metadata': result['processing_metadata'],
                'raw_text': result.get('raw_text', [])
            }
            
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        
        error_response = {
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }
        
        if 'httpMethod' in event:
            return {
                'statusCode': 500,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps(error_response)
            }
        else:
            return error_response

def handle_api_gateway_request(event: Dict[str, Any]) -> str:
    """Handle API Gateway request with image in body"""
    
    # Handle CORS preflight
    if event.get('httpMethod') == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type'
            }
        }
    
    if event.get('httpMethod') != 'POST':
        raise ValueError("Only POST method is supported")
    
    body = event.get('body', '')
    is_base64 = event.get('isBase64Encoded', False)
    
    if not body:
        raise ValueError("Request body is empty")
    
    # Decode if base64 encoded
    if is_base64:
        image_data = base64.b64decode(body)
    else:
        # Assume it's JSON with image_data field
        try:
            body_json = json.loads(body)
            image_data = base64.b64decode(body_json['image_data'])
        except:
            raise ValueError("Invalid request format")
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
        tmp_file.write(image_data)
        return tmp_file.name

def handle_s3_request(event: Dict[str, Any]) -> str:
    """Handle S3 reference request"""
    bucket = event['s3_bucket']
    key = event['s3_key']
    
    # Download from S3
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(key).suffix) as tmp_file:
        s3_client.download_fileobj(bucket, key, tmp_file)
        return tmp_file.name

def handle_direct_image_request(event: Dict[str, Any]) -> str:
    """Handle direct base64 image data"""
    image_data = base64.b64decode(event['image_data'])
    image_format = event.get('image_format', 'png')
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{image_format}') as tmp_file:
        tmp_file.write(image_data)
        return tmp_file.name