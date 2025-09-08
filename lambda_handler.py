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

# Configure logging
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger()

def init_logger(log_local=False):
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    if log_local:
        logging.basicConfig(level=logging.INFO, format=log_format, datefmt=date_format) # logger.info no terminal
    else:
        logger.setLevel(logging.INFO) # logger.info em AWS logs
    return logger

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
    
    1. API Gateway v2.0 (Lambda Function URL or HTTP API):
    {
        "version": "2.0",
        "body": "base64_encoded_image_data",
        "isBase64Encoded": true,
        "headers": {"content-type": "image/png"}
    }
    
    2. API Gateway v1.0 (REST API):
    {
        "httpMethod": "POST",
        "body": "base64_encoded_image_data",
        "isBase64Encoded": true,
        "headers": {"content-type": "image/png"}
    }
    
    3. Direct invocation with image data:
    {
        "image_data": "base64_encoded_image",
        "image_format": "png"
    }
    
    4. S3 reference:
    {
        "s3_bucket": "my-bucket",
        "s3_key": "path/to/image.png"
    }
    """
    
    try:
        init_logger(False)
        logger.info(event)
        logger.info(f"Processing event version: {event.get('version', 'unknown')}")
        logger.info(f"Request context: {event.get('requestContext', {}).get('http', {}).get('method', 'unknown')}")
        
        # Parse the event based on source
        image_path = None
        is_api_gateway = False
        
        if event.get('version') == '2.0' and 'requestContext' in event:
            # API Gateway v2.0 (HTTP API or Lambda Function URL)
            is_api_gateway = True
            image_path = handle_api_gateway_v2_request(event)
        elif 'httpMethod' in event:
            # API Gateway v1.0 (REST API)
            is_api_gateway = True
            image_path = handle_api_gateway_v1_request(event)
        elif 's3_bucket' in event and 's3_key' in event:
            # S3 reference
            is_api_gateway = False
            image_path = handle_s3_request(event)
        elif 'image_data' in event:
            # Direct base64 image data
            is_api_gateway = False
            image_path = handle_direct_image_request(event)
        else:
            raise ValueError("Invalid event format. Expected API Gateway request, image_data, or S3 reference")
        
        # Process the image
        ocr_processor = get_processor()
        result = ocr_processor.process_item_image(image_path)
        
        # Clean up temporary file
        if image_path and os.path.exists(image_path):
            os.unlink(image_path)
        
        # Format response based on source
        if is_api_gateway:
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
                    'body': result,
                })
            }
        else:
            return {
                'success': True,
                'body': result,
            }
            
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        
        error_response = {
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }
        
        if is_api_gateway:
            return {
                'statusCode': 500,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'POST, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type'
                },
                'body': json.dumps(error_response)
            }
        else:
            return error_response

def handle_api_gateway_v2_request(event: Dict[str, Any]) -> str:
    """Handle API Gateway v2.0 request (HTTP API or Lambda Function URL)"""
    
    # Handle CORS preflight
    if event.get('requestContext', {}).get('http', {}).get('method') == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type'
            }
        }
    
    method = event.get('requestContext', {}).get('http', {}).get('method')
    if method != 'POST':
        raise ValueError(f"Only POST method is supported, got: {method}")
    
    body = event.get('body', '')
    is_base64 = event.get('isBase64Encoded', False)
    content_type = event.get('headers', {}).get('content-type', '').lower()
    
    if not body:
        raise ValueError("Request body is empty")
    
    logger.info(f"Content-Type: {content_type}, isBase64Encoded: {is_base64}")
    
    # Handle different content types
    if 'image/' in content_type:
        # Direct image upload
        if is_base64:
            image_data = base64.b64decode(body)
        else:
            # Should not happen with binary image data, but handle it
            image_data = body.encode('utf-8')
        
        # Determine file extension from content-type
        if 'png' in content_type:
            ext = '.png'
        elif 'jpeg' in content_type or 'jpg' in content_type:
            ext = '.jpg'
        elif 'gif' in content_type:
            ext = '.gif'
        elif 'bmp' in content_type:
            ext = '.bmp'
        else:
            ext = '.png'  # default
    
    elif 'application/json' in content_type:
        # JSON payload with base64 image data
        if is_base64:
            body = base64.b64decode(body).decode('utf-8')
        
        try:
            body_json = json.loads(body)
            if 'image_data' not in body_json:
                raise ValueError("JSON body must contain 'image_data' field")
            
            image_data = base64.b64decode(body_json['image_data'])
            ext = f".{body_json.get('image_format', 'png')}"
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in request body: {e}")
    
    else:
        # Assume it's base64 encoded image data
        try:
            if is_base64:
                image_data = base64.b64decode(body)
            else:
                image_data = base64.b64decode(body)
            ext = '.png'  # default
        except Exception as e:
            raise ValueError(f"Failed to decode image data: {e}")
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
        tmp_file.write(image_data)
        logger.info(f"Saved image to temporary file: {tmp_file.name} ({len(image_data)} bytes)")
        return tmp_file.name

def handle_api_gateway_v1_request(event: Dict[str, Any]) -> str:
    """Handle API Gateway v1.0 request (REST API)"""
    
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
    content_type = event.get('headers', {}).get('content-type', '').lower()
    
    if not body:
        raise ValueError("Request body is empty")
    
    logger.info(f"Content-Type: {content_type}, isBase64Encoded: {is_base64}")
    
    # Handle different content types
    if 'image/' in content_type:
        # Direct image upload
        if is_base64:
            image_data = base64.b64decode(body)
        else:
            image_data = body.encode('utf-8')
        
        # Determine file extension from content-type
        if 'png' in content_type:
            ext = '.png'
        elif 'jpeg' in content_type or 'jpg' in content_type:
            ext = '.jpg'
        elif 'gif' in content_type:
            ext = '.gif'
        elif 'bmp' in content_type:
            ext = '.bmp'
        else:
            ext = '.png'  # default
    
    elif 'application/json' in content_type:
        # JSON payload with base64 image data
        if is_base64:
            body = base64.b64decode(body).decode('utf-8')
        
        try:
            body_json = json.loads(body)
            if 'image_data' not in body_json:
                raise ValueError("JSON body must contain 'image_data' field")
            
            image_data = base64.b64decode(body_json['image_data'])
            ext = f".{body_json.get('image_format', 'png')}"
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in request body: {e}")
    
    else:
        # Assume it's base64 encoded image data
        try:
            if is_base64:
                image_data = base64.b64decode(body)
            else:
                image_data = base64.b64decode(body)
            ext = '.png'  # default
        except Exception as e:
            raise ValueError(f"Failed to decode image data: {e}")
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
        tmp_file.write(image_data)
        logger.info(f"Saved image to temporary file: {tmp_file.name} ({len(image_data)} bytes)")
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