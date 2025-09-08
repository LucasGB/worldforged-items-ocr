# !/usr/bin/env python3

import base64
import json
import requests
import argparse
from pathlib import Path

def encode_image(image_path: str) -> str:
    """Encode image file to base64"""
    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def test_api_gateway(api_url: str, image_path: str):
    """Test the API Gateway endpoint"""
    print(f"🧪 Testing API Gateway endpoint: {api_url}")
    print(f"📷 Image: {image_path}")
    
    # Encode image
    image_data = encode_image(image_path)
    
    # Prepare request
    payload = {
        "image_data": image_data
    }
    
    headers = {
        'Content-Type': 'application/json'
    }
    
    try:
        print("📤 Sending request...")
        response = requests.post(api_url, json=payload, headers=headers, timeout=30)  # 5 minute timeout
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            print(json.dumps(result, indent=2))
        else:
            print("❌ Error!")
            print(f"Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("⏱️ Request timed out (this might be expected for cold starts)")
    except Exception as e:
        print(f"❌ Request failed: {e}")

def test_lambda_direct(function_name: str, image_path: str, region: str = 'us-east-1'):
    """Test Lambda function directly"""
    import boto3
    
    print(f"🧪 Testing Lambda function directly: {function_name}")
    print(f"📷 Image: {image_path}")
    
    # Encode image
    image_data = encode_image(image_path)
    
    # Prepare payload
    payload = {
        "image_data": image_data
    }
    
    # Invoke Lambda
    lambda_client = boto3.client('lambda', region_name=region)
    
    try:
        print("📤 Invoking Lambda...")
        response = lambda_client.invoke(
            FunctionName=function_name,
            Payload=json.dumps(payload)
        )
        
        result = json.loads(response['Payload'].read())
        
        if result.get('success'):
            print("✅ Success!")
            print(json.dumps(result, indent=2))
        else:
            print("❌ Error!")
            print(json.dumps(result, indent=2))
            
    except Exception as e:
        print(f"❌ Lambda invocation failed: {e}")

def main():
    parser = argparse.ArgumentParser(description='Test Game Item OCR Lambda/API')
    parser.add_argument('image_path', help='Path to test image')
    parser.add_argument('--api-url', help='API Gateway endpoint URL')
    parser.add_argument('--lambda-name', help='Lambda function name for direct testing')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    
    args = parser.parse_args()
    
    if not Path(args.image_path).exists():
        print(f"❌ Image file not found: {args.image_path}")
        return

    api_URL = "https://myjj6lt159.execute-api.us-east-1.amazonaws.com/v1/process-image"
    
    if api_URL:
        test_api_gateway(api_URL, args.image_path)
    elif args.lambda_name:
        test_lambda_direct(args.lambda_name, args.image_path, args.region)
    else:
        print("❌ Please provide either --api-url or --lambda-name")

if __name__ == "__main__":
    main()