#!/bin/bash

# Game Item OCR Lambda Deployment Script
set -e

# Configuration
AWS_REGION="us-east-1"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPOSITORY="worlforged-items-ocr-ecr"
IMAGE_TAG="latest"
STACK_NAME="worlforged-items-ocr-stack"

echo "Starting deployment process..."
echo "Region: $AWS_REGION"
echo "Account: $AWS_ACCOUNT_ID"

# Step 1: Create ECR repository if it doesn't exist
echo "Setting up ECR repository..."
aws ecr describe-repositories --repository-names $ECR_REPOSITORY --region $AWS_REGION >/dev/null 2>&1 || {
    echo "Creating ECR repository..."
    aws ecr create-repository --repository-name $ECR_REPOSITORY --region $AWS_REGION
}

# Step 2: Get ECR login token
echo "Logging into ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Step 3: Build Docker image
echo "Building Docker image..."
docker build -t $ECR_REPOSITORY:$IMAGE_TAG .

# Step 4: Tag and push to ECR
IMAGE_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:$IMAGE_TAG"
echo "Pushing to ECR: $IMAGE_URI"
docker tag $ECR_REPOSITORY:$IMAGE_TAG $IMAGE_URI
docker push $IMAGE_URI

# Step 5: Deploy CloudFormation stack
# echo "Deploying CloudFormation stack..."
# sam deploy \
#     --template-file template.yaml \
#     --stack-name $STACK_NAME \
#     --parameter-overrides \
#         ImageUri=$IMAGE_URI \
#         Stage=prod \
#     --capabilities CAPABILITY_IAM \
#     --region $AWS_REGION

# Step 6: Get outputs
echo "Deployment complete!"
# echo "📋 Stack outputs:"
# aws cloudformation describe-stacks \
#     --stack-name $STACK_NAME \
#     --region $AWS_REGION \
#     --query 'Stacks[0].Outputs[*].[OutputKey,OutputValue]' \
#     --output table
