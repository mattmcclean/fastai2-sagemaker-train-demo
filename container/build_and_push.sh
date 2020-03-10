#!/usr/bin/env bash

# This script shows how to build the Docker image and push it to ECR to be ready for use
# by SageMaker.

# The argument to this script is the image name. This will be used as the image on the local
# machine and combined with the account and region to form the repository name for ECR.

set -i

image=$1

if [ "$image" == "" ]
then
    echo "Usage: $0 <image-name>"
    exit 1
fi

# Get the account number associated with the current IAM credentials
account=$(aws sts get-caller-identity --query Account --output text)

if [ $? -ne 0 ]
then
    exit 255
fi


# Get the region defined in the current configuration (default to us-west-2 if none defined)
region=$(aws configure get region)
region=${region:-us-west-2}

fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}"
echo "Creating Docker image name is ${fullname}"

for repo in ${image}-training ${image}-inference
do
    # If the repository doesn't exist in ECR, create it.
    aws ecr describe-repositories --repository-names "${repo}" > /dev/null 2>&1

    if [ $? -ne 0 ]
    then
        echo "Creating ECR repo with name ${repo}"
        aws ecr create-repository --repository-name "${repo}" > /dev/null
    fi
done


# Get the login command from ECR and execute it directly
$(aws ecr get-login --region ${region} --no-include-email)

# Get the login command from ECR in order to pull down the SageMaker PyTorch image
$(aws ecr get-login --registry-ids 763104351884 --region ${region} --no-include-email)

# Build the docker image locally with the image name and then push it to ECR
# with the full name.
docker build -t ${image}-inference . --build-arg REGION=${region} --build-arg ARCH=cpu --build-arg TYPE=inference --build-arg CUDA_VERSION=""
docker build -t ${image}-training . --build-arg REGION=${region} --build-arg ARCH=gpu --build-arg TYPE=training --build-arg CUDA_VERSION="-cu101"

docker tag ${image}-inference ${fullname}-inference
docker tag ${image}-training ${fullname}-training

docker push ${fullname}-inference
docker push ${fullname}-training
