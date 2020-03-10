# Copyright 2017-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

# For more information on creating a Dockerfile
# https://docs.docker.com/compose/gettingstarted/#step-2-create-a-dockerfile
# https://github.com/awslabs/amazon-sagemaker-examples/master/advanced_functionality/pytorch_extending_our_containers/pytorch_extending_our_containers.ipynb
ARG REGION=eu-west-1
ARG ARCH=gpu
ARG TYPE=training
ARG CUDA_VERSION=

# SageMaker PyTorch image
FROM 763104351884.dkr.ecr.$REGION.amazonaws.com/pytorch-$TYPE:1.3.1-$ARCH-py36$CUDA_VERSION-ubuntu16.04

COPY requirements.txt /opt/app/

RUN pip install --no-cache-dir -r /opt/app/requirements.txt