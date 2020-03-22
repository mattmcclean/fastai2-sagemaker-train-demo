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

import subprocess
#import sys
#subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'fastai2', 'ipykernel'])

import ast
import argparse
import logging
import json
import io
import os

import shlex

from fastai2.vision.all import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

JSON_CONTENT_TYPE = 'application/json'
EXPORT_MODEL_NAME = 'model.pkl'

def _train(args):
    # get the images path
    path_img = Path(args.data_dir)/'images'
    fnames = get_image_files(path_img)
    # create the image data loader
    dls = ImageDataLoaders.from_path_re(path_img, fnames, pat=r'(.+)_\d+.jpg$', 
                                        item_tfms=RandomResizedCrop(460, min_scale=0.75), bs=args.batch_size,
                                        batch_tfms=[*aug_transforms(size=args.image_size, max_warp=0),
                                        Normalize.from_stats(*imagenet_stats)])
    # create the learner
    learn = cnn_learner(dls, resnet50, metrics=error_rate).to_fp16()
    
    # train the model
    learn.fit_one_cycle(args.epochs)
    logger.info("Finished training")
    
    # save the model
    logger.info("Saving the model.")
    path_model = Path(args.model_dir)
    learn.export(path_model/EXPORT_MODEL_NAME)
    logger.info("Model saved")

# loads the model into memory from disk and returns it
def model_fn(model_dir):
    logger.info('model_fn')
    path_model = Path(model_dir)
    logger.debug(f'Loading model from path: {str(path_model/EXPORT_MODEL_NAME)}')
    defaults.device = torch.device('cpu')
    learn = load_learner(path_model/EXPORT_MODEL_NAME, cpu=True)
    logger.info('model loaded successfully')
    return learn

# Deserialize the Invoke request body into an object we can perform prediction on
def input_fn(request_body, content_type=JSON_CONTENT_TYPE):
    logger.info('Deserializing the input data.')
    # process a URL submitted to the endpoint
    if content_type == JSON_CONTENT_TYPE:
        logger.debug(f'Request body is: {request_body}')
        request = json.loads(request_body)
        logger.debug(f'Loaded JSON object: {request}')
        url = request['url']
        logger.info(f'Downloading image from URL: {url}')
        img_content = requests.get(url).content
        logger.info(f'Returning image bytes')
        return io.BytesIO(img_content).read()
    raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))

# Perform prediction on the deserialized object, with the loaded model
def predict_fn(input_object, model):
    logger.info("Calling model")
    start_time = time.time()
    predict_class,predict_idx,predict_values = model.predict(input_object)
    logger.info("--- Inference time: %s seconds ---" % (time.time() - start_time))
    predict_class=os.path.basename(predict_class)
    logger.info(f'Predicted class is {str(predict_class)}')
    logger.info(f'Predict confidence score is {predict_values[predict_idx.item()].item()}')
    return { "class": str(predict_class),
        "confidence": predict_values[predict_idx.item()].item() }

# Serialize the prediction result into the desired response content type
def output_fn(prediction, accept=JSON_CONTENT_TYPE):        
    logger.info('Serializing the generated output.')
    if accept == JSON_CONTENT_TYPE: 
        logger.debug(f'Returning response {json.dumps(prediction)}')
        return json.dumps(prediction), accept
    raise Exception('Requested unsupported ContentType in Accept: {}'.format(accept))  
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--workers', type=int, default=2, metavar='W',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', type=int, default=8, metavar='E',
                        help='number of total epochs to run (default: 8)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='BS',
                        help='batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='initial learning rate (default: 0.001)')
    parser.add_argument('--image-size', type=int, default=299, metavar='IS',
                        help='image size (default: 299)')    
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--dist-backend', type=str, default='gloo', help='distributed backend (default: gloo)')

    # The parameters below retrieve their default values from SageMaker environment variables, which are
    # instantiated by the SageMaker containers framework.
    # https://github.com/aws/sagemaker-containers#how-a-script-is-executed-inside-the-container
    parser.add_argument('--hosts', type=str, default=ast.literal_eval(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    _train(parser.parse_args())