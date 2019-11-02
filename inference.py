import os
import glob
import pathlib

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

def load_model(model_name):
    base_url = 'http://download.tensorflow.org/models/object_detection/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(
    fname=model_name, 
    origin=base_url + model_file,
    untar=True)

    model_dir = pathlib.Path(model_dir)/"saved_model"

    model = tf.saved_model.load(str(model_dir))
    model = model.signatures['serving_default']

    return model

def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis,...]

    # Run inference
    output_dict = model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                  output_dict['detection_masks'], output_dict['detection_boxes'],
                   image.shape[0], image.shape[1])      
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict

PATH_TO_LABELS = 'data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
model = load_model(model_name)

# [<tf.Tensor 'image_tensor:0' shape=(None, None, None, 3) dtype=uint8>]
# {'detection_scores': tf.float32,
#  'detection_classes': tf.float32,
#  'num_detections': tf.float32,
#  'detection_boxes': tf.float32}

# {'detection_scores': TensorShape([None, 100]),
#  'detection_classes': TensorShape([None, 100]),
#  'num_detections': TensorShape([None]),
#  'detection_boxes': TensorShape([None, 100, 4])}

failed_list = []
dataset_path = "./models/research/object_detection/ExDark"
test_path = glob.glob(dataset_path+"/*/*.jpg")
test_path.extend(glob.glob(dataset_path+"/*/*.png"))

output_hub = []
for idx, img_path in enumerate(test_path):
    try:
        output_hub.append({path: img_path,
                           resutl: run_inference_for_single_image(model, img_path)})
    except:
        failed_list.append(img_path)
np.save("inference_result.npy", output_hub)
