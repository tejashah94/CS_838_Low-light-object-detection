#!/usr/bin/python
# -*- coding: UTF-8 -*-
import argparse

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


def load_model(model_name):
    base_url = 'http://download.tensorflow.org/models/object_detection/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(fname=model_name, origin=base_url + model_file, untar=True)
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
    output_dict = {key:value[0, :num_detections].numpy() for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections
    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    # Handle models with masks:
    if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
              image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    return output_dict


def main():
    parser = argparse.ArgumentParser(
        description='''This is a beta script for inferencing images on tf_model_zoo models ''',
        epilog="""All's well that ends well.""")
    parser.add_argument('--name_model', metavar='', type=str, default="ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03",
                        help='The name of model to inference images.')
    parser.add_argument('--name_dataset', metavar='', type=str, default="ExDark",
                        help='The name of dataset to be inferenced.')
    parser.add_argument('--idx_start', metavar='', type=int, default=0,
                        help='Index of the first image.') 
    parser.add_argument('--idx_end', metavar='', type=int, default=6000,
                        help='Index of the last image.')     

    args = parser.parse_args()

    name_model = args.name_model
    name_dataset = args.name_dataset
    idx_start = args.idx_start
    idx_end = args.idx_end

    # patch tf1 into `utils.ops`
    utils_ops.tf = tf.compat.v1

    # Patch the location of gfile
    tf.gfile = tf.io.gfile

    PATH_TO_LABELS = './object_detection/data/mscoco_label_map.pbtxt'
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
    # model_name = 'faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28'
    model = load_model(name_model)

    failed_list = []
    dataset_path = "./object_detection/"+name_dataset
    test_path = glob.glob(dataset_path+"/*/*.jpg")
    test_path.extend(glob.glob(dataset_path+"/*/*.png"))

    n_total = len(test_path)
    output_hub = []

    for idx, img_path in enumerate(test_path[idx_start:idx_end]):
        try:
            img_np = np.array(Image.open(img_path))
            result = run_inference_for_single_image(model, img_np)
            output_hub.append({"path": img_path, "result": result})
            print(idx, "out of ", n_total, ":", img_path)
        except:
            failed_list.append(img_path)

    save_name = name_model+"@"+name_dataset+"@"+str(idx_start)+"@"+str(idx_end)
    np.save(save_name+".npy", output_hub)

if __name__ == "__main__":
    main()