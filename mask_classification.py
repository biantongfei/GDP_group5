#!/usr/bin/env python

""" Mask Classification helper functions.

This file contains all the helper functions required for pre-processing
input data and making predictions for mask classification.

Author: BD Fraser, https://github.com/BenjaminFraser

"""

import numpy as np
import tensorflow as tf


def compute_head_boxes(pose_feats, factor=1.1):
    """ Obtain bounding box of heads using head average
        co-ordinates and torso length as a rough guide. 
    
    Args:
        pose_feats (mxnet array) : 2D array containing x and y co-ords
                columns for the 17 COCO keypoints (17 rows).
        factor (float) : factor to multiply torso length by for the
                        extracted head region (default 1.0)
    Returns:
        head_regions (np.array) : Array with head region box co-ordinates
            in the form [x_mins, y_mins, w, h] for each person (row).
    """
    # calculate the average head x and y coords
    head_x_avgs = pose_feats[:,:5,0].mean(axis=1).asnumpy().reshape(-1, 1)
    head_y_avgs = pose_feats[:,:5,1].mean(axis=1).asnumpy().reshape(-1, 1)
    head_avgs = np.column_stack([head_x_avgs, head_y_avgs])
    
    # calculate left-ear to right-ear abs x dist as approx head width
    head_widths = np.abs(pose_feats[:,4,0].asnumpy() - 
                         pose_feats[:,3,0].asnumpy()).reshape(-1, 1)
    
    # calculate average shoulder co-ordinates
    shoulder_x_avgs = pose_feats[:,5:7,0].mean(axis=1).asnumpy().reshape(-1, 1)
    shoulder_y_avgs = pose_feats[:,5:7,1].mean(axis=1).asnumpy().reshape(-1, 1)
    shoulder_avgs = np.column_stack([shoulder_x_avgs, shoulder_y_avgs])

    # calculate average waist (hip) co-ordinates
    waist_x_avgs = pose_feats[:,11:13,0].mean(axis=1).asnumpy().reshape(-1, 1)
    waist_y_avgs = pose_feats[:,11:13,1].mean(axis=1).asnumpy().reshape(-1, 1)
    waist_avgs = np.column_stack([waist_x_avgs, waist_y_avgs])
    
    # calculate torso length using obtained co-ordinates
    torso_lengths = np.linalg.norm(shoulder_avgs - 
                    waist_avgs, axis=1).astype(int).reshape(-1, 1)
    
    # ensure torso length is at least 2x head_width, or worst-case >= 1
    torso_lengths = np.maximum.reduce([torso_lengths,
                            head_widths*2,
                            np.ones(shape=torso_lengths.shape)])
    
    # adjust torso lengths on multiplier, then roundup & convert to int
    torso_lengths = np.ceil(torso_lengths*factor).astype(int).reshape(-1, 1)
    
    # find head box xmins, ymins, widths and heights
    x_mins = head_x_avgs - (torso_lengths / 2.0)
    y_mins = head_y_avgs - (torso_lengths / 2.0)
    w = torso_lengths.copy()
    h = torso_lengths.copy()
    
    # ensure xmins and y mins are not below zero:
    x_mins = np.maximum(x_mins, 0).astype(int)
    y_mins = np.maximum(y_mins, 0).astype(int)
    
    return np.column_stack([x_mins, y_mins, w, h])


def get_image_region_array(img_array, head_regions, 
						   reshape_size, normalise=True):
    """ Helper function to gather head regions tensor.

    Args:
        df (pd.DataFrame) : pandas dataframe with pose results.
        img_name (str) : string containing exact name of image to plot.
        image_dir (str) : string containing path to image directory.
    
    Returns:
        tensor_stack (tf tensor) : tensor with resized image regions.
    """
    
    # get x max and y max of img to ensure we don't go out-of-bounds
    y_max, x_max = img_array.shape[:2]
    
    # create a list of numpy arrays with our images
    img_stack = [np.expand_dims(
                    img_array[reg[1]: reg[1] + reg[3], 
                              reg[0]: reg[0] + reg[2]], axis=0) 
                 for reg in head_regions]
    
    # convert list of np arrays into ragged tensor
    tensor_stack = tf.ragged.constant(img_stack)

    # resize all images within our ragged tensor
    tensor_stack = tf.concat(
        [tf.image.resize(tensor_stack[i].to_tensor(), reshape_size) 
         for i in tf.range(tensor_stack.nrows())], axis=0)
    
    if normalise:
        tensor_stack = tensor_stack / 255.0
    
    return tensor_stack


def model_predict_probs(model, image_batch, softmax=True):
    """ Helper function for making probability predictions 
        on an image batch 

    Args:
        model (TFHub Model) : Trained model for making predictions.
        image_batch (tf.tensor) : Tensor containing images for prediction.
        softmax (bool) : Whether to apply softmax activation or not.
    
    Returns:
        preds (np.array) : 2D output array of image predictions. First column 
                           is output for mask, second is for no_mask.
    """
    preds = model(image_batch)
    if softmax:
        return tf.keras.layers.Softmax()(preds).numpy()
    else:
        return preds


def get_prediction_labels(preds, threshold=None):
    """ Get output prediction labels (0 = no_mask, 1 = mask) 
    
    Args:
        preds (np.array) : 2d array containing class probabilities.
        threshold (float) : class threshold, based on second preds column.
    """
    # if theshold chosen, find label based on probability
    if threshold is not None:
        return tf.where(preds[:, 1] >= threshold, 1, 0)
    # otherwise simply return argmax for class label
    else:
        return tf.argmax(preds, axis=1)