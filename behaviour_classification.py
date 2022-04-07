#!/usr/bin/env python

""" Pose Behaviour Classification helper functions.

This file contains all the helper functions required for pre-processing
input data and making predictions for pose behaviour classification.

Author: Bian Tongfei, https://github.com/biantongfei

"""

import numpy as np


def behavior_analysis(pred_coords, keypoint_confidences, 
                      bbox, width, height, model):
    """Using 34 cooradinate value and score of pose features, information of bounding box and model to analyze behavior

    Args:
        pred_coords (mxnet array) : 2D array containing x and y co-ords
                  columns for the 17 COCO keypoints (17 rows).
        keypoint_confidences (mxnet array) : 2D array containing score of 17 keppoints. 
        bbox (mxnet array) : 1D array containing bounding box information
        width (int) : width of the frame
        height (int) : height of the frame
        model (tf.keras.models.Sequential()) : FCNN model for behavior analysis
  
    Return:
        behavior_class (str) : name of behavior class, including standing, sitting, walking, lying down and others
        score (flost) : Score of classification results
    """
    # Prepare data
    pose_feats = pred_coords.asnumpy()
    confidence = keypoint_confidences.asnumpy()
    
    # Normalize data
    result = (pose_feats[:, :, 0].T - bbox[:, 0]) / (bbox[:, 2] - bbox[:, 0])
    pose_feats[:, :, 0] = result.T.copy()
    result = (pose_feats[:, :, 1].T - bbox[:, 1]) / (bbox[:, 3] - bbox[:, 1])
    pose_feats[:, :, 1] = result.T.copy()
    
    pose_feats = np.concatenate((pose_feats, confidence), axis=2)
    pose_feats = pose_feats.reshape(pose_feats.shape[0], -1)
    pose_feats = np.c_[pose_feats, (bbox[:, 2] - bbox[:, 0]) / (bbox[:, 3] - bbox[:, 1])]
    pose_feats = np.c_[pose_feats, (bbox[:, 2] - bbox[:, 0]) / width]
    pose_feats = np.c_[pose_feats, (bbox[:, 3] - bbox[:, 1]) / height]
    
    # predict class and get label and score
    result = model.predict(pose_feats)
    label = np.argmax(result, axis=1)
    score = np.max(result, axis=1)
    return label, score