#!/usr/bin/env python

""" Configuration file for AlphaPose & Downstrean Modelling Pipeline

This file contains all of the variables and settings to be used by the
AlphaPose & Downstrean Modelling Pipeline Tool (run_pipeline.py).

Guidance:

Set all of the variables to below to the required values, and then re-run
the main pipeline, ensuring that this config file is passed correctly.


Author: BD Fraser, https://github.com/BenjaminFraser

"""

###############################################
####### ALPHAPOSE MODELLING SETTINGS ##########
###############################################

# object detector for human bounding boxes
OBJECT_DETECTOR = 'yolo3_mobilenet1.0_coco'

# estimator to extract poses from human bboxes
POSE_ESTIMATOR = 'alpha_pose_resnet101_v1b_coco'

# define how often to make predictions (2 times per second by default)
# a higher modelling fps will slow the response-time of our system down
MODELLING_FPS = 4

# length to resize video short size in before detector
VID_RESIZE_SHORT = 800



##################################################
####### MASK CLASSIFICATION MODEL SETTINGS #######
##################################################

# image processing settings (must match trained model)
IMG_SIZE = (160, 160)
RESIZE_TO = 128
BATCH_SIZE = 32

# mask class label dict 
MASK_PREDS_IDS = { 0 : 'No Mask', 1 : 'Mask'}

# probability theshold for classifying a person as masked
MASK_THRESHOLD = 0.7

# class id map for our mask model predictions
CLASS_ID_MAP = { 0 : 'No Mask',
                 1 : 'Mask'}

# color map dictionary for mask classes predictions
COLOR_MAP = {0 : [255.0, 0.0, 0.0],
             1 : [0.0, 255.0, 0.0]}

# specify marker colors depending on mask status
MASK_COLOR_MAP = {'No Mask' : 'tab:red', 'Mask' : 'tab:blue'}



############################################################
####### POSE BEHAVIOUR CLASSIFICATION MODEL SETTINGS #######
############################################################

# ID mappings and color map for behaviour classification
BEH_DIC = {0:"Standing", 1:"Sittting", 2:"Walking", 3:"Lying Down",4:"Others"}
BEH_color = {0:[225, 0, 0], 1:[0, 225, 0], 2:[0, 0, 225], 3:[225, 225, 0], 4:[225,0,225]}

# mappings for pose status risk factors (how they influence social risk)
# 0 = no added risk, 1 = added risk (used by the weight computation)
BEH_RISK_FACTORS = {0:0, 1:1, 2:0, 3:1, 4:1}



#####################################################
####### SOCIAL DISTANCING ESTIMATION SETTINGS #######
#####################################################
# social distancing class label dict
SOCIAL_DIST_IDS = { 0 : 'Violating Distance', 1 : 'Safe Distance'}

# specify marker style depending on social distancing violation status
DIST_STYLE_MAP = {'Violating Distance' : 'X', 'Safe Distance' : 'o'}



#############################################
####### RISK DENSITY HEATMAP SETTINGS #######
#############################################
# bandwidth parameter adjustment for gaussian KDE density heatmap
# smaller = less smooth, smaller and more focussed cluster regions
BW_ADJUST = 0.4

# risk factors for each social distancing parameter (used for computing weights)
MASK_FACTOR = 0.5
VIOLATION_FACTOR = 1.0
POSE_STATUS_FACTOR = 0.5