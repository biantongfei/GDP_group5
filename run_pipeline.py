#!/usr/bin/env python

""" AlphaPose & Downstrean Modelling Pipeline Tool

This CLI tool will take a given input video source and apply the system
modelling pipeline, comprising pose estimation & downstream modelling. The results
will be saved to the chosen directory given when calling the script.

Guidance:

Run the script with a source file (--file argument) full path argument and 
specified area name (within airport context, e.g. security) (--area_name argument). 
The source file should be a video souce, either .mp4, .mov etc, which should
be located within the dir given by area_name.

    python run_pipeline.py --file Security.mp4 --area_name Security

Optional arguments include the following:
    - '--pose_vid, -m' : Whether to save pose features as video results.
    - '--heatmap' -h' : Whether to save heatmap on scene as video results.
    - '--data_dir' -d' : Directory containing main data for project.
    - '--model_dir' -m' : Directory containing pre-traine models for project.


Author: BD Fraser, https://github.com/BenjaminFraser

"""

# import all dependencies and packages
import argparse
import cv2
import io
import mxnet as mx
import gluoncv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import pandas as pd
import PIL.Image as Image
import seaborn as sns
import skimage.io
import tensorflow as tf
import tensorflow_hub as hub
import sys

from collections import deque
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KernelDensity
from gluoncv.model_zoo import get_model
from gluoncv.data.transforms.pose import detector_to_alpha_pose, heatmap_to_coord
from gluoncv.utils.viz import cv_plot_image, cv_plot_keypoints, plot_image, cv_plot_bbox
from gluoncv import utils
from matplotlib.backends.backend_agg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from tensorflow import keras
from tqdm import tqdm

# import configuration variables for file (stored in system_modelling_config.py)
import modelling_config as model_cfg

# import local helper functions for distance estimation
from distance_estimation import to_3D, from_2D_to_3D, get_respect_social_distancing, \
								get_clusters

# import local helper functions for pose behaviour classification
from behaviour_classification import behavior_analysis

# import local helper functions for mask classification
from mask_classification import compute_head_boxes, get_image_region_array, \
								model_predict_probs, get_prediction_labels

# import local helper functions for density heatmap generation
from density_heatmap import get_sample_weights, get_density_heatmap_on_frame


def parse_command_args():
    """ Parse command line arguments if called from command line """
    parser = argparse.ArgumentParser(description='Apply system pipeline on video')

    # positional args - source file to make predictions on
    parser.add_argument('--file', '--f',
                       type=str, required=True,
                       help='Input video file to process.')
    parser.add_argument('--area_name', '--a',
                       type=str, default='data/predictions',
                       help='Dest data dir to create to save the results.')

    # optional - whether to save pose feature video results or not
    parser.add_argument('--pose_vid', '--p', type=bool,
                       default=True, 
                       help='Save pose features video results.')

    # optional - whether to save pose feature heatmap video or not
    parser.add_argument('--heatmap', '--h', type=bool,
                       default=True, 
                       help='Save heatmap video results.')

     # optional - path to data directory, uses 'static' by default
    parser.add_argument('--data_dir', '--d', type=str,
                       default='static', 
                       help='Data directory for inputs and results')

     # optional - path to model directory, uses default (current dir)
    parser.add_argument('--model_dir', '--m', type=str,
                       default='', 
                       help='Directory containing trained models.')

    args, _ = parser.parse_known_args()
    return args


def save_images_as_video(img_list, original_fps, modelling_fps, 
						 save_dir, save_filename, convert_rgb_to_bgr=True, 
						 encoding='mp4v'):
	""" Helper function to save image sequence as a video 

	Args:
		img_list (list) : list of ordered image arrays to save as video.
		original_fps (int) : frames-per-second (fps) of original video file.
		modelling_fps (int) : fps that was used for modelling.
		save_dir (str) : directory that video should be saved within.
		save_filename (str) : filename that the video should have, including ext.
		convert_rgb_to_bgr (bool) : If true, converts img array from rgb to bgr, as
									required for opencv.
		encoding (str) : Encoding type to use to encode the final video.
	"""

	# calculate how many times we need to duplicate frames
	# to match original input video fps
	fps_multiplier = int(original_fps / modelling_fps)

	# read all heatmap frames to cv2 format, ready to save
	img_array = []
	for i in range(len(img_list)):

		# if needed, load img and convert RGB to BGR (for opencv)
		if convert_rgb_to_bgr:
			cv_img = np.array(img_list[i].convert('RGB'))
			cv_img = cv_img[:, :, ::-1].copy()
		else:
			cv_img = pose_images[i][:, :, ::-1].copy()
    
    	# get dims for producing our final video
		height, width, layers = cv_img.shape
		size = (width,height)
    
    	# duplicate image req'd num of times to get original fps
		for dupl in range(fps_multiplier):
			img_array.append(cv_img)

	# configure video name, dest 
	vid_filepath = f"{save_dir}/{save_filename}"

	# if file already exists, need to remove first to create new one
	if os.path.isfile(vid_filepath):
		os.remove(vid_filepath)

	# configure encoding - may need adjusting depending on OS
	fourcc = cv2.VideoWriter_fourcc(*encoding)

	# configure output video writer settings
	out_video = cv2.VideoWriter(vid_filepath, fourcc, original_fps, size)

	# finally, write all the frames to video
	for i in range(len(img_array)):
		out_video.write(img_array[i])
	out_video.release()

	print(f"\nSuccessfully saved {vid_filepath} to results.")

	return


# if being called from command line, accept args and process data
if __name__ == '__main__':

	# get passed arguments
	args = parse_command_args()

	# set data directory appropriately
	PROJECT_PATH = os.getcwd()
	DATA_DIR = os.path.join(PROJECT_PATH, args.data_dir)
	MODELS_DIR = os.path.join(PROJECT_PATH, args.model_dir)

	# get positional args and set paths
	INPUT_FILENAME = args.file
	AREA_NAME = args.area_name

	# set area directory, and associated input video filepath
	AREA_DIR = os.path.join(DATA_DIR, AREA_NAME)
	INPUT_FILEPATH = os.path.join(AREA_DIR, INPUT_FILENAME)

    # ensure area input exists (if not new dir for this needs creating!)
	if not os.path.isdir(AREA_DIR):
		print(f'Area specified ({AREA_DIR}) does not exist. Please create.')
		sys.exit()

     # also ensure input video exists within the area dir above
	if not os.path.isfile(INPUT_FILEPATH):
		print(f'Specified file: ({INPUT_FILEPATH}) does not exist. Please check path.')
		sys.exit()

	# load pre-trained mask classification model
	mask_model_dir = os.path.join(MODELS_DIR, 'mask_clf_model')
	mask_model = hub.KerasLayer(mask_model_dir, trainable=False)

	# load pose bahavior analysis model
	behavior_model_dir = os.path.join(MODELS_DIR, 
									  'pose_ana_model', 'five_9304.h5')
	behavior_model = tf.keras.models.load_model(behavior_model_dir)

	# load and initialise alphapose modelling components
	# either cpu or gpu subject to architecture
	ctx = mx.cpu()

	# load selected object detector (as set in config file)
	detector = get_model(model_cfg.OBJECT_DETECTOR, pretrained=True, ctx=ctx)
	detector.reset_class(classes=['person'], reuse_weights={'person':'person'})
	detector.hybridize()

	# load selected pose estimator (as selected from config file)
	estimator = get_model(model_cfg.POSE_ESTIMATOR, pretrained=True, ctx=ctx)
	estimator.hybridize()

	print(f"\nStarting pipeline processing for {args.area_name}...")
	print(f"{'-'*80}")

    # load uploaded video into opencv:
	cap = cv2.VideoCapture(INPUT_FILEPATH)

	# get video fps, and total number of frames
	fps = cap.get(cv2.CAP_PROP_FPS)
	total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

	# set current frame to zero
	frame_count = 0

	# dictionary for scene-summary results (e.g. person counts, mask %)
	frame_results = {'image_id' : [],
					'person_count' : [],
					'clusters_count' : [],
					'social_distancing_compliance' : [],
					'mask_count' : [],
					'mask_proportions' : [], 
					'standing_count' : [],
					'sitting_count' : [],
					'standing_proportions' : [],
					'frame_H' : [],
					'frame_theta' : [],
					'frame_f' : [],
					'frame_dim_x' : [],
					'frame_dim_y' : [],
					'total_risk_profile' : []}

	# dictionary for per-person level results (bbox, keypoints, mask_on etc.)
	person_results = {'image_id' : [],
					  'bbox' : [],
					  'keypoints' : [],
					  'confidences' : [],
					  'position' : [],
					  'in_cluster' : [],
					  'respect_social_distancing' : [],
					  'behavior' : [],
					  'behavior_score' : [],
					  'mask_preds' : [],
					  'mask_pred_probs' : [],
					  'mask_head_regions' : [],
					  'person_risk_weights' : []}


	# container to store risk density-heatmaps
	frame_heatmaps = []

	# container to store pose feature images
	pose_images = []

	# parameters tracking
	time_step_average = 120
	theta_queue = deque()
	H_queue = deque()
	f_queue = deque()


	# process video frames by extracting pose features and applying downstream models
	while(True):
	    
	    print(f"- Currently processing Frame {int(frame_count)} out of {total_frames}...")

	    # check video capture, and obtain current frame (image from video)
	    ret, vid_frame = cap.read()

	    # if our video is still being processed, do the following...
	    if(ret):

	        # preprocess frame as required for object detector & AlphaPose
	        frame = mx.nd.array(cv2.cvtColor(vid_frame, 
	                                         cv2.COLOR_BGR2RGB)).astype('uint8')
	        x, frame = gluoncv.data.transforms.presets.ssd.transform_test(frame, 
	                                          short=model_cfg.VID_RESIZE_SHORT)
	        x = x.as_in_context(ctx)

	        # obtain human bounding boxes using object detector
	        class_IDs, scores, bounding_boxs = detector(x)

	        # get pose estimations using AlphaPose
	        pose_input, upscale_bbox = detector_to_alpha_pose(frame, class_IDs, 
	                                                          scores, bounding_boxs)

	        # add frame id, person count to frame-level results
	        image_id = f"frame_{int(frame_count)}"
	        n_people = upscale_bbox.shape[0]
	        frame_results['image_id'].append(image_id)
	        frame_results['person_count'].append(n_people)

	        # add frame_id, person bbox to person-level results
	        person_results['image_id'].append([image_id for x in range(n_people)])
	        person_results['bbox'].append(upscale_bbox)

	        # if we have predictions, obtain keypoint co-ords and confidences
	        if upscale_bbox is not None:

	            # obtain predicted heatmap
	            predicted_heatmap = estimator(pose_input.as_in_context(ctx))

	            # obtain keypoint co-ordinates from heatmap results
	            pred_coords, confidence = heatmap_to_coord(predicted_heatmap, 
	                                                       upscale_bbox)
	            
	            # add person keypoints and confidences to person-level results
	            person_results['keypoints'].append(pred_coords.asnumpy())
	            person_results['confidences'].append(confidence.asnumpy())

	            ########################################################
	            ######### DOWNSTREAM MODELLING PROCESSING AREA #########
	            ########################################################
	            
	            ####### DISTANCE ESTIMATION / CLUSTERING #########
	            
	            # obtain x / y dims of frame
	            image_dim_y, image_dim_x = frame.shape[:2]
	            
	            # obtain estimation of parameters of the camera 
	            _, _, theta, H, f = from_2D_to_3D(pred_coords, image_dim_x, 
	                                              image_dim_y)
	            
	            # update parameters tracking lists
	            if len(theta_queue) > time_step_average:
	                theta_queue.popleft()
	                H_queue.popleft()
	                f_queue.popleft()
	            theta_queue.append(theta)
	            H_queue.append(H)
	            f_queue.append(f)
	            
	            # use z-scores to remove extreme values (95% confidence)
	            zscore_select_theta = np.where(
	                            np.abs((np.array(theta_queue) - np.mean(theta_queue)) 
	                                   / np.std(theta_queue)) < 1.96)[0]
	            
	            zscore_select_H = np.where(
	                        np.abs((np.array(H_queue) - np.mean(H_queue)) 
	                               / np.std(H_queue)) < 1.96)[0]
	            
	            zscore_select_f = np.where(
	                        np.abs((np.array(f_queue) - np.mean(f_queue)) 
	                               / np.std(f_queue)) < 1.96)[0]
	            
	            zscore_select = np.intersect1d(zscore_select_theta, 
	                                           np.intersect1d(zscore_select_H, 
	                                                          zscore_select_f))
	            # remove extreme values
	            if len(zscore_select) > 10:
	                theta = np.mean(np.array(theta_queue)[zscore_select])
	                H = np.mean(np.array(H_queue)[zscore_select])
	                f = np.mean(np.array(f_queue)[zscore_select])
	            else:
	                theta, H, f = np.mean(theta_queue), np.mean(H_queue), np.mean(f_queue)
	            
	            # obtain estimation of persons' positions
	            X, Y, _, _, _ = from_2D_to_3D(pred_coords, image_dim_x, 
	                                          image_dim_y, theta, H, f)
	            
	            # get positions into a single array
	            positions = np.column_stack([X, Y])
	            
	            # obtain social distancing compliance classification
	            respect_social_distancing = get_respect_social_distancing(positions, 
	                                                                      social_distance=2)
	            
	            # get compliance proportion in scene
	            compliance_prop = np.array(np.bincount(respect_social_distancing, 
	                                                   minlength=2)[1] 
	                                       / respect_social_distancing.shape[0])
	            
	            # obtain social clusters
	            persons_clusters = get_clusters(positions, treshold=1)
	            
	            # get cluster classification (0 = not in a cluster, 1 = in a cluster)
	            in_cluster = (persons_clusters >= 0).astype(int)
	            
	            # append positions, clusters and social distancing compliance results to per-person results
	            person_results['position'].append(positions)
	            person_results['in_cluster'].append(in_cluster)
	            person_results['respect_social_distancing'].append(respect_social_distancing)

	            # add number of clusters and compliance proportion to frame-results
	            frame_results['clusters_count'].append(np.max(persons_clusters)+1)
	            frame_results['social_distancing_compliance'].append(compliance_prop)
	            
	            # add the frame img dimensions and derived homography params to frame results
	            frame_results['frame_H'].append(H)
	            frame_results['frame_theta'].append(theta)
	            frame_results['frame_f'].append(f)
	            frame_results['frame_dim_x'].append(image_dim_x)
	            frame_results['frame_dim_y'].append(image_dim_y)
	            
	            
	            ####### POSE STATUS CLASSIFICATION ############
	            
	            # get predictions for pose status
	            label, score = behavior_analysis(pred_coords, confidence,  
	            								 upscale_bbox, 
	            								 frame.shape[1], frame.shape[0], 
	            								 behavior_model)
	            
	            # add results to our person-level results
	            person_results['behavior'].append(label)
	            person_results['behavior_score'].append(score)
	            
	            # determine counts & proportions for status of scene
	            status_counts = np.bincount(label, minlength=5)

	            # simplify standing by combining standing [0] & walking [2]
	            stand_count = status_counts[0] + status_counts[2]
	            
	            # simplify sitting by combining sitting [1], lying [3] & other [4]
	            sitting_count = status_counts[1]+status_counts[3]+status_counts[4]
	        
	            # append sitting and standing counts to frame results
	            frame_results['standing_count'].append(stand_count)
	            frame_results['sitting_count'].append(sitting_count)
	            
	            # add standing proportion to frame results
	            standing_prop = np.array(stand_count / label.shape[0])
	            frame_results['standing_proportions'].append(standing_prop)

	            # get pose status risk-factor array from pose status labels
	            pose_factors = np.vectorize(model_cfg.BEH_RISK_FACTORS.__getitem__)(label)
	            
	            
	            ####### MASK CLASSIFICATION CODE #######

	            # obtain head regions boxes from our pose features
	            head_regions = compute_head_boxes(pred_coords)

	            # get extract tensor of head regions from original frame
	            region_tensor = get_image_region_array(frame, head_regions, 
	            					reshape_size=(model_cfg.RESIZE_TO, model_cfg.RESIZE_TO))
	            
	            # obtain probabilities (softmax normalised) from model
	            mask_pred_probs = model_predict_probs(mask_model, region_tensor)

	            # obtain hard class labels from our probabilities - use threshold
	            mask_preds = get_prediction_labels(mask_pred_probs, 
	                                  model_cfg.MASK_THRESHOLD).numpy()

	            # get mask proportion in scene, add to our total results
	            mask_count = np.bincount(mask_preds, minlength=2)[1]
	            mask_prop = np.array(mask_count / mask_preds.shape[0])

	            # append mask results to per-person results
	            person_results['mask_preds'].append(mask_preds)
	            person_results['mask_pred_probs'].append(mask_pred_probs)
	            person_results['mask_head_regions'].append(head_regions)

	            # add mask counts and proportions to frame-results
	            frame_results['mask_count'].append(mask_count)
	            frame_results['mask_proportions'].append(mask_prop)

	            ############################################
	            ######### DOWNSTREAM MODELLING END #########
	            ############################################
	            
	            ############################################
	            ######### DENSITY HEATMAP MODELLING ########
	            ############################################
	            
	            # get frame params needed for density heatmap
	            hgraphy_params = [H, theta, f]
	            image_dims = [image_dim_x, image_dim_y]
	            
	            # get risk-based weights using predictions for each person
	            person_weights = get_sample_weights(mask_preds, 
	                                        respect_social_distancing,
	                                        pose_factors,
	                                        model_cfg.MASK_FACTOR, 
	                                        model_cfg.VIOLATION_FACTOR,
	                                        model_cfg.POSE_STATUS_FACTOR)
	            
	            # append person weights to our person results
	            person_results['person_risk_weights'].append(person_weights)
	            
	            # also add risk profile (summed person weights) to frame results
	            frame_results['total_risk_profile'].append(person_weights.sum())
	            
	            # get density heatmap for current frame 
	            try:
	                frame_heatmap = get_density_heatmap_on_frame(frame, positions, 
	                                                             person_weights, 
	                                                             hgraphy_params, 
	                                                             image_dims,
	                                                             bw_adjust=model_cfg.BW_ADJUST)
	            
	            # Catch LinAlgError (not positive definite matrix) - use previous
	            # also catch ValueError - Contour levels must be increasing
	            except (ValueError, np.linalg.LinAlgError) as e:
	                frame_heatmap = frame_heatmaps[-1].copy()
	            
	            # append current frame heatmap to total results
	            frame_heatmaps.append(frame_heatmap)
	                
	            ############################################
	            ######### DENSITY HEATMAP END ##############
	            ############################################
	            
	            # annotate our original frame with alphapose extracted keypoints
	            frame_cp = frame.copy()
	            img = cv_plot_keypoints(frame_cp, pred_coords, confidence, 
	           							class_IDs, bounding_boxs, scores, 
	           							box_thresh=0.5, keypoint_thresh=0.2)
	                
	            # add pose points to our pose images
	            pose_images.append(img)
	        
	        # increment current frame by amount selected by MODELLING_FPS
	        frame_count += fps / model_cfg.MODELLING_FPS
	        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_count))

	    # break loop if video is not available / ended
	    else:
	        break

	    if cv2.waitKey(1) == 27:
	        break

	# empty video capture cache
	cap.release()
	cv2.destroyAllWindows()

	# PROCESS RESULTS AND SAVE AS REQUIRED
	# standardise frame results as np arrays and then convert into dataframe
	final_frame_results = {}
	for key in frame_results.keys():
		print(f"Key {key}, length: {len(frame_results[key])}")
		final_frame_results[key] = np.array(frame_results[key])
	final_frame_results = pd.DataFrame(final_frame_results)

	# standardise person results as np arrays and then convert into dataframe
	final_person_results = {}
	for key in person_results.keys():
		final_person_results[key] = np.concatenate(person_results[key], axis=0)
		final_person_results[key] = final_person_results[key].tolist()
	final_person_results = pd.DataFrame(final_person_results)

	# set paths to save results as JSON files (within test video dir)
	PERSON_RESULTS_FILEPATH = os.path.join(AREA_DIR, 
								f"{AREA_NAME}_person_results.JSON")
	FRAME_RESULTS_FILEPATH = os.path.join(AREA_DIR, 
								f"{AREA_NAME}_frame_results.JSON")

	# save results as JSON files to area directory specified
	final_frame_results.to_json(FRAME_RESULTS_FILEPATH)
	final_person_results.to_json(PERSON_RESULTS_FILEPATH)

	print(f"\nSaved final JSON results to {AREA_NAME} directory successfully!")

	# if chosen, save video of pose features 
	save_images_as_video(pose_images, fps, model_cfg.MODELLING_FPS, 
						 save_dir=AREA_DIR, save_filename=f"{AREA_NAME}_Poses.mov",
						 convert_rgb_to_bgr=False)

	# if chosen, save video of heatmap on scene
	save_images_as_video(frame_heatmaps, fps, model_cfg.MODELLING_FPS, 
						 save_dir=AREA_DIR, save_filename=f"{AREA_NAME}_Heatmap.mov",
						 convert_rgb_to_bgr=True)
