# Airport Integrated Crowd Monitoring and Social Distancing Analysis Platform
- Group Design Project (Group 5) 
- Cranfield University, Applied Artificial Intelligence MSc.

## Authors

- Ben Fraser, https://github.com/BenjaminFraser
- Bian Tongfei, https://github.com/biantongfei
- Brendan Copp, https://github.com/bdvcp
- Gurpreet Singh
- Orhan Keyvan, https://github.com/CaptainSaturn
- Valentin Sonntag, https://github.com/ValentinSonntag


## Introduction and Overview

This project addresses issues presented during the recent pandemic, in particular, the need to ensure safety and sufficient public social distancing within crowded regions, including airports. In order to achieve this, an integrated airport crowd-monitoring and analytics system is developed, which works on the principle of applying a range of applied AI techniques to existing video surveillance camera feeds. 

The core aim is to provide airports with the means to collect and analyse crowd-based data, in order to enhance existing processes and improve safety throughout all managed locations. 

The key innovation of the work is the combining of various novel areas of computer vision and distance homography into an integrated computer-vision based framework, built on a foundation of crowd-pose estimation, which feeds into a series of downstream modelling tasks.

The outputs of the system are presented to the end-user using a custom dashboard application.

All code for the project has been developed in Python and associated open-source packages, including Numpy, OpenCV, MXNet, TensorFlow, GluonCV, Flask and Dash.


## Main System Modelling Process

The overall modelling process can be broken down into three distinct phases:

1. Video input and pre-processing, followed by extraction of human pose features using crowd-pose estimation.

2. Downstream modelling, which applies a range of AI and computer-vision based techniques on the extracted crowd pose features.

3. Integration and presentation of results, using a customised dashboard application.



![System Overview](examples/system_overview.jpg?raw=True "Simplified System Overview Diagram")


## Downstream Models Summary (Phase 2)

The downstream modelling tasks apply a series of computer-vision and applied AI techniques to the extracted pose features for each frame. There are five downstream modelling tasks in total, including person counting, distance estimation, pose behaviour classification, mask classification, and group clustering. See detailed process below for more information on each of these, including the associated code files.

The overall purpose of the downstream models, along with how they work, is illustrated in the diagram below.


![Downstream Modelling Overview](examples/downstream_models.jpg?raw=True "Illustration of the downstream modelling process")



## Main System Modelling Code and Detailed Process (Phases 1 & 2)

The main system modelling loop extracts pose features from all persons in a given input video stream (i.e. .mp4, .mov video file, or a live-stream) using OpenCV and AlphaPose (with YOLO object detector as pre-estimation human bounding boxes, and GluonCV for AlphaPose estimation). Following this, the five downstream modelling tasks are applied to the pose features to obtain a collection of useful crowd-analytics and social distancing results, including:

1. Persons count within a given scene.
2. Distance estimation between all persons in a scene, including whether violation according to a threshold has occurred for each person (using functions defined in `distance_estimation.py`).
3. Mask classification, which provides predictions for all people on whether they are wearing a mask or not (using functions defined in `mask_classification.py`).
4. Pose behaviour classification, which predicts the current behaviour status (including standing, walking, sitting, lying and other) for all persons in a scene (using functions defined in `behaviour_classification.py`).
5. Group clustering / social analysis, which estimates the number of distinct groups of people in a scene, along with their associated group sizes (using functions defined in `distance_estimation.py`).

All of these results are obtained frame-by-frame on the given input video source, at the desired FPS setting given in the configuration file `system_configuration.py`.

All results, including JSON files and result videos are saved to the desired results location (by default this is for the associated airport area directory being processed, i.e. Terminal, Security).

This main system modelling process can be run on an input video stream (or file) by simply using the `run_pipeline.py` command-line-tool. This can either be run using Python in the terminal, like so:

    python run_pipeline.py --file 'input_example.mp4' -area_name Airport_Shops

Conversely, this can also be run using another program, e.g. the dashboard application or API for further processing or use of the extracted results.

All system modelling and configuration must be set appropriately in `system_configuration.py` for this to run successfully.



## Dashboard Application

The dashboard application is developed using Flask and Dash. It provides a summary of the results provided by the system, and allows exploration of these in real-time for each area throughout the airport. 

It also allows the extraction and saving of all results from each area as a .csv, for further processing and analysis as required.

Provide suitable area directories have been established in the project directory, the dashboard can be run as follows:

    python GDP_Dashboard.py

The application can then be accessed at localhost (http://127.0.0.1:8050/) by default.

An example of the application is shown below.


![Dashboard Overview](examples/dashboard_example.jpg?raw=True "Example of the dashboard design.")



