#!/usr/bin/env python

""" Heatmap helper functions.

This file contains all the helper functions required for generating the
risk-based density heatmap analysis for the system.

"""

import cv2
import io
import numpy as np
import PIL.Image as Image
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.backends.backend_agg import FigureCanvas
from matplotlib.figure import Figure


def get_density_heatmap_on_frame(frame, person_posns, person_weights, 
                                 hgraphy_params, image_dims, 
                                 bw_adjust, alpha=0.8,
                                 alpha_coeff=1., figsize=(8,7)):
    """ Helper function for plotting density heatmap of scene on original view.
    
    Args:
        frame (np.array) : Original image array.
        person_posns (np.array) : 2D array with top-down x/y co-ords persons.
        sample_weights (np.ndarray): risk-based weights for each sample.
        hgraphy_params (np.ndarray) : Array of three elements, containing
            frame homography H, theta & f, precisely as so: [H, theta, f].
        image_dims (np.array) : Array containing x and y dims of scene image.
        bw_adjust (float) : value for adjusting bandwidth of gaussian KDE.
        alpha (float) : heatmap transparency (must be > 0 to see heatmap)
        alpha_coeff (float) : alpha attenuation coefficient for lower risk values
        figsize (tuple): tuple of desired figure size to plot.
    """
    # get observed area and limits of our scene top-down plot
    limits = np.array([[-image_dims[0]/2, -image_dims[1]/2], 
                       [-image_dims[0]/2, image_dims[1]/2], 
                       [image_dims[0]/2, image_dims[1]/2], 
                       [image_dims[0]/2, -image_dims[1]/2]])
    
    # get x and y limits for scene
    limits_y = (hgraphy_params[0] * limits[:, 1] / 
                (hgraphy_params[2]*np.sin(hgraphy_params[1])**2 - 
                 limits[:, 1]*np.cos(hgraphy_params[1])*np.sin(hgraphy_params[1])))
    
    limits_x = limits_y * limits[:, 0] * np.sin(hgraphy_params[1]) / limits[:, 1]         
    
    # get axis and limits
    m_x, M_x = np.min(limits_x), np.max(limits_x)
    m_y, M_y = np.min(limits_y), np.max(limits_y)
    
    # get top-view density heatmap
    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.subplots()
    ax = fig.add_axes([0, 0, 1, 1])
    
    # plot gaussian KDE of our top-down scene
    sns.kdeplot(x=person_posns[:, 0], y=person_posns[:, 1], 
                shade=True, ax=ax, bw_adjust=bw_adjust, cmap="Reds", 
                weights=person_weights, alpha=1.0)
    ax.set_xlim(m_x, M_x)
    ax.set_ylim(m_y, M_y)
    canvas.draw()
    
    # get heatmap as image array
    heatmap = np.array(canvas.renderer.buffer_rgba())
    
    # get source and dest points to get homography H using opencv
    src_points = np.column_stack([(limits_x - m_x)/(M_x - m_x)*heatmap.shape[1],
                    (M_y - limits_y)/(M_y - m_y)*heatmap.shape[0]])
    # as above - dest points
    dst_points = (np.column_stack([limits[:, 0], -limits[:, 1]]) + 
                    np.array([image_dims[0]/2, image_dims[1]/2]))
    
    # get homography to convert top-view to original camera view
    homography, _ = cv2.findHomography(src_points, dst_points)
    
    # convert heatmap to original camera view
    heatmap = cv2.warpPerspective(heatmap, homography, (image_dims[0], image_dims[1]))
    
    # initialize alpha for heatmap
    h_alpha = np.ones_like(heatmap[:, :, 3], dtype=float) * alpha
    
    # add full transparency for blank areas
    h_alpha *= (heatmap[:, :, 0] != heatmap[:, :, 1])
    h_alpha[heatmap[:, :, 3] < 255] = 0
    
    # add transparency for areas with lower risks
    h_alpha *= (1 - heatmap[:, :, 1]/255)**alpha_coeff
    
    # get alpha for original image
    frame_alpha = 1 - h_alpha
    
    # get heatmap on original image
    img_out = ((frame * np.repeat(frame_alpha[:, :, None], 3, axis=2) +
                heatmap[:, :, 0:3] * np.repeat(h_alpha[:, :, None], 3, axis=2))).astype(int)
    
    # plot original image with transformed heatmap as overlay
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(img_out)
    plt.axis("off")
    plt.tight_layout()
    
    # save plot as BytesIO object and return img
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    return img


def get_sample_weights(mask_preds, violation_preds, status_preds,
                       mask_factor, violation_factor,
                       pose_status_factor):
    """ Compute risk-factor weights for each person for heatmap. 
        People that are in close proximity / violating are the only ones that
        contribute towards risk in the scene. 
        
    Args:
        mask_preds (np.array) : Mask predictions (0=no mask, 1=mask)
        violation_preds (np.array) : Whether people are violating or not.
        status_preds (np.array) : Whether people are standing or not.
        mask_factor (float) : Factor of risk for non-masked people.
        violation_factor (float) : Factor of risk for violating people.
        pose_status_factor (float) : Factor of risk for non-standing people.
        
    Returns:
       weights (np.array) : Weights for each person for given inputs.
    """
    
    # find people violating in scene - use to set initial weights
    are_violating = (1 - violation_preds)
    weights = violation_factor * are_violating
    
    # determine weights using sum of wearing mask, multiplied by mask_factor
    weights += mask_factor * (1 - mask_preds)
    
    # further tune weights based on sitting or standing predictions
    weights += pose_status_factor * status_preds
    
    # set all non-violating people back to zero, and keep weights for those violating
    final_weights = np.multiply(weights, are_violating)
    
    return weights