#!/usr/bin/env python

""" Distance estimation helper functions.

This file contains all the helper functions required for performing
distance estimation and clustering for the system pipeline.

Author: Valentin Sonntag, https://github.com/ValentinSonntag

"""

import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.neighbors import KernelDensity

def to_3D(xf, yf, xh, yh, f, h=1.5):
    """ Gives the 3D euclidian coordinates of a person (on feet position) given
        its head and feet 2D coordinates in the image, the focal length of the
        camera and the average  (with origin of 3D coordinates system projected
        on the center of the 2D image, center of 2D coordinates system)

    Args:
        xf (float): abscissa of the feet position in the image
        yf (float): ordinate of the feet position in the image
        xh (float): abscissa of the head position in the image
        yh (float): ordinate of the head position in the image
        f (float): focal length of the camera
        h (float): average nose-to-feet height of a person (default 1.5)
                
    Method:
        Remove every error sensible operations or special cases, and do:
        x = xh*h/((f*(1-xh/xf))**2 + ((yh*xf-xh*yf)/xf)**2)**0.5
        H = x/xf*(f*(1-(x*(yh*xf-xh*yf)/(xh*xf*h))**2)**0.5
            - x*yf*(yh*xf-xh*yf)/(xh*xf*h))
        y = x*yf/xf/(1-(x*(yh*xf-xh*yf)/(xf*xh*h))**2)**0.5
        theta = np.arccos(x*(yh*xf-xh*yf)/(xf*xh*h))
    
    Returns:
        x (float): position of the person's feet along x-axis (axis located on the floor)
        y (float): position of the person's feet along y-axis (axis located on the floor)
        theta (float): angle of the camera from the ceiling, in rad
        H (float): height of the camera
    """
    # particular case, avoid division by 0 with simpler equations
    if xf * xh == 0:
        # fix values such as it can be removed later, in particular with H = -1
        return 0, 0, 0, -1

    else:
        # case due to high error or if top view and person in (0, 0)
        if xh == xf and yh == yf:
            return 0, 0, 0, -1

        x = xh*h/((f*(1-xh/xf))**2 + (yh-xh*yf/xf)**2)**0.5
        temp1 = (yh/xh - yf/xf)/h

        # case due to high error on the given coordinates
        if abs(x*temp1) >= 1:
            return 0, 0, 0, -1
        
        temp2 = (1-(x*temp1)**2)**0.5
        H = x/xf*(f*temp2 - x*yf*temp1)
        y = x*yf/(xf*temp2)
        theta = np.arccos(x*temp1)
        
    return x, y, theta, H


def from_2D_to_3D(pose_feats, image_dim_x, image_dim_y, theta=None, H=None, f=None):
    """ Gives the 3D euclidian coordinates of the persons in a image (on feet
        position) given their heads and feet 2D coordinates in the image, and
        the dimensions of the image (with origin of 3D coordinates system
        projected on the center of the 2D image, center of 2D coordinates system)

    Args:
        image_keypoints (mxnet.nd.NDArray): 2D array containing x and y co-ords columns
                                            for the 17 COCO keypoints (17 rows)
        image_dim_x (float): width of the image
        image_dim_y (float): height of the image
        theta (float): angle of the camera from the ceiling, in rad (default None)
        H (float): height of the camera
        f (float): focal length of the camera (default None)

    Returns:
        X (numpy.ndarray): positions of the persons' feet along x-axis (axis located on the floor)
        Y (numpy.ndarray): positions of the persons' feet along y-axis (axis located on the floor)
        theta (float): angle of the camera from the ceiling, in rad (default None)
        H (float): height of the camera (default None)
        f (float): focal length of the camera (default None)
    """
    image_keypoints = pose_feats.asnumpy()
    n_persons = image_keypoints.shape[0]
    eps = np.pi / 2 / 100
    
    # initialize some variables whether to determine theta or H
    if theta:
        theta_defined = True
    else:
        theta_defined = False
    if H:
        H_defined = True
    else:
        H_defined = False
    
    # translate the positions to the right 2D coordinate system
    feet = np.zeros((n_persons, 2))
    nose = np.zeros((n_persons, 2))
    for index in range(n_persons):
        # get keypoints of the person
        p = image_keypoints[index]

        # from img_array coordinates to 2D euclidian coordinates where origin is image center
        feet[index] = ((p[-1, 0]+p[-2, 0])/2 - image_dim_x/2, 
                       image_dim_y/2 - (p[-1, 1]+p[-2, 1])/2)
        
        nose[index] = p[0, 0] - image_dim_x/2, image_dim_y/2 - p[0, 1]
    
    # determine f if not given (with trichotomy)
    if not f:
        # interval boundaries for trichotomy (here 30° < FOV < 120°)
        b = [image_dim_x / (2 * np.tan(120/180 * np.pi/2)), 
             image_dim_x / (2 * np.tan(30/180 * np.pi/2))]
        # f = d_image / (2 * np.tan(FOV/180 * np.pi/2))

        while b[1]-b[0] > np.mean(b) / 50: # maximum 1% error on value found

            # trichotomy first third value
            f_test = b[0]+(b[1]-b[0])/10
            
            # determine 3D positions, theta and H for each person
            res = np.zeros((n_persons, 4))
            for i in range(n_persons):
                res[i] = to_3D(feet[i, 0], feet[i, 1], nose[i, 0], nose[i, 1], f_test)
            
            # remove wrong values
            selected = (res[:, 3] > 1) * (res[:, 2] > eps) * (res[:, 2] < np.pi/2 - eps)

            if np.sum(selected) == 0:
                selected = np.ones(n_persons, dtype=bool)

            # if not given, estimate theta and H: median to avoid outliers impact
            if not theta_defined:
                theta = np.median(res[selected, 2])
            if not H_defined:
                H = np.median(res[selected, 3])
            
            # calculate median error on positions: median to avoid outliers impact
            temp = f_test*np.sin(theta) / (H + np.sin(theta)*np.cos(theta)*res[selected, 1])
            r1 = np.median((temp*res[selected, 0] - feet[selected, 0])**2
                            + (temp*res[selected, 1]*np.sin(theta) - feet[selected, 1])**2)

            # trichotomy second third value
            f_test = b[0]+9*(b[1]-b[0])/10
            
            # determine 3D positions, theta and H for each person
            res = np.zeros((n_persons, 4))
            for i in range(n_persons):
                res[i] = to_3D(feet[i, 0], feet[i, 1], nose[i, 0], nose[i, 1], f_test)
            
            # remove wrong values
            selected = (res[:, 3] > 1) * (res[:, 2] > eps) * (res[:, 2] < np.pi/2 - eps)
        
            if np.sum(selected) == 0:
                selected = np.ones(n_persons, dtype=bool)
                
            # if not given, estimate theta and H: median to avoid outliers impact
            if not theta_defined:
                theta = np.median(res[selected, 2])
            if not H_defined:
                H = np.median(res[selected, 3])
            
            # calculate median error on positions: median to avoid outliers impact
            temp = f_test*np.sin(theta) / (H + np.sin(theta)*np.cos(theta)*res[selected, 1])
            r2 = np.median((temp*res[selected, 0] - feet[selected, 0])**2
                            + (temp*res[selected, 1]*np.sin(theta) - feet[selected, 1])**2)
            
            # compare values and update the interval boundaries
            if r1 > r2:
                b[0] = b[0]+(b[1]-b[0])/10
            else:
                b[1] = b[0]+9*(b[1]-b[0])/10
        
        # finally take the center of interval as value found
        f = np.mean(b)
    
    if not theta_defined or not H_defined:
        # final iteration with found/given f value
        res = np.zeros((n_persons, 4))
        for i in range(n_persons):
            res[i, :] = to_3D(feet[i, 0], feet[i, 1], nose[i, 0], nose[i, 1], f)

        # remove wrong values
        selected = (res[:, 3] > 1) * (res[:, 2] > eps) * (res[:, 2] < np.pi/2 - eps)
        
        if np.sum(selected) == 0:
            selected = np.ones(n_persons, dtype=bool)
    
        # If not given, estimate theta and H: use density to increase precision
        if not theta_defined:
            d_theta = np.median(res[selected, 2])
            density_range_theta = np.linspace(min(res[selected, 2]), 
                                              max(res[selected, 2]), 200)
            kde_skl = KernelDensity(bandwidth=d_theta)
            kde_skl.fit(res[selected, 2, np.newaxis])
            log_pdf = kde_skl.score_samples(density_range_theta[:, np.newaxis])
            density_theta = np.exp(log_pdf)
            theta = density_range_theta[np.argmax(density_theta)]
            
        if not H_defined:
            d_H = np.median(res[selected, 3])
            density_range_H = np.linspace(min(res[selected, 3]), max(res[selected, 3]), 200)
            kde_skl = KernelDensity(bandwidth=d_H)
            kde_skl.fit(res[selected, 3, np.newaxis])
            log_pdf = kde_skl.score_samples(density_range_H[:, np.newaxis])
            density_H = np.exp(log_pdf)
            H = density_range_H[np.argmax(density_H)]
    
    # 3D positions (top view)
    Y = H * feet[:, 1] / (f*np.sin(theta)**2 - feet[:, 1]*np.cos(theta)*np.sin(theta))
    X = H * feet[:, 0] / (f*np.sin(theta) - feet[:, 1]*np.cos(theta))
    
    return X, Y, theta, H, f


def get_respect_social_distancing(positions, social_distance=2):
    """ Get social distancing labels (0 = no respect, 1 = respect)

    Args:
        positions (numpy.ndarray): positions of people
        social_distance (float): social distancing treshold (default 2)
    
    Returns:
        respect_social_distancing (numpy.ndarray): whether the persons repect social
                                                   distancing (0 = no, 1 = yes)
    """
    # compute the distance matrix then find wheter people are respecting
    respect_social_distancing = (np.sum(
        np.linalg.norm(
            positions[:, None, :] - positions[None, :, :], axis=-1) 
        < social_distance, axis = 0) == 1).astype(int)

    return respect_social_distancing


def get_clusters(positions, treshold=1):
    """ Get clusters information (number of clusters and persons' clusters labels)

    Args:
        positions (numpy.ndarray): positions of people
        treshold (float): maximum distance separating a person from the closest person
                          in the cluster (default 1)
    
    Returns:
        persons_clusters (numpy.ndarray): clusters to which each person belongs
    """
    # compute the clusters, where a cluster is made by persons less than a 
    # treshold distance from another one of the same cluster
    clusters = DBSCAN(eps=treshold, min_samples=2).fit(positions)
    persons_clusters = clusters.labels_
    
    return persons_clusters