#!/usr/bin/env python
import rospy
import tf
import tf.transformations as tr
from std_msgs.msg import String, Header, ColorRGBA
from nav_msgs.msg import OccupancyGrid, MapMetaData, Odometry
from geometry_msgs.msg import Twist, PoseStamped, Point
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from math import sqrt, cos, sin, pi, atan2
from threading import Thread, Lock
from math import pi, log, exp
import random
import numpy as np
from scipy.optimize import least_squares
import sys
import yaml

import cv2
from matplotlib import pyplot as plt


def compute_depth_map(disparity_map, calib_params):
    """Return the depth map from the disparity. Recall that in the slides
    we saw disparities being expressed in meters whereas here they are 
    expressed in pixels, so you need to modify that formula accordingly,
    using the camera calibration parameters""" 

    # TODO: implement this
    depth_map = None
    return depth_map


def find_match_along_epipolar_line(img_left, img_right, row_left, col_left, patch_size):
    """Returns the column that best matches the patch of width path_size around (row_left, col_left)
    in the right image. Since these cameras are stereo rectified (parallel stereo cameras)
    you only need to search along the horizontal epipolar line on the right image,
    corresponding to the pixel (row_left, col_left). Note that pixel disparities cannot
    be negative, otherwise the estimated depth is going to be negative """

    # TODO: implement this
    col_right = 0
    return col_right
        
    
def compute_disparity_map(img_left, img_right, calib_params, patch_size, step):
    rows, cols = img_right.shape
    hps = patch_size/2
    
    disparity_map = np.zeros((rows/step, cols/step), dtype='float')
    
    for r in xrange(hps, rows-hps, step):
        print "Computing disparities along row", r
        for c in xrange(hps, cols-hps, step):
            c_right, _ = find_match_along_epipolar_line(img_left, img_right, r, c, patch_size)
            disparity_map[r / step, c / step] = c - c_right  
    
    return disparity_map

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print "Usage: stereo_disparity_map.py  path/to/image_left.png  path/to/image_right.png params.yaml"
        sys.exit(1)

    
    image0 = sys.argv[1]
    image1 = sys.argv[2]
    yaml_file = sys.argv[3]
    
    imgL = cv2.imread(image0, 0)
    imgR = cv2.imread(image1, 0)

    stream = open(yaml_file, "r")
    yaml_params = yaml.load(stream)
    
    K_left = np.array([yaml_params['fmx_left'], 0.0, yaml_params['cx_left'],
                       0.0, yaml_params['fmy_left'], yaml_params['cy_left'],
                       0.0, 0.0, 1.0]).reshape((3,3))

    K_right = np.array([yaml_params['fmx_right'], 0.0, yaml_params['cx_right'],
                       0.0, yaml_params['fmy_right'], yaml_params['cy_right'],
                       0.0, 0.0, 1.0]).reshape((3,3))

    baseline = yaml_params['baseline']
    
    calib_params = {'K_left': K_left,
                    'K_right': K_right,
                    'baseline': baseline}
    
    
    print "Rows =", imgL.shape[0], "Cols = ", imgL.shape[1]
    disparity_map = compute_disparity_map(imgL, imgR, calib_params, patch_size=yaml_params['patch_size'], step=yaml_params['step'])
    valid = (disparity_map < yaml_params['max_valid_disparity_in_pixels']) * (disparity_map > yaml_params['min_valid_disparity_in_pixels'])

    depth_map = compute_depth_map(disparity_map * valid, calib_params)

    plt.figure()
    plt.imshow(disparity_map * valid, cmap='jet_r')
    plt.colorbar()
    plt.title("Disparity (in pixels)")

    valid = (depth_map < yaml_params['max_valid_depth_in_meters']) * (depth_map > yaml_params['min_valid_depth_in_meters'])
    
    plt.figure()
    plt.imshow(depth_map * valid, cmap='jet_r')
    plt.colorbar()
    plt.title("Depth (in m)")
    
    plt.show()
