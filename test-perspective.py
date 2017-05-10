import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import helper
import glob


# Read in the saved camera matrix and distortion coefficients
# These are the arrays we calculated using cv2.calibrateCamera()
cal_directory = '../CarND-Camera-Calibration/calibration_wide'
calibration_output_dict = pickle.load(open(os.path.join(cal_directory, 'calibration_output_dict.p'), 'rb'))
mtx = calibration_output_dict["mtx"]
dist = calibration_output_dict["dist"]


# Read in an image - image has height 960 pixels and width 1280 pixels
nx = 8  # the number of inside corners in x
ny = 6  # the number of inside corners in y

image_filenames = glob.glob(os.path.join(cal_directory, 'test_image2.jpg'))
draw_flag = True

# read in each image
for fname in image_filenames:

    img = cv2.imread(fname)

    # since we read with cv2.imread the gray transform is BRG2GRAY inside
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    warped, perspective_M, src, dst = helper.corners_warp(img, nx, ny, mtx, dist)

    ret = True
    h, (h1, h2) = plt.subplots(1, 2, figsize=(24, 9))
    h.tight_layout()
    h1.imshow(img)
    h1.set_title('Original', fontsize=30)
    h2.imshow(warped)
    h2.set_title('Warped', fontsize=30)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.savefig(os.path.join(cal_directory, 'original-vs-warped.jpg'), bbox_inches='tight')

