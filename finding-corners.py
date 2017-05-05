import numpy as np
import cv2
import glob
import pickle
import helper
import os

# size of the chessboard in x and y directions
nx = 8
ny = 6

# Make a list of calibration images
cal_directory = '../CarND-Camera-Calibration/calibration_wide'
images = glob.glob(os.path.join(cal_directory, 'GOPR*.jpg'))

objpoints, imgpoints = helper.find_object_image_points(images, nx, ny, True)

# save in the pickled dict of calibration input data
calibration_input_pickle = {}
calibration_input_pickle["objpoints"] = objpoints
calibration_input_pickle["imgpoints"] = imgpoints
pickle.dump(calibration_input_pickle, open(os.path.join(cal_directory, 'calibration_input_dict.p'), "wb"))
