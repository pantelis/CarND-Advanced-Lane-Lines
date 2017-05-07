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


all_objpoints = []
all_imgpoints = []

for fname in images:
        ret, objpoints, imgpoints = helper.find_object_image_points(fname, nx, ny, True)

        # some images may not result in detected corners - only append those that do
        if ret:
            all_objpoints.append(objpoints)
            all_imgpoints.append(imgpoints)

# save in the pickled dict of calibration input data
calibration_input_pickle = {}
calibration_input_pickle["objpoints"] = all_objpoints
calibration_input_pickle["imgpoints"] = all_imgpoints
pickle.dump(calibration_input_pickle, open(os.path.join(cal_directory, 'calibration_input_dict.p'), "wb"))
