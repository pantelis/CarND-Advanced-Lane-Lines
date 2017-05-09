import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import helper
import os
import pickle

# Read in the saved objpoints and imgpoints
cal_directory = '../CarND-Camera-Calibration/calibration_wide'
calibration_input_dict = pickle.load(open(os.path.join(cal_directory, 'calibration_input_dict.p'), 'rb'))
objpoints = calibration_input_dict["objpoints"]
imgpoints = calibration_input_dict["imgpoints"]

# Test removal of distortion on an image
img = cv2.imread(os.path.join(cal_directory, 'test_image.jpg'))

undistorted, mtx, dist = helper.calibration_undistort(img, objpoints, imgpoints)

# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
calibration_output_dict = {}
calibration_output_dict["mtx"] = mtx
calibration_output_dict["dist"] = dist
pickle.dump(calibration_output_dict, open(os.path.join(cal_directory, 'calibration_output_dict.p'), "wb"))

# Do camera calibration given object points and image points
cv2.imwrite(os.path.join(cal_directory, 'test_undist.jpg'), undistorted)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(undistorted)
ax2.set_title('Undistorted Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()