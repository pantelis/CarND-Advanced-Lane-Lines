import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import helper_advanced
import os
import pickle
import configparser

if __name__ == "__main__":

    # Select the chessboard type
    # Parameters are a nested dictionary (addict library)
    config = configparser.ConfigParser()
    config.read('config.ini')

    chessboard = config.get('Calibration', 'chessboard')

    if chessboard == '8x6':
        cal_directory = '../CarND-Camera-Calibration/calibration_wide'
        test_directory = cal_directory
        # size of the chessboard in x and y directions
        nx = 8
        ny = 6
        # Test removal of distortion on this image
        img = cv2.imread(os.path.join(test_directory, 'test_image.jpg'))
    elif chessboard == '9x6':
        cal_directory = './camera_cal'
        test_directory = './test_images'
        # size of the chessboard in x and y directions
        nx = 9
        ny = 6
        # Test removal of distortion on this image
        img = cv2.imread(os.path.join(test_directory, 'test1.jpg'))

calibration_input_dict = pickle.load(open(os.path.join(cal_directory, 'calibration_input_dict.p'), 'rb'))
objpoints = calibration_input_dict["objpoints"]
imgpoints = calibration_input_dict["imgpoints"]

# Do camera calibration given object points and image points
undistorted, mtx, dist = helper_advanced.calibration_undistort(img, objpoints, imgpoints)

# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
calibration_output_dict = {}
calibration_output_dict["mtx"] = mtx
calibration_output_dict["dist"] = dist
pickle.dump(calibration_output_dict, open(os.path.join(cal_directory, 'calibration_output_dict.p'), "wb"))

#cv2.imwrite(os.path.join('output_images', 'test_camera_calibration.jpg'), undistorted)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image (BGR)', fontsize=30)
ax2.imshow(undistorted)
ax2.set_title('Undistorted Image (BGR)', fontsize=30)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.savefig(os.path.join('output_images', 'test_camera_calibration.jpg'))
plt.show()