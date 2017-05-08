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
calibration_input_dict = pickle.load(open(os.path.join(cal_directory, 'calibration_output_dict.p'), 'rb'))
mtx = calibration_input_dict["mtx"]
dist = calibration_input_dict["dist"]


# Read in an image - image has height 960 pixels and width 1280 pixels
nx = 8  # the number of inside corners in x
ny = 6  # the number of inside corners in y

image_filenames = glob.glob(os.path.join(cal_directory, 'test_image2.jpg'))
draw_flag = True


def corners_unwarp(img, nx, ny, mtx, dist):

    # Undistort using mtx and dist
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)

    # cv2 expects image size in (width, height)
    img_width = undistorted.shape[1]
    img_height = undistorted.shape[0]
    img_size = (img_width, img_height)

    # save undistorted
    undistorted_filename = os.path.join(cal_directory,
                                        os.path.splitext(os.path.basename(fname))[0] + '_undist.jpg')
    cv2.imwrite(undistorted_filename, undistorted)

    # Find the chessboard corners
    ret, objpoints, imgpoints = helper.find_object_image_points(undistorted_filename, nx, ny, draw_flag)

    # If corners found:
    if ret:

        # source points are the 4 outer detected corners
        src = np.float32([imgpoints[0], imgpoints[nx-1], imgpoints[-1], imgpoints[-nx]])

        # destination points are 4 outer points in a reference (ideal) image
        offset = 100

        dst = np.float32([[offset, offset], [img_size[0] - offset, offset],
                          [img_size[0] - offset, img_size[1] - offset],
                          [offset, img_size[1] - offset]])

        # get the transform matrix M
        M_perspective = cv2.getPerspectiveTransform(src, dst)

        # warp your image to a top-down view
        warped = cv2.warpPerspective(undistorted, M_perspective, img_size, flags=cv2.INTER_LINEAR)

        if draw_flag:
            f, (f1, f2) = plt.subplots(1, 2, figsize=(24, 9))
            f.tight_layout()
            f1.imshow(img)
            f1.set_title('Original Image', fontsize=50)
            f2.imshow(cv2.drawChessboardCorners(undistorted, (nx, ny), imgpoints, ret))
            f2.set_title('Undistorted Image with Detected Corners', fontsize=50)
            plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
            plt.savefig(os.path.join(cal_directory, 'original-vs-undistorted.jpg'), bbox_inches='tight')

            g, (g1, g2) = plt.subplots(1, 2, figsize=(24, 9))
            g.tight_layout()
            g1.imshow(cv2.drawChessboardCorners(undistorted, (nx, ny), src, ret))
            g1.set_title('Undistorted with Source Points', fontsize=50)
            g2.imshow(cv2.drawChessboardCorners(warped, (nx, ny), dst, ret))
            g2.set_title('Warped with Destination Points', fontsize=50)
            plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
            plt.savefig(os.path.join(cal_directory, 'undistorted-vs-warped.jpg'), bbox_inches='tight')

    return warped, M_perspective

# read in each image
for fname in image_filenames:
    img = cv2.imread(fname)
    # since we read with cv2.imread the gray transform is BRG2GRAY inside
    warped, perspective_M = corners_unwarp(img, nx, ny, mtx, dist)

    h, (h1, h2) = plt.subplots(1, 2, figsize=(24, 9))
    h.tight_layout()
    h1.imshow(img)
    h1.set_title('Original Image', fontsize=50)
    h2.imshow(warped)
    h2.set_title('Undistorted and Warped Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()




