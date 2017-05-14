import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import helper_advanced
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
        img = cv2.imread(os.path.join(test_directory, 'test_image2.jpg'))
    elif chessboard == '9x6':
        cal_directory = './camera_cal'
        test_directory = './test_images'
        # size of the chessboard in x and y directions
        nx = 9
        ny = 6
        # Test removal of distortion on this image
        img = cv2.imread(os.path.join(test_directory, 'straight_lines2.jpg'))

    calibration_output_dict = pickle.load(open(os.path.join(cal_directory, 'calibration_output_dict.p'), 'rb'))
    mtx = calibration_output_dict["mtx"]
    dist = calibration_output_dict["dist"]

    # Undistort using mtx and dist
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)

    # cv2 expects image size in (width, height)
    img_width = undistorted.shape[1]
    img_height = undistorted.shape[0]
    img_size = (img_width, img_height)
    midpoint = img_width / 2

    if chessboard == '8x6':
        # Find the chessboard corners
        ret, objpoints, imgpoints = helper_advanced.find_object_image_points(undistorted, nx, ny)

        # source points are the 4 outer detected corners
        src = np.float32([imgpoints[0], imgpoints[nx-1], imgpoints[-1], imgpoints[-nx]])

        # destination points are 4 outer points in a reference (ideal) image
        offset = 100

        dst = np.float32([[offset, offset], [img_size[0] - offset, offset],
                          [img_size[0] - offset, img_size[1] - offset],
                          [offset, img_size[1] - offset]])

    elif chessboard == '9x6':

        src = np.float32([[midpoint - 440, img_height - 40],
                      [midpoint - 100, img_height - 250],
                      [midpoint + 100, img_height - 250],
                      [midpoint + 440, img_height - 40]])


        dst = np.float32(
            [[midpoint - 300, img_height],
             [midpoint - 300, 0],
             [midpoint + 300, 0],
             [midpoint + 300, img_height]])

    warped, M, Minv = helper_advanced.corners_warp(undistorted, img_size, src, dst)

    x_src = src.reshape(8, 1)[::2].squeeze()
    y_src = src.reshape(8, 1)[1::2].squeeze()
    x_dst = dst.reshape(8, 1)[::2].squeeze()
    y_dst = dst.reshape(8, 1)[1::2].squeeze()

    src_rectangle = zip(x_src, y_src)
    h, (h1, h2) = plt.subplots(1, 2, figsize=(24, 9))
    h.tight_layout()
    h1.imshow(undistorted)
    h1.scatter(x_src, y_src, c='b')
    h1.set_title('Original (BGR)', fontsize=30)
    h2.imshow(warped)
    h2.scatter(x_dst, y_dst, c='r')
    h2.set_title('Warped (RGB)', fontsize=30)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.savefig(os.path.join('output_images', 'original_vs_warped.jpg'), bbox_inches='tight')
    plt.show()
