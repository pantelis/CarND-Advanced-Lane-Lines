import numpy as np
import cv2


def find_object_image_points(image_filename, nx, ny, draw_flag):

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ny*nx, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Step through the list and search for chessboard corners

    img = cv2.imread(image_filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If found, add object points, image points
    if draw_flag:
        if ret:
            objpoints = objp
            imgpoints = corners

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)

        cv2.destroyAllWindows()

    return ret, objpoints, imgpoints


def calibration_undistort(img, objpoints, imgpoints):

    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)

    return undistorted, mtx, dist


def sobel_transform_threshold(img, method='gradient-magnitude', orient='xy', sobel_kernel=3, mag_thresh=(0, 255), angle_thresh=(0.7, 1.3)):

    if method == 'gradient-magnitude':
        thresh_max = max(mag_thresh)
        thresh_min = min(mag_thresh)
        if orient == 'x':
            # Take the derivative in x (orient = 'x')
            sobel_direction = cv2.Sobel(img, cv2.CV_64F, 1, 0)
            # Take the absolute value of the derivative or gradient
            abs_sobel = np.absolute(sobel_direction)
        elif orient == 'y':
            # Take the derivative in y (orient = 'y')
            sobel_direction = cv2.Sobel(img, cv2.CV_64F, 0, 1)
            # Take the absolute value of the derivative or gradient
            abs_sobel = np.absolute(sobel_direction)
        elif orient == 'xy':
            # Take the derivative in x and y (orient = 'xy')
            sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0)
            sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1)
            abs_sobel = np.sqrt(sobel_x**2 + sobel_y**2)
        # Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

        # Create a mask of 1's where the scaled gradient magnitude
        # is > thresh_min and < thresh_max
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    elif method == 'gradient-angle':
        thresh_max = max(angle_thresh)
        thresh_min = min(angle_thresh)
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1)
        abs_sobelx = np.absolute(sobel_x)
        abs_sobely = np.absolute(sobel_y)
        grad_direction = np.arctan2(abs_sobely, abs_sobelx)

        # Create a mask of 1's where the scaled gradient magnitude
        # is > thresh_min and < thresh_max
        sxbinary = np.zeros_like(grad_direction)
        sxbinary[(grad_direction >= thresh_min) & (grad_direction <= thresh_max)] = 1

    return sxbinary

def color_space_threshold(img_hls, channel='S', thresh=(0, 255)):


    # Apply a threshold to the desired channel
    if channel == 'H':
        channel = img_hls[:, :, 0]
    elif channel == 'L':
        channel = img_hls[:, :, 1]
    elif channel == 'S':
        channel = img_hls[:, :, 2]

    binary_output = np.zeros_like(channel)
    binary_output[(channel > thresh[0]) & (channel <= thresh[1])] = 1

    return binary_output