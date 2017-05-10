import numpy as np
import cv2
import matplotlib.image as mpimg
import pickle

def find_object_image_points(img, nx, ny):

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ny*nx, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    # objpoints: 3d points in real world space
    # imgpoints: 2d points in image plane.

    # Find the chessboard corners
    ret, imgpoints = cv2.findChessboardCorners(img, (nx, ny), None)

    return ret, objp, imgpoints


def generate_calibration_input(images, nx, ny):

    all_objpoints = []
    all_imgpoints = []

    for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, objpoints, imgpoints = find_object_image_points(gray, nx, ny)

            # some images may not result in detected corners - only append those that do
            if ret:
                all_objpoints.append(objpoints)
                all_imgpoints.append(imgpoints)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (nx, ny), imgpoints, ret)
                cv2.imshow('img', img)
                cv2.waitKey(500)

            cv2.destroyAllWindows()

    return ret, all_objpoints, all_imgpoints



def calibration_undistort(img, objpoints, imgpoints):

    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)

    return undistorted, mtx, dist


def corners_warp(img, nx, ny, mtx, dist):

    # Undistort using mtx and dist
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)

    # cv2 expects image size in (width, height)
    img_width = undistorted.shape[1]
    img_height = undistorted.shape[0]
    img_size = (img_width, img_height)

    # Find the chessboard corners
    ret, objpoints, imgpoints = find_object_image_points(undistorted, nx, ny)

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

    return warped, M_perspective, src, dst


def sobel_transform_threshold(img, method='gradient-magnitude', orient='xy', sobel_kernel=3, mag_thresh=(0, 255),
                              angle_thresh=(0.7, 1.3)):

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


def process_image_advanced(image, objpoints, imgpoints, config):

    # Do camera calibration given object points and image points
    # Return the camera calibration matrix and distortion coefficients and apply a
    # distortion correction to raw images.
    undistorted, mtx, dist = calibration_undistort(image, objpoints, imgpoints)

    # * Use color transforms, gradients, etc., to create a thresholded binary image.
    # * Apply a perspective transform to rectify binary image ("birds-eye view").
    # * Detect lane pixels and fit to find the lane boundary.
    # * Determine the curvature of the lane and vehicle position with respect to center.
    # * Warp the detected lane boundaries back onto the original image.
    # * Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
    return

def process_video_advanced(image, config):

    return