import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
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


def corners_warp(undistorted, img_size, src, dst):

    # get the transform matrix M
    M_perspective = cv2.getPerspectiveTransform(src, dst)

    # warp your image to a top-down view
    warped = cv2.warpPerspective(undistorted, M_perspective, img_size, flags=cv2.INTER_LINEAR)

    return warped, M_perspective


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

# def project_lines(undistorted_img, warped_img, ploty, left_fitx, right_fitx)
#
#     '''
#     Once you have a good measurement of the line positions in warped space, it's time to
#     project your measurement back down onto the road! Let's suppose, as in the previous example,
#     you have a warped binary image called warped, and you have fit the lines with a polynomial
#     and have arrays called ploty, left_fitx and right_fitx, which represent the x and y pixel values of the lines.
#     '''
#
#     # Create an image to draw the lines on
#     warp_zero = np.zeros_like(warped).astype(np.uint8)
#     color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
#
#     # Recast the x and y points into usable format for cv2.fillPoly()
#     pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
#     pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
#     pts = np.hstack((pts_left, pts_right))
#
#     # Draw the lane onto the warped blank image
#     cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
#
#     # Warp the blank back to original image space using inverse perspective matrix (Minv)
#     newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
#
#     # Combine the result with the original image
#     result = cv2.addWeighted(undistorted, 1, newwarp, 0.3, 0)
#     plt.imshow(result)

def process_image_advanced(image, objpoints, imgpoints, draw_flag):

    # Do camera calibration given object points and image points
    # Return the camera calibration matrix and distortion coefficients and apply a
    # distortion correction to raw images.
    undistorted, mtx, dist = calibration_undistort(image, objpoints, imgpoints)

    # Use color transforms and gradients to create a thresholded binary image.
    # Convert to HLS and gray color spaces - input image is assumed to be in BGR format
    hls = cv2.cvtColor(undistorted, cv2.COLOR_BGR2HLS)
    gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)

    gradx_binary = sobel_transform_threshold(gray, method='gradient-magnitude', sobel_kernel=3,
                                                    orient='x', mag_thresh=(30, 100))
    s_binary = color_space_threshold(hls, channel='S', thresh=(170, 255))

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(gradx_binary)
    combined_binary[(s_binary == 1) | (gradx_binary == 1)] = 1

    # Apply a perspective transform to rectify binary image ("birds-eye view").

    warped, perspective_M, src, dst = corners_warp(combined_binary, 9, 6, mtx, dist)


    # Detect lane pixels and fit to find the lane boundary.

    # Determine the curvature of the lane and vehicle position with respect to center.

    # Warp the detected lane boundaries back onto the original image.

    # Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
    return undistorted, combined_binary, warped

def process_video_advanced(image, config):

    return