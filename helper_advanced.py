import numpy as np
import cv2
from moviepy.editor import VideoFileClip


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

    # get the transform matrix M and its inverse
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)

    # warp your image to a top-down view
    warped = cv2.warpPerspective(undistorted, M, img_size, flags=cv2.INTER_LINEAR)

    return warped, M, M_inv


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


def detect_lanes_histogram(binary_warped):

    y_num_pixels = binary_warped.shape[0]
    x_num_pixels = binary_warped.shape[1]

    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0] / 2):, :], axis=0)

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    x_midpoint = np.int(x_num_pixels / 2)
    y_midpoint = np.int(y_num_pixels / 2)
    leftx_base = np.argmax(histogram[:x_midpoint])
    rightx_base = np.argmax(histogram[x_midpoint:]) + x_midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height =    np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
        nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
        nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit_coeff = np.polyfit(lefty, leftx, 2)
    right_fit_coeff = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, y_num_pixels - 1, y_num_pixels)
    left_fitx = left_fit_coeff[0] * ploty ** 2 + left_fit_coeff[1] * ploty + left_fit_coeff[2]
    right_fitx = right_fit_coeff[0] * ploty ** 2 + right_fit_coeff[1] * ploty + right_fit_coeff[2]

    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # plt.imshow(out_img)
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    # plt.xlim(0, binary_warped.shape[1])
    # plt.ylim(binary_warped.shape[0], 0)

    # Estimate the curvature
    # Define y-value where we want radius of curvature
    # We choose the the maximum y-value, corresponding to the bottom of the image
    y_eval = y_num_pixels

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    left_curvature, right_curvature = estimate_curvature(leftx, lefty, rightx, righty, y_eval, ym_per_pix, xm_per_pix)

    # You can assume the camera is mounted at the center of the car, such that the lane center
    # is the midpoint at the bottom of the image between the two lines you've detected.
    # The offset of the lane center from the center of the image (converted from pixels to meters)
    # is your distance from the center of the lane.
    offset = (abs(left_fitx[y_eval-1] - right_fitx[y_eval-1])/2.0 + left_fitx[y_eval-1] - x_midpoint) * xm_per_pix

    return left_fitx, right_fitx, left_curvature, right_curvature, offset


def estimate_curvature(leftx, lefty, rightx, righty, y_eval, ym_per_pix, xm_per_pix):
    # Assume that if you're projecting a section of lane, the lane is about 30
    # meters long and 3.7 meters wide. Or, if you prefer to derive a conversion from pixel space to world space in
    # your own images, compare your images with U.S. regulations that require a minimum lane width of 12 feet or 3.7
    # meters, and the dashed lane lines are 10 feet or 3 meters long each.

    # Fit new polynomials to x,y in world space

    # Fit a second order polynomial to each
    left_fit_coeff = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_coeff = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_coeff[0] * y_eval * ym_per_pix + left_fit_coeff[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_coeff[0])

    right_curverad = ((1 + (2 * right_fit_coeff[0] * y_eval * ym_per_pix + right_fit_coeff[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_coeff[0])

    # Now our radius of curvature is in meters
    # print(left_curverad, 'm', right_curverad, 'm')
    # Example values: 632.1 m    626.2 m
    return left_curverad, right_curverad

def project_lines(undistorted, binary_warped, left_fitx, right_fitx, Minv):

    '''
    Project from warped to world space. 
    project your measurement back down onto the road! Let's suppose, as in the previous example,
    you have a warped binary image called warped, and you have fit the lines with a polynomial
    and have arrays called ploty, left_fitx and right_fitx, which represent the x and y pixel values of the lines.
    '''

    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    new_warp = cv2.warpPerspective(color_warp, Minv, (undistorted.shape[1], undistorted.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB), 1, new_warp, 0.3, 0)

    return result


def process_image_advanced(image, objpoints, imgpoints):

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

    # cv2 expects image size in (width, height)
    img_width = combined_binary.shape[1]
    img_height = combined_binary.shape[0]
    img_size = (img_width, img_height)
    midpoint = img_width / 2

    # Apply a perspective transform to rectify binary image ("birds-eye view").
    src = np.float32([[midpoint - 440, img_height - 40],
                      [midpoint - 100, img_height - 250],
                      [midpoint + 100, img_height - 250],
                      [midpoint + 440, img_height - 40]])

    dst = np.float32(
        [[midpoint - 300, img_height],
         [midpoint - 300, 0],
         [midpoint + 300, 0],
         [midpoint + 300, img_height]])

    warped, M, Minv = corners_warp(combined_binary, img_size, src, dst)

    # Detect lane pixels and fit to find the lane boundary.
    left_fitx, right_fitx, left_curvature, right_curvature, offset = detect_lanes_histogram(warped)

    # Warp the detected lane boundaries back onto the original image.
    projected_img = project_lines(undistorted, warped, left_fitx, right_fitx, Minv)
    curvature = (left_curvature + right_curvature) / 2.0

    cv2.putText(projected_img, 'Curvature = ' + str(round(curvature, 0)) + ' (m)', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(projected_img, 'Offset = ' + str(round(offset, 2)) + ' (m)', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
    return undistorted, combined_binary, warped, projected_img


def process_video_advanced(clip, objpoints, imgpoints):

    def detect_lanes_video(image):
        undistorted, combined_binary, warped, projected_img = process_image_advanced(image, objpoints, imgpoints)
        return projected_img

    return clip.fl_image(detect_lanes_video)
