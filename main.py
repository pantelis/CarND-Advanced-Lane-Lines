import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import helper
import configparser
from moviepy.editor import VideoFileClip
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import helper
import io


def process_image(image):

    # * Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
    # * Apply a distortion correction to raw images.
    # * Use color transforms, gradients, etc., to create a thresholded binary image.
    # * Apply a perspective transform to rectify binary image ("birds-eye view").
    # * Detect lane pixels and fit to find the lane boundary.
    # * Determine the curvature of the lane and vehicle position with respect to center.
    # * Warp the detected lane boundaries back onto the original image.
    # * Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


    # # Read in an image, you can also try test1.jpg or test4.jpg
    # img = mpimg.imread('test_images/test6.jpg')
    #
    # # Convert to HLS color space
    # hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #
    # gradx_binary = helper.sobel_transform_threshold(gray, method='gradient-magnitude', sobel_kernel=3,
    #                                                 orient='x', mag_thresh=(30, 100))
    # s_binary = helper.color_space_threshold(hls, channel='S', thresh=(170, 255))
    #
    # # Stack each channel to view their individual contributions in green and blue respectively
    # # This returns a stack of the two binary images, whose components you can see as different colors
    # color_binary = np.stack((np.zeros_like(gradx_binary), gradx_binary, s_binary))
    #
    # # Combine the two binary thresholds
    # combined_binary = np.zeros_like(gradx_binary)
    # combined_binary[(s_binary == 1) | (gradx_binary == 1)] = 1
    #
    # # Plotting thresholded images
    # f, ((f1, f2), (f3, f4)) = plt.subplots(2, 2, figsize=(20, 10))
    # f1.set_title('Gradient x')
    # f1.imshow(gradx_binary, cmap='gray')
    # f2.set_title('S Channel')
    # f2.imshow(s_binary, cmap='gray')
    # f3.set_title('Combined S channel and gradient thresholds')
    # f3.imshow(combined_binary, cmap='gray')
    # plt.show()
    #


# Import everything needed to edit/save/watch video clips





# Parameters are a nested dictionary (addict library)
config = configparser.ConfigParser()
config.read('config.ini')


def process_image_basic(image):
    # NOTE: this function expects color images

    # Pull out the x and y sizes and make a copy of the image
    ysize = image.shape[0]
    xsize = image.shape[1]
    region_select = np.copy(image)

    # Convert to Grayscale
    image_gray = helper.grayscale(image)

    # Blurring
    blurred_image = helper.gaussian_blur(image_gray, int(config.get('Blurring', 'kernel_size')))

    # Canny Transform
    edges = helper.canny(blurred_image, int(config.get('Canny', 'low_threshold')), int(config.get('Canny', 'high_threshold')))

    # masking
    # Four sided polygon to mask
    imshape = image.shape
    lower_left = (50, imshape[0])
    upper_left = (400, 320)
    upper_right = (524, 320)
    lower_right = (916, imshape[0])
    vertices = np.array([[lower_left, upper_left, upper_right, lower_right]], dtype=np.int32)

    masked_edges = helper.region_of_interest(edges, vertices)

    # Run Hough on edge detected image
    hough_lines, raw_hough_lines_img = helper.hough_lines(masked_edges,
                                                          int(config.get('Hough', 'rho')),
                                                          eval(config.get('Hough', 'theta')),
                                                          int(config.get('Hough', 'threshold')),
                                                          int(config.get('Hough', 'min_line_length')),
                                                          int(config.get('Hough', 'max_line_gap')))

    # classify left and right lane lines
    left_lane_lines, right_lane_lines = helper.classify_left_right_lanes(hough_lines)

    # Raw hough_lines image
    helper.draw_lines(raw_hough_lines_img, hough_lines, color=[255, 0, 0], thickness=2)

    # RANSAC fit left and right lane lines
    fitted_left_lane_points = helper.ransac_fit_hough_lines(left_lane_lines)
    fitted_right_lane_points = helper.ransac_fit_hough_lines(right_lane_lines)
    helper.draw_model(image, fitted_left_lane_points, color=[255, 0, 0], thickness=2)
    helper.draw_model(image, fitted_right_lane_points, color=[255, 0, 0], thickness=2)
clip = VideoFileClip("./project_video.mp4")

project_video_output_fname = './project_video_output.mp4'
output_clip = clip.fl_image(process_image_classic)

output_clip.write_videofile(project_video_output_fname, audio=False)
    # 1D Interpolator - does not work as good as RANSAC so its commented out
    # interpolated_left_lane_line = helpers.interpolate_hough_lines(left_lane_lines)
    # interpolated_right_lane_line = helpers.interpolate_hough_lines(left_lane_lines)
    # helpers.draw_model(image, interpolated_left_lane_line, color=[255, 0, 0], thickness=2)
    # helpers.draw_model(image, interpolated_right_lane_line, color=[255, 0, 0], thickness=2)

    # superpose images
    # superposed_image = helpers.weighted_img(image, raw_hough_lines_img, α=0.8, β=1., λ=0.)

    return image

if __name__ == "__main__":

    if config.get('Project', 'processing') == 'basic':
        clip = VideoFileClip("./project_video.mp4")
        project_video_output_fname = './project_video_output.mp4'
        output_clip = clip.fl_image(process_image_basic)
        output_clip.write_videofile(project_video_output_fname, audio=False)
    elif config.get('Project', 'processing') == 'advanced':
        clip = VideoFileClip("./project_video.mp4")
        project_video_output_fname = './project_video_output.mp4'
        output_clip = clip.fl_image(process_image_basic)
        output_clip.write_videofile(project_video_output_fname, audio=False)

