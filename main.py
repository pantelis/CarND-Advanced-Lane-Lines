import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import helper_basic
import helper_advanced
import configparser
from moviepy.editor import VideoFileClip
import glob
import os
import pickle

# Parameters are a nested dictionary (addict library)
config = configparser.ConfigParser()
config.read('config.ini')

draw_flag = config.get('Project', 'draw_flag')

if __name__ == "__main__":

    if config.get('Project', 'processing') == 'basic':
        clip = VideoFileClip("../FindingLaneLines/test_videos/white.mp4")
        project_video_output_fname = './white_with_lines.mp4'
        output_clip = clip.fl_image(helper_basic.process_image_basic, config)
        output_clip.write_videofile(project_video_output_fname, audio=False)
    elif config.get('Project', 'processing') == 'advanced':

        # Initially the test-finding-corners.py is used to create the pickled calibration input dict
        # that is stored in the camera_cal dir. This is a on-shot execution and the input to the script is the
        # chessboard type that points to the corresponding directory with the calibration images.

        chessboard = config.get('Calibration', 'chessboard')

        if chessboard == '8x6':
            cal_directory = '../CarND-Camera-Calibration/calibration_wide'
            # size of the chessboard in x and y directions
            nx = 8
            ny = 6
            images = glob.glob(os.path.join(cal_directory, 'GOPR*.jpg'))
        elif chessboard == '9x6':
            cal_directory = './camera_cal'
            # size of the chessboard in x and y directions
            nx = 9
            ny = 6
            images = glob.glob(os.path.join(cal_directory, 'calibration*.jpg'))

        if config.get('Calibration', 'Recalibration_flag') == 'True':

            # Given a set of chessboard images find the image points
            ret, objpoints, imgpoints = helper_advanced.generate_calibration_input(images, nx, ny)

        elif config.get('Calibration', 'Recalibration_flag') == 'False':

            # Read the calibration input data
            calibration_input_dict = pickle.load(open(os.path.join(cal_directory, 'calibration_input_dict.p'), 'rb'))
            objpoints = calibration_input_dict["objpoints"]
            imgpoints = calibration_input_dict["imgpoints"]

        if config.get('Project', 'media_type') == 'images':

            test_directory = './test_images'
            images = glob.glob(os.path.join(test_directory, '*.jpg'))

            for fname in images:
                image = cv2.imread(fname)
                undistorted, combined_binary, warped = helper_advanced.process_image_advanced(image, objpoints, imgpoints, draw_flag)

                h, ((h1, h2), (h3, h4)) = plt.subplots(2, 2, figsize=(24, 9))
                h.tight_layout()
                h1.imshow(undistorted)
                h1.set_title('Original', fontsize=30)
                h2.imshow(combined_binary)
                h2.set_title('Binary', fontsize=30)
                h3.imshow(warped)
                h3.set_title('Warped', fontsize=30)
                h4.imshow(combined_binary)
                h4.set_title('Binary', fontsize=30)
                plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
                plt.show()

                plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
                plt.savefig(os.path.join(cal_directory, 'original-vs-warped.jpg'), bbox_inches='tight')

        elif config.get('Project', 'media_type') == 'video':

            clip = VideoFileClip("./project_video.mp4")
            project_video_output_fname = './project_video_output.mp4'
            output_clip = clip.fl_image(helper_advanced.process_image_advanced, config)
            output_clip.write_videofile(project_video_output_fname, audio=False)

