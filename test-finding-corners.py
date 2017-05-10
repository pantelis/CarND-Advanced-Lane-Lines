import cv2
import glob
import pickle
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

    ret, all_objpoints, all_imgpoints = helper_advanced.generate_calibration_input(images, nx, ny)

    # save in the pickled dict of calibration input data
    calibration_input_pickle = {}
    calibration_input_pickle["objpoints"] = all_objpoints
    calibration_input_pickle["imgpoints"] = all_imgpoints
    pickle.dump(calibration_input_pickle, open(os.path.join(cal_directory, 'calibration_input_dict.p'), "wb"))
