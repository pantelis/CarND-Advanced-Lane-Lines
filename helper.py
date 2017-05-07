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