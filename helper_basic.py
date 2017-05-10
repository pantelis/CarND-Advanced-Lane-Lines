import numpy as np
import cv2
from functools import reduce
import pandas as pd
from sklearn import linear_model
from scipy import interpolate
import itertools


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def classify_left_right_lanes(lines):

    left_lane_lines = []
    right_lane_lines = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            a = ((y2 - y1) / (x2 - x1)) # this assumes that the road is mostly straight and there is slope difference
            # in the field of view.
            if a > 0:
                right_lane_lines.append([x1, y1, x2, y2])
            else:
                left_lane_lines.append([x1, y1, x2, y2])

    # Apply spline interpolation between line segments of the left and right lines
    num_right_lines = len(right_lane_lines)
    num_left_lines = len(left_lane_lines)
    return np.array(left_lane_lines, dtype=np.int32).reshape(num_left_lines,1,4), \
           np.array(right_lane_lines, dtype=np.int32).reshape(num_right_lines,1,4)

def interpolate_hough_lines(lines):

    assert len(lines) > 1

    x=[]
    y=[]
    for line in lines:
        for x1, y1, x2, y2 in line:
            x.append(x1)
            x.append(x2)
            y.append(y1)
            y.append(y2)

    f = interpolate.interp1d(np.asarray(x), np.asarray(y), kind='linear')

    # Predict data of estimated models
    line_X = np.arange(min(x), max(x))
    line_y_interp = f(line_X[:, np.newaxis])

    points = list(zip(line_X, line_y_interp.astype(np.int64)))

    return points

def ransac_fit_hough_lines(lines):

    #assert len(lines) > 1

    x = []
    y = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            x.append(x1)
            x.append(x2)
            y.append(y1)
            y.append(y2)

    X=np.asarray(x).reshape(len(x),1)
    y=np.asarray(y)

    # Robustly fit linear model with RANSAC algorithm
    model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression(), residual_threshold=10)
    model_ransac.fit(X, y)
    inlier_mask = model_ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    # Predict data of estimated models
    line_X = np.arange(min(X[inlier_mask]), max(X[inlier_mask]))
    line_y_ransac = model_ransac.predict(line_X[:, np.newaxis])

    #line_segments=np.array([line_X, line_y_ransac])#.reshape(4,int(int(line_X.size)/int(2)))
    points = list(zip(line_X, line_y_ransac.astype(np.int64)))

    return points

def draw_model(img, points, color=[255, 0, 0], thickness=2):
    #assert len(points) > 1
    cv2.line(img, points[0], points[-1], color, thickness)

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """

    for line in lines:
         for x1, y1, x2, y2 in line:
             cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return lines, line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


#http://stackoverflow.com/questions/14692690/access-nested-dictionary-items-via-a-list-of-keys
def getFromDict(dataDict, mapList):
    for k in mapList: dataDict = dataDict[k]
    return dataDict

def nested_set(dic, keys, value):
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value

def keysInDict(dataDict, parent=[]):
    if not isinstance(dataDict, dict):
        return [tuple(parent)]
    else:
        return reduce(list.__add__,
            [keysInDict(v,parent+[k]) for k,v in dataDict.items()], [])

def dict_to_df(dataDict):
    ret = []
    for k in keysInDict(dataDict):
        v = np.array( getFromDict(dataDict, k), )
        v = pd.DataFrame(v)
        v.columns = pd.MultiIndex.from_product(list(k) + [v.columns])
        ret.append(v)
    return reduce(pd.DataFrame.join, ret)


def process_image_basic(image, config):
    # NOTE: this function expects color images

    # Pull out the x and y sizes and make a copy of the image
    ysize = image.shape[0]
    xsize = image.shape[1]
    region_select = np.copy(image)

    # Convert to Grayscale
    image_gray = grayscale(image)

    # Blurring
    blurred_image = gaussian_blur(image_gray, int(config.get('Blurring', 'kernel_size')))

    # Canny Transform
    edges = canny(blurred_image, int(config.get('Canny', 'low_threshold')), int(config.get('Canny', 'high_threshold')))

    # masking
    # Four sided polygon to mask
    imshape = image.shape
    lower_left = (50, imshape[0])
    upper_left = (400, 320)
    upper_right = (524, 320)
    lower_right = (916, imshape[0])
    vertices = np.array([[lower_left, upper_left, upper_right, lower_right]], dtype=np.int32)

    masked_edges = region_of_interest(edges, vertices)

    # Run Hough on edge detected image
    hough_lines, raw_hough_lines_img = hough_lines(masked_edges,
                                                          int(config.get('Hough', 'rho')),
                                                          eval(config.get('Hough', 'theta')),
                                                          int(config.get('Hough', 'threshold')),
                                                          int(config.get('Hough', 'min_line_length')),
                                                          int(config.get('Hough', 'max_line_gap')))

    # classify left and right lane lines
    left_lane_lines, right_lane_lines = classify_left_right_lanes(hough_lines)

    # Raw hough_lines image
    draw_lines(raw_hough_lines_img, hough_lines, color=[255, 0, 0], thickness=2)

    # RANSAC fit left and right lane lines
    fitted_left_lane_points = helper.ransac_fit_hough_lines(left_lane_lines)
    fitted_right_lane_points = helper.ransac_fit_hough_lines(right_lane_lines)
    helper.draw_model(image, fitted_left_lane_points, color=[255, 0, 0], thickness=2)
    helper.draw_model(image, fitted_right_lane_points, color=[255, 0, 0], thickness=2)

    # 1D Interpolator - does not work as good as RANSAC so its commented out
    # interpolated_left_lane_line = helpers.interpolate_hough_lines(left_lane_lines)
    # interpolated_right_lane_line = helpers.interpolate_hough_lines(left_lane_lines)
    # helpers.draw_model(image, interpolated_left_lane_line, color=[255, 0, 0], thickness=2)
    # helpers.draw_model(image, interpolated_right_lane_line, color=[255, 0, 0], thickness=2)

    # superpose images
    # superposed_image = helpers.weighted_img(image, raw_hough_lines_img, α=0.8, β=1., λ=0.)

    return image