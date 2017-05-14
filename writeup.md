# Advanced Lane Finding

Pantelis Monogioudis

NOKIA

---


This project identifies traffic lanes using the following several steps:

* The camera used in the car is calibrated given a set of chessboard images. 
* The calibration matrix and distortion coefficients is then used to correct the distortion of the raw images.
* Each image/frame is then converted to an HLS and GRAY color space. The later feeds a gradient in the x direction 
estimator. The HLS and gradient images are sutably thresholded and combined to create a binary image
 used in the subsequent step.  
* A perspective transform to this binary image is then applied resulting in a "birds-eye view".
* The lane pixels are searched using a sliding window method and the identified pixels are polynomial fit to find the lane boundary.
* The curvature of the lane and vehicle position with respect to the center point between the identified lanes is then measured.
* Lastly the detected lane boundaries are warped back onto the original image - the final image includes also the curvature and 
vehicle position estimates.

[//]: # (Image References)

[test_camera_calibration]: ./output_images/test_camera_calibration.jpg "Undistorted image after camera calibration"
[hls_gradient_magnitude_x_combination]: ./output_images/hls_gradient_magnitude_x_combination.jpg "Binary combination of HLS and x-oriented GRAY Gradient"
[original_vs_warped]: ./output_images/original_vs_warped.jpg "Warp Example"
[turn1_pipeline]: ./output_images/turn1_pipeline.jpg "Turn 1 Result "
[turn2_pipeline]: ./output_images/turn2_pipeline.jpg "Turn 2 Result"
[project_video_output]: https://youtu.be/s5t3WVToHsk "Outpout of Project Video"

Throughout this project we have used the `ConfigParser` module and the `config.ini` file to list all the possible configuration parameters. Also, it is worth noting that the project can be configured with either the `basic` or `advanced` lane finding algorithms. The former corresponds to the implementation of the https://github.com/pantelis/FindingLaneLines git repo. Similarly, there main algorithmic code is essentially contained in the helper files `helper_basic.py` and `helper_advanced.py` (this project). The `main.py` file essentially calls the image and video processing pipeline function.  

In the following sections we provide details and results for each step executed as outlined before. 

## Camera Calibration
The file `test-calibration.py` demonstrates the camera calibration procedure that was used. The implementation accomodates both 8x6 and 9x6 chessboard calibration images. 

Initially, the `test-finding-corners.py` file outlined bellow, uses the corresponding cv2 function to detect object points - the (x, y, z) coordinates of the chessboard corners in the world coordinates. A set of the expected `imgpoints` in image coordinates, the code stores these arrays in a dictionary for subsequent usage. 


```python
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
```

```python
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

```

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  

```python

    elif chessboard == '9x6':
        cal_directory = './camera_cal'
        test_directory = './test_images'
        # size of the chessboard in x and y directions
        nx = 9
        ny = 6
        # Test removal of distortion on this image
        img = cv2.imread(os.path.join(test_directory, 'straight_lines2.jpg'))

calibration_input_dict = pickle.load(open(os.path.join(cal_directory, 'calibration_input_dict.p'), 'rb'))
objpoints = calibration_input_dict["objpoints"]
imgpoints = calibration_input_dict["imgpoints"]

# Do camera calibration given object points and image points
undistorted, mtx, dist = helper_advanced.calibration_undistort(img, objpoints, imgpoints)

# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
calibration_output_dict = {}
calibration_output_dict["mtx"] = mtx
calibration_output_dict["dist"] = dist
pickle.dump(calibration_output_dict, open(os.path.join(cal_directory, 'calibration_output_dict.p'), "wb"))
```

As we dont need to calibrate the camera again, `main.py` branches off this step effectively using the calibration output dictionary when it is needed. At any opoint we can set the `Recalibration_flag` to repeat the calibration if needed.

```python

        if config.get('Calibration', 'Recalibration_flag') == 'True':

            # Given a set of chessboard images find the image points
            ret, objpoints, imgpoints = helper_advanced.generate_calibration_input(images, nx, ny)

        elif config.get('Calibration', 'Recalibration_flag') == 'False':

            # Read the calibration input data
            calibration_input_dict = pickle.load(open(os.path.join(cal_directory, 'calibration_input_dict.p'), 'rb'))
            objpoints = calibration_input_dict["objpoints"]
            imgpoints = calibration_input_dict["imgpoints"]
            
```

##Pipeline
The overall pipeline is shown below. All contained functions are in the `helper_advanced.py` file. 

```python

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
```
### Distortion-correction

I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![test_camera_calibration][test_camera_calibration]


### Color transforms and gradients 
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `test_.py`).  Here's an example of my output for this step using `test6.jpg` as input. 

![hls_gradient_magnitude_x_combination][hls_gradient_magnitude_x_combination]

### Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The portion of the pipeline for the perspective transform is repeated below. The function takes as inputs an image (`combined_binary`), as well as source (`src`) and destination (`dst`) points.  

```python
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
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![original_vs_warped][original_vs_warped]


### Lane identification 

```python

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
```
For lane identification the sliding window method was used after a thresholding the histogram of iluminated in the binary picture pixels. A 2nd order polynomial was used to fit the identified left and right lane pixels. Examples of lane identification are shown later in this write up. 


### Curvature and Offset

The curvature estimation is shown in the following function. The offset was covered by the code of the previous section. 

```python
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
```

### Reprojection back down onto the road of estimated lane lines
Configuring `config.ini` as follows:
```python
[Project]
# 'basic' or 'advanced'
processing = advanced
# test 'images' or 'video'
media_type = images
draw_flag = True

[Calibration]
# '8x6' is the chessboard type for the CarND-Camera-Calibration git repo
# '9x6' is the chessboard type for this git repo
chessboard=9x6
Recalibration_flag = False

```
and executing the `main.py` the following figures coverinbg the later stages of the pipeline can be observed. We provide results for two images that indicate the vehicle to be in a turn. 

![turn1_pipeline][turn1_pipeline]

![turn2_pipeline][turn2_pipeline]


---

#### Pipeline (video)

Here's a  link to the video 


[![project_video_output](https://img.youtube.com/vi/s5t3WVToHsk/0.jpg)](https://youtu.be/s5t3WVToHsk)

---

### Discussion

The pipeline adopted a conventional histogram-based sliding window detection approach. The detection, although correct in 95% of the frames, it does occasionally oscilate especially when a car next to the detected lane appears or lighting conditions change ubruptly. 

Below we provide three areas that we believe can improve performance:

1. *Smoothing*: A smoothing filter with input the pixel that we have detected the peak and start the sliding windows would certainly eliminate the occasiaonal oscilations. Effectively the information of previous *and* subsequent frames is used to maintain the right starting point. 
     
2. *Parameter Optimization*: The combination of HLS color coding and gradient information can be further imporved with further adjustments of the associated thresholds. Due to the lack of time we didnt have a chance to optimize these parameters. 
  
3. *Safeguards*: We notice that in most cases one of the two lanes is detected with almost 100% accuracy but the prior information of the other lane relative position was not used:  the other lane cant possibly be further than the average lane width for the US road network. 
