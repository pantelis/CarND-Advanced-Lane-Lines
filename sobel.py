import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

# Read in an image and grayscale it
image = mpimg.imread('examples/signs_vehicles_xygrad.png')


# Define a function that applies Sobel x or y,
# then takes an absolute value and applies a threshold.
# Note: calling your function with orient='x', thresh_min=5, thresh_max=100
# should produce output like the example image shown above this quiz.
def abs_sobel_thresh(img, orient='xy', mag_thresh = (0, 255)):

    # Convert to grayscale - since img is read by mpimg we need to use RGB2GRAY
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Take the derivative in x or y given orient = 'x' or 'y'
    thresh_max = max(mag_thresh)
    thresh_min = min(mag_thresh)
    if orient == 'x':
        sobel_direction = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        # Take the absolute value of the derivative or gradient
        abs_sobel = np.absolute(sobel_direction)
    elif orient == 'y':
        sobel_direction = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        # Take the absolute value of the derivative or gradient
        abs_sobel = np.absolute(sobel_direction)
    elif orient == 'xy':
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        abs_sobel = np.sqrt(sobel_x**2 + sobel_y**2)


    # Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    # Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    return sxbinary


# Run the function
grad_binary = abs_sobel_thresh(image, orient='xy', mag_thresh=(30, 100))

# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(grad_binary, cmap='gray')
ax2.set_title('Thresholded Gradient', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()