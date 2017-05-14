import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import helper_advanced
import os


# Read in an image, you can also try test1.jpg or test4.jpg
img = mpimg.imread('test_images/test6.jpg')

# Convert to HLS color space
hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

gradx_binary = helper_advanced.sobel_transform_threshold(gray, method='gradient-magnitude', sobel_kernel=3,
                                                orient='x', mag_thresh=(30, 100))
s_binary = helper_advanced.color_space_threshold(hls, channel='S', thresh=(170, 255))

# Stack each channel to view their individual contributions in green and blue respectively
# This returns a stack of the two binary images, whose components you can see as different colors
color_binary = np.stack((np.zeros_like(gradx_binary), gradx_binary, s_binary))

# Combine the two binary thresholds
combined_binary = np.zeros_like(gradx_binary)
combined_binary[(s_binary == 1) | (gradx_binary == 1)] = 1

# Plotting thresholded images
f, ((f1, f2), (f3, f4)) = plt.subplots(2, 2, figsize=(20, 10))
f1.set_title('Gradient x')
f1.imshow(gradx_binary, cmap='gray')
f2.set_title('S Channel')
f2.imshow(s_binary, cmap='gray')
f3.set_title('Combined S channel and gradient thresholds')
f3.imshow(combined_binary, cmap='gray')
plt.savefig(os.path.join('output_images', 'hls_gradient_magnitude_x_combination.jpg'))
plt.show()