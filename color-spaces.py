import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import helper

# Read in an image, you can also try test1.jpg or test4.jpg
image = mpimg.imread('test_images/test6.jpg')

# Convert to HLS color space
img_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

# Define a function that thresholds the S-channel of HLS
# Use exclusive lower bound (>) and inclusive upper (<=)
hls_binary = helper.color_space_threshold(img_hls, channel='S', thresh=(150, 255))

# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(hls_binary, cmap='gray')
ax2.set_title('Thresholded S', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()