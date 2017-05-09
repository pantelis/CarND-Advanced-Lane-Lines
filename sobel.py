import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import helper

# Read in an image and grayscale it
image = mpimg.imread('examples/signs_vehicles_xygrad.png')

# Convert to grayscale - since img is read by mpimg we need to use RGB2GRAY
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# ------
# Get the identified lines via a Sobel thresholded gradient
grad_binary = helper.sobel_transform_threshold(gray, method='gradient-magnitude', orient='xy', sobel_kernel=3, mag_thresh=(30, 100))
# Plot the result
f, (f1, f2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
f1.imshow(image)
f1.set_title('Original Image', fontsize=30)
f2.imshow(grad_binary, cmap='gray')
f2.set_title('Thresholded Gradient', fontsize=30)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()

# ------
# Get the identified lines via a Sobel thresholded gradient direction
dir_binary = helper.sobel_transform_threshold(gray, method='gradient-angle', sobel_kernel=15, angle_thresh=(0.7, 1.3))
# Plot the result
g, (g1, g2) = plt.subplots(1, 2, figsize=(24, 9))
g.tight_layout()
g1.imshow(image)
g1.set_title('Original Image', fontsize=30)
g2.imshow(dir_binary, cmap='gray')
g2.set_title('Thresholded Grad. Dir.', fontsize=30)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()

# Get the identified lines via a Combination of the above methods
combined = np.zeros_like(dir_binary)
#combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
combined[(grad_binary == 1) | (dir_binary == 1)] = 1
# Plot the result
h, ((h1, h2,), (h3, h4)) = plt.subplots(2, 2, figsize=(24, 9))
h.tight_layout()
h1.imshow(image)
h1.set_title('Original', fontsize=30)
h2.imshow(grad_binary, cmap='gray')
h2.set_title('Grad Magnitude', fontsize=30)
h3.imshow(dir_binary, cmap='gray')
h3.set_title('Grad Angle', fontsize=30)
h4.imshow(combined, cmap='gray')
h4.set_title('Grad Mag + Angle', fontsize=30)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()