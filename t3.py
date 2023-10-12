import cv2
import numpy as np
import math

# Create a black image of size 600x401
image = np.zeros((401, 600), dtype=np.uint8)

# Calculate the radius for a circle with area of ~8000 pixels
# Using the formula for the area of a circle: area = pi * r^2
radius = int(np.sqrt(8000 / np.pi))

# Get the center of the image
center = (math.floor(image.shape[1] // 2 * 0.8), math.floor(image.shape[0] // 2 * 0.8))  # (width/2, height/2)

# Draw a white circle at the center of the image
cv2.circle(image, center, radius, (255), -1)  # -1 indicates fill the circle

# Save the image
cv2.imwrite('circle_mask.png', image)

# Optionally display the image
# cv2.imshow('Circle Image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
