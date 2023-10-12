import cv2
import numpy as np

# Load the source image
source_image = cv2.imread('img/archive2/6/mask.png')

# Create a mask where all channels are 255 (indicating white regions)
white_mask = np.all(source_image == [255, 255, 255], axis=-1)

# Count the white pixels
number_of_white_pixels = np.sum(white_mask)

# Convert the boolean mask to a uint8 mask with values 255 for white and 0 for non-white
output_mask = np.where(white_mask, 255, 0).astype(np.uint8)

# Save the mask image
cv2.imwrite('mask_image.png', output_mask)

print(f"Number of white pixels: {number_of_white_pixels}")

# Optional: To visualize the mask
# cv2.imshow('Mask', output_mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

