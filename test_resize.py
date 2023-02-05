import cv2
from rectify_one import *

img = cv2.imread('1_frame.png')
mask = cv2.imread('1_mask.png', cv2.IMREAD_GRAYSCALE)

# Resizing
# area = img.shape[0] * img.shape[1]
# scaling_factor = int((256 * 128 / area)**0.5) + 1
# print(scaling_factor)
# new_shape = (img.shape[1] * scaling_factor, img.shape[0] * scaling_factor)
# new_shape = (256, 128)
# img_slice = cv2.resize(img, new_shape)
# mask_slice = cv2.resize(mask, new_shape)

scaling_factor = 1.0
img_slice = img
mask_slice = mask

# Get corners and resize back
corners = get_corners(img_slice, mask_slice)
print(f'corners: {corners}')
corners /= scaling_factor

warp_shape = [128, 64]
matrix = warp(corners, shape=warp_shape, padding=False)
        
# Warp the original image and its mask using combined transform
rectified_plate = cv2.warpPerspective(img, matrix, warp_shape, flags=cv2.INTER_LINEAR)

# Save results
show(rectified_plate, "Warped plate", debug_lvl=3)
cv2.imwrite(f'plate.png', rectified_plate)