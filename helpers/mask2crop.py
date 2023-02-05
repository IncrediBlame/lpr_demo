import numpy as np
import cv2
import os
import json

try:
    from helpers.lpr_utils import *                    
except ImportError:
    from lpr_utils import *


input_root = "./train_set/"
output_root = "./crop_set/"
img_list = os.listdir(os.path.join(input_root, "images"))
img_list.sort()
mask_list = os.listdir(os.path.join(input_root, "masks"))
mask_list.sort()

# Load images and crop masks
for i in range(len(img_list)):
    img_name = img_list[i]
    print(img_name)
    img_path = os.path.join(input_root, "images", img_name)
    img = cv2.imread(img_path)
    mask_name = mask_list[i]
    mask_path = os.path.join(input_root, "masks", mask_name)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    debug_show(img, "Original image", 1)
    debug_shade_defects(img, mask, 2)

    # Split into contours and crop
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    for cnt in contours:
        # Get bounding rect
        left, top, width, height = cv2.boundingRect(cnt)
        if width < 5 or height < 5:
            # print(width, height)
            continue
        right = left + width
        bottom = top + height

        # Wobble rect by random amount
        pad_factor = 5
        width_pad = max(1, width // pad_factor)
        height_pad = max(1, height // pad_factor)
        left_pad = np.random.randint(width_pad)
        right_pad = np.random.randint(width_pad)
        top_pad = np.random.randint(height_pad)
        bottom_pad = np.random.randint(height_pad)
        left = max(0, left - left_pad)
        right = min(img.shape[1] - 1, right + right_pad)
        top = max(0, top - top_pad)
        bottom = min(img.shape[0] - 1, bottom + bottom_pad)

        # Crop slices out
        img_slice = img[top: bottom, left: right]
        mask_slice = mask[top: bottom, left: right]
        debug_shade_defects(img_slice, mask_slice, 2)
        debug_show(mask_slice.astype(np.float32), "Mask slice", 1)

        # resize image and mask
        # width and height must be divisible by 32
        new_width = 256
        factor = new_width / img_slice.shape[1]
        new_height = factor * img_slice.shape[0]
        new_height = int(np.ceil(new_height / 32) * 32)
        new_shape = (new_width, new_height)
        img_slice = cv2.resize(img_slice, new_shape)
        mask_slice = cv2.resize(mask_slice, new_shape)
        debug_shade_defects(img_slice, mask_slice, 3)

        if img_name[-4] == ".":
            ext = img_name[-4: ]
            no_ext = img_name[: -4]
        elif img_name[-5] == ".":
            ext = img_name[-5: ]
            no_ext = img_name[: -5]
        else:
            raise Exception("Extension should only contain 3 or 4 chars")
        new_file_name = f"{no_ext}_plate{count}.png"
        cv2.imwrite(os.path.join(output_root, "images", new_file_name), img_slice)
        cv2.imwrite(os.path.join(output_root, "masks", new_file_name), mask_slice)
        count += 1
