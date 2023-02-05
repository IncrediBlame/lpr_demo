import cv2
import numpy as np
import os
import json

try:
    from helpers.lpr_utils import *                    
except ImportError:
    from lpr_utils import *

root = "./train_set/"
file_list = os.listdir(os.path.join(root, "images"))
file_list.sort()

# Load images and draw masks
missing_files = []
for file_name in file_list:
    print(file_name)
    file_path = os.path.join(root, "images", file_name)
    img = cv2.imread(file_path)
    if img is None:
        missing_files.append(file_name)
        continue
    mask = np.zeros(img.shape[: 2], dtype=np.uint8)

    # Open segmask and convert to mask
    no_ext = file_name[: -3] if file_name[-4] == "." else file_name[: -4]
    file_name_png = no_ext + "png"
    seg_mask_path = os.path.join(root, "defaultannot", file_name_png)
    seg_mask = cv2.imread(seg_mask_path)
    if seg_mask is not None:
        mask[np.any(seg_mask != 0, axis = 2)] = 1
    else:
        missing_files.append(file_name)

    debug_show(img, "Original image", 1)
    debug_shade_defects(img, mask, 1)

    cv2.imwrite(os.path.join(root, "masks", file_name_png), mask)

print(f"Missing files: {missing_files}")
