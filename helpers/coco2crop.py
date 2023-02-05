import cv2
import numpy as np
import os
import json

try:
    from helpers.lpr_utils import *                    
except ImportError:
    from lpr_utils import *

root = "./train_set/"
output_root = "./crop_set/"

json_path = os.path.join(root, "plates_dataset_full_coco_6_2.json")

# Fix numpy seed
np.random.seed(0)

# Create dict for every image indexed by its id, containing file_name and segmentations list
dataset_properties = json.load(open(json_path, "r"))
files_dict = {}
for image_properties in dataset_properties["images"]:
    img_name = image_properties["file_name"]
    image_id = image_properties["id"]
    files_dict[image_id] = {"file_name": img_name, "segmentations": []}

# Populate segmentations list
for seg_properties in dataset_properties["annotations"]:
    assert seg_properties["category_id"] == 1, "Wrong category for annotation"
    image_id = seg_properties["image_id"]
    polygon = seg_properties["segmentation"]
    assert len(polygon) == 1,  "Segmentation list contains multiple elements"
    files_dict[image_id]["segmentations"].append(polygon[0])

# Load images and draw masks
for image_id in files_dict:
    img_name = files_dict[image_id]["file_name"]
    print(img_name)
    file_path = os.path.join(root, "images", img_name)
    img = cv2.imread(file_path)

    # Read mask
    if img_name[-4] == ".":
        no_ext = img_name[: -4]
        ext = img_name[-4: ]
    elif img_name[-5] == ".":
        no_ext = img_name[: -5]
        ext = img_name[-5: ]
    else:
        raise Exception("Strange file ext")
    new_file_name = f"{no_ext}.png"
    mask_path = os.path.join(root, "masks", new_file_name)
    # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = np.zeros(img.shape[: 2], dtype=np.uint8)
    debug_show(img, "Original image", 1)
    debug_shade_defects(img, mask, 2)

    # Draw Sheet into mask first
    count = 0
    for polygon in files_dict[image_id]["segmentations"]:
        x_coords = [polygon[i] for i in range(0, len(polygon), 2)]
        y_coords = [polygon[i] for i in range(1, len(polygon), 2)]
        contour = np.empty([len(x_coords), 2], dtype=np.int32)
        contour[:, 0] = x_coords
        contour[:, 1] = y_coords
        
        # Get bounding rect
        left, top, width, height = cv2.boundingRect(contour)
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
        pivot = np.array([[left, top]], dtype = np.float32)

        # Crop slices out
        img_slice = img[top: bottom, left: right]
        mask_slice = np.zeros(img_slice.shape[: 2], dtype=np.uint8)
        contour -= pivot.astype(np.int32)
        cv2.fillPoly(mask_slice, [contour.astype(np.int32)], 1)
        debug_shade_defects(img_slice, mask_slice, 2)

        # resize image and mask
        # width and height must be divisible by 32
        new_width = 256
        factor = new_width / img_slice.shape[1]
        new_height = factor * img_slice.shape[0]
        new_height = int(np.ceil(new_height / 32) * 32)
        new_shape = (new_width, new_height)
        new_shape = (192, 192)
        resized_slice = cv2.resize(img_slice, new_shape)

        # Tweak contour coordinates
        contour = np.empty([len(x_coords), 2], dtype=np.float32)
        contour[:, 0] = x_coords
        contour[:, 1] = y_coords
        contour -= pivot
        contour[:, 0] *= resized_slice.shape[1] / img_slice.shape[1]
        contour[:, 1] *= resized_slice.shape[0] / img_slice.shape[0]
        
        # Create a mask
        mask_slice = np.zeros(resized_slice.shape[: 2], dtype=np.uint8)
        cv2.fillPoly(mask_slice, [contour.astype(np.int32)], 1)
        debug_shade_defects(resized_slice, mask_slice, 2)

        if img_name[-4] == ".":
            ext = img_name[-4: ]
            no_ext = img_name[: -4]
        elif img_name[-5] == ".":
            ext = img_name[-5: ]
            no_ext = img_name[: -5]
        else:
            raise Exception("Extension should only contain 3 or 4 chars")
        new_file_name = f"{no_ext}_plate{count}.png"
        cv2.imwrite(os.path.join(output_root, "images", new_file_name), resized_slice)
        cv2.imwrite(os.path.join(output_root, "masks", new_file_name), mask_slice)
        count += 1

