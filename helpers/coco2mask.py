import cv2
import numpy as np
import os
import json

try:
    from helpers.lpr_utils import *                    
except ImportError:
    from lpr_utils import *

root = "./train_set/"

json_path = os.path.join(root, "plates_dataset_full_coco_6_2.json")

# Create dict for every image indexed by its id, containing file_name and segmentations list
dataset_properties = json.load(open(json_path, "r"))
files_dict = {}
for image_properties in dataset_properties["images"]:
    # Prep mask and json
    file_name = image_properties["file_name"]
    image_id = image_properties["id"]
    files_dict[image_id] = {"file_name": file_name, "segmentations": []}

# Populate segmentations list
for seg_properties in dataset_properties["annotations"]:
    assert seg_properties["category_id"] == 1, "Wrong category for annotation"
    image_id = seg_properties["image_id"]
    polygon = seg_properties["segmentation"]
    assert len(polygon) == 1,  "Segmentation list contains multiple elements"
    files_dict[image_id]["segmentations"].append(polygon[0])

# Load images and draw masks
missing_files = []
for image_id in files_dict:
    file_name = files_dict[image_id]["file_name"]
    # if file_name != "gh987diyrjfh.webp":
    #     continue
    print(file_name)
    file_path = os.path.join(root, "images", file_name)
    img = cv2.imread(file_path)
    if img is None:
        missing_files.append(file_name)
        continue
    mask = np.zeros(img.shape[: 2], dtype=np.uint8)

    # Draw Sheet into mask first
    for polygon in files_dict[image_id]["segmentations"]:
        x_coords = [polygon[i] for i in range(0, len(polygon), 2)]
        y_coords = [polygon[i] for i in range(1, len(polygon), 2)]
        contour = np.empty([len(x_coords), 2], dtype=np.int32)
        contour[:, 0] = x_coords
        contour[:, 1] = y_coords
        
        cv2.fillPoly(mask, [contour], 1)

    debug_show(img, "Original image", 1)
    debug_shade_defects(img, mask, 1)

    if file_name[-4] == ".":
        no_ext = file_name[: -4]
        ext = file_name[-4: ]
    elif file_name[-5] == ".":
        no_ext = file_name[: -5]
        ext = file_name[-5: ]
    else:
        raise Exception("Strange file ext")
    new_file_name = f"{no_ext}.png"
    cv2.imwrite(os.path.join(root, "masks", new_file_name), mask)

print(f"Missing files: {missing_files}")
