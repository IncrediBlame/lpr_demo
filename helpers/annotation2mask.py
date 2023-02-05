import cv2
import numpy as np
import os
import json

try:
    from helpers.lpr_utils import *                    
except ImportError:
    from lpr_utils import *

root = "./train_set/"

json_path = os.path.join(root, "plates_dataset_full_1_json.json")
dataset_properties = json.load(open(json_path, 'r'))
missing_files = []
for entry in dataset_properties:
    # Prep mask and json
    image_properties = dataset_properties[entry]
    file_name = image_properties['filename']
    print(file_name)
    file_path = os.path.join(root, 'images', file_name)
    img = cv2.imread(file_path)
    if img is None:
        missing_files.append(file_name)
        continue
    mask = np.zeros(img.shape[: 2], dtype=np.uint8)

    # Read all regions from json
    category = 'Plates'
    regions = {category: []}
    for region in image_properties['regions']:
        regions[category].append(region['shape_attributes'])

    # Draw Sheet into mask first
    for region in regions['Plates']:
        assert region['name'] == 'polygon', f"Unknown region type: {region['name']}"
        x_coords = region['all_points_x']
        y_coords = region['all_points_y']
        contour = np.empty([len(x_coords), 2], dtype=np.int32)
        contour[:, 0] = x_coords
        contour[:, 1] = y_coords
        # print(contour)

        cv2.drawContours(mask, [contour], -1, 1, cv2.FILLED)

    debug_show(img, "Original image", 1)
    debug_shade_defects(img, mask, 1)

    cv2.imwrite(os.path.join(root, 'masks', file_name), mask)

print(f"Missing files: {missing_files}")
