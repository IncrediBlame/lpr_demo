import cv2

from os import listdir
from os.path import isfile, join

try:
    from helpers.lpr_utils import *                    
except ImportError:
    from lpr_utils import *

path = "../train_set/images/"
full_path = join(os.path.dirname(__file__), path)

files_only = [file for file in listdir(full_path) 
                            if isfile(join(full_path, file))]
resolutions = {}
new_low_height = 10000
new_high_height = 0
for file in files_only:
    image = cv2.imread(join(full_path, file))
    res = image.shape
    if res not in resolutions:
        resolutions[res] = []
    resolutions[res].append(file)
    
    # Calculate new resolution
    new_width = 1024
    factor = new_width / image.shape[1]
    new_height = factor * image.shape[0]
    new_height = int(np.ceil(new_height / 32) * 32)
    new_low_height = min(new_low_height, new_height)
    new_high_height = max(new_high_height, new_height)

print("Original resolutions:")
print(resolutions.keys())

print("Low resolution frames:")
for res in resolutions:
    if res[0] < 300 or res[1] < 300:
        print(res, resolutions[res])

print(f"New resolutions: (1024, {new_low_height}) - (1024, {new_high_height})")
