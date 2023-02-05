from os import listdir
from os.path import isfile, join

from utils import *

path = "./warped_digits/"

files_only = [f for f in listdir(path) if isfile(join(path, f))]
unique_files = files_only
unique_files = set()
for file in files_only:
    # Get rid of moronic ".DS.DS_Store" and similar files
    if file == "combined.png" or file == "combined_originals.png":
        continue
    if file.split('.')[0] == "":
        continue
    unique_files.add(file)
unique_files = list(unique_files)
unique_files.sort()

warp_shape = np.array([128, 64], dtype=int)

# determine the dimensions of the combined image
img_per_width = int(len(unique_files) ** 0.5)
img_per_height = int(np.ceil(len(unique_files) / img_per_width))
combined = np.zeros([img_per_height * warp_shape[1], img_per_width * warp_shape[0], 3], dtype=np.uint8)

count = 0
for file in unique_files:
    print(file)

    img = cv2.imread(path + file)
    img = cv2.resize(img, (warp_shape[0], warp_shape[1]))
    x = count % img_per_width
    y = count // img_per_width
    combined[y * warp_shape[1]: (y + 1) * warp_shape[1], x * warp_shape[0]: (x + 1) * warp_shape[0]] = img

    count += 1

cv2.imwrite(path + "combined.png", combined)