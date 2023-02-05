import cv2
from os import listdir
from os.path import isfile, join

path = "./plates2/"
res_path = "./plates2/"

files_only = [f for f in listdir(path) if isfile(join(path, f))]
unique_files = files_only
unique_files = set()
for file in files_only:
    name = file.split('_')[0]
    if name == ".DS.DS_Store":
        continue
    if name == "":
        continue
    if name == "masked" or name == "mask":
        unique_files.add(file.split('_')[1])
    else:
        ext = file.split('.')[-1]
        if ext == "":
            continue
        unique_files.add(name + '.' + ext)
unique_files = list(unique_files)
unique_files.sort()

for file in unique_files:
    print(file)
    name = file[: -4]
    ext = file[-3: ]
    img = cv2.imread(path + "masked_" + name + "." + ext)
    cv2.imwrite(res_path + name + "_frame." + ext, img)
    mask = cv2.imread(path + "mask_" + name + "." + ext)
    cv2.imwrite(res_path + name + "_mask." + ext, mask)