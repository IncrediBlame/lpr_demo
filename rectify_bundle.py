import cv2

from rectify_one import *


# Read unique filenames with extensions without mask, plate, frame, etc
path = "./plates/plates1/"
res_path = "./warped_digits/"
unique_files = list_files(path)

start_from = ""

num_plates = 0
for file in unique_files:
    # Skip up to some specific file
    if start_from != "":
        if file != start_from:
            continue
        else:
            start_from = ""
    print(file)

    # Read plates and their masks from the original
    img, mask = read_imgs(path, file)

    # Process images
    warped_list = rectify_one(img, mask)

    # Write rectified plates
    subimg_count = 0
    for warped in warped_list:
        cv2.imwrite(res_path + file + "_plate{}.png".format(subimg_count), warped)
        subimg_count += 1
    num_plates += subimg_count

        

