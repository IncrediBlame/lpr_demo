import cv2
import os

images_dir = "./images/frames/with_defects/"
res_dir = "./detected_with_defects/images/"

file_list = os.listdir(images_dir)

for file in file_list:
    print(file[: -4])
    img = cv2.imread(os.path.join(images_dir, file))
    cv2.imwrite(os.path.join(res_dir, file[: -4]) + '.png', img)