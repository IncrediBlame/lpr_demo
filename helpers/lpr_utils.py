import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

import os
from os import listdir
from os.path import isfile, join
from timeit import default_timer as timer
from typing import Any

# Ignore annoying certificate problem
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

try:
    from helpers.debug import *                    
except ImportError:
    from debug import *


def debug_show(img: np.ndarray, msg: str="Debug image", debug_lvl: int=1) -> None:
    """
    Shows image with message and waits for keypress, if appropriate debug level is enabled.
    Could be replaced with NOOP in production.
    """
    visible = False
    if DEBUG_LVL_1 and debug_lvl >= 1:
        visible = True
    if DEBUG_LVL_2 and debug_lvl >= 2:
        visible = True
    if DEBUG_LVL_3 and debug_lvl >= 3:
        visible = True
    if DEBUG_LVL_10 and debug_lvl >= 10:
        visible = True
    
    if visible:
        cv2.namedWindow(msg, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(msg, 864, 648)
        cv2.imshow(msg, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def debug_draw_rt_lines(img: np.ndarray, lines: np.ndarray, msg: str="Lines", debug_lvl: int=1) -> None:
    """
    Draws red edge lines on a copy of original image.
    For debug, could be replaced with NOOP in production.
    """
    debug_img = img.copy()
    for line in lines:
        rt_line(debug_img, line[0], (0, 0, 255))
    debug_show(debug_img, msg, debug_lvl)


def debug_shade_sheet(img: np.ndarray, mask: np.ndarray, debug_lvl: int=1) -> None:
    """
    Shade the band for debug display.
    For debug, could be replaced with NOOP in production.
    """
    shaded_sheet = img.copy()
    indices = mask == 255
    shaded_sheet[indices] = shaded_sheet[indices] * 0.75 + [0, 64, 0]
    debug_show(shaded_sheet, "Shaded middle sheet", debug_lvl)


def debug_shade_defects(img: np.ndarray, mask: np.ndarray, debug_lvl: int=1) -> None:
    """
    Shades band and defects on an image.
    For debug, could be replaced with NOOP in production.
    """
    shaded_img = img.copy()
    sheet_indices = mask == 1
    defects_indices = mask == 2
    shaded_img[sheet_indices] = shaded_img[sheet_indices] * 0.75 + [0, 64, 0]
    shaded_img[defects_indices] = shaded_img[defects_indices] * 0.75 + [0, 0, 64]
    debug_show(shaded_img, "Shaded sheet and defects", debug_lvl)


def read_imgs(path: str, file: str) -> np.ndarray:
    """
    Read img and mask from path and filename without extension
    """
    # Read image by trying different extensions
    
    for ext in ['png', 'jpg']:
        if isfile(path + file + "." + ext):
            img = cv2.imread(path + file + "." + ext)
            break
    
    debug_show(img, "Original image", 1)
    return img


def list_files(path: str) -> list[str]:
    files_only = [f for f in listdir(path) if isfile(join(path, f))]
    unique_files = files_only
    unique_files = set()
    for file in files_only:
        # Get rid of moronic ".DS.DS_Store" and such
        if file.split('.')[0] == "":
            continue
        name = file.split(".")[0]
        unique_files.add(name)
    unique_files = list(unique_files)
    unique_files.sort()

    return unique_files


def rt_line(img: np.ndarray, rt: np.ndarray, color: int | tuple) -> None:
    """
    Draws a line given with rho-theta parameters.
    """
    half_len = 1024

    a = np.cos(rt[1])
    b = np.sin(rt[1])
    p0 = np.array([rt[0] * a, rt[0] * b], dtype=np.float32)

    perp = np.array([-b, a], dtype=np.float32)
    p1 = (p0 + half_len * perp).astype(int)
    p2 = (p0 - half_len * perp).astype(int)
    cv2.line(img, p1, p2, color, 2)


def sample_error(img: np.ndarray, point: np.ndarray, direction: np.ndarray, num_samples: int) -> np.float32:
    """
    Steps num_samples times from point in along the direction and records pixel values from img.
    """
    # All calculations and inputs are in np-coordinates, not cv2-coordinates
    max_error = 1000.0
    shift = np.array([0, 30], dtype=np.float32)
    vec0 = np.empty((num_samples, 3), dtype=np.int32)
    vec1 = np.empty((num_samples, 3), dtype=np.int32)
    vec2 = np.empty((num_samples, 3), dtype=np.int32)
    cur_center = point
    cur_left = point - shift
    cur_right = point + shift
    for i in range(num_samples):
        cur_center += direction
        y = int(cur_center[0])
        x = int(cur_center[1])
        if not (0 <= x < img.shape[1]):
            return max_error
        vec0[i] = img[y, x]
        cur_left += direction
        y = int(cur_left[0])
        x = int(cur_left[1])
        if not (0 <= x < img.shape[1]):
            return max_error
        vec1[i] = img[y, x]
        cur_right += direction
        y = int(cur_right[0])
        x = int(cur_right[1])
        if not (0 <= x < img.shape[1]):
            return max_error
        vec2[i] = img[y, x]
    
    diff = np.linalg.norm(vec1 - vec2) + np.linalg.norm(vec2 - vec0) + np.linalg.norm(vec0 - vec1)
    # print(diff)
    return diff


def filter_middle(lines: np.ndarray, img: np.ndarray) -> np.ndarray:
    """
    For an array of lines filters out those not on the edge of the band.
    """
    # All calculations in np-coordinates, not cv2-coordinates
    num_samples = 20
    vec_threshold = 200
    good_indices = []
    for i in range(lines.shape[0]):
        line = lines[i, 0]
        rho = line[0]
        a = np.cos(line[1])
        b = np.sin(line[1])
        point = np.array([0, rho / a], dtype=np.float32)

        direction = np.array([a, -b], dtype=np.float32)
        direction *= img.shape[0] / (num_samples + 1) / a
        if sample_error(img, point, direction, num_samples) > vec_threshold:
            good_indices.append(i)
    
    return lines[good_indices]


def keep_best(lines: np.ndarray) -> np.ndarray:
    """
    Groups lines by rho values and only keeps num_keep with largest votes in each group
    """
    num_keep = 4
    good_indices = []
    groups_seen = []
    rho_threshold = 30
    for i in range(lines.shape[0]):
        rho = lines[i, 0, 0]
        found = False
        # Check if line belongs to one of the previously seen groups
        for j in range(len(groups_seen)):
            if abs(rho - groups_seen[j][0]) < rho_threshold:
                if groups_seen[j][1] < num_keep:
                    good_indices.append(i)
                    groups_seen[j][1] += 1
                found = True
                break
        # Create another group if not seen before
        if not found:
            groups_seen.append([rho, 1])
            good_indices.append(i)
    
    return lines[good_indices]


def get_lines(img: np.ndarray) -> np.ndarray:
    """
    Obtains vertical edges of the middle band.
    """
    # Detect lines in the left-most part of the image: 
    # no blur, more sensitive Canny, less Hough votes, less vertical lines, no middle-prunning
    left_cutoff = 150
    edges = cv2.Canny(img, 10, 25, apertureSize=3, L2gradient=False)
    debug_show(edges, "Left Canny edges", 1)

    # Clean the edges a little
    kernel = np.array([ [1],
                        [1],
                        [1]], dtype=np.uint8)
    erosion = cv2.erode(edges, kernel, iterations = 1)
    debug_show(erosion, "Erosion of horizontal edges", 1)

    left_lines = cv2.HoughLines(erosion, 1, np.pi / (180 * 4), 65, 
                            lines=None, srn=0, stn=0, min_theta=np.pi * 1 / 180, max_theta=np.pi * 8 / 180)
    # TODO Consider using accumulator version below
    # zzz = cv2.HoughLinesWithAccumulator(erosion, 1, np.pi / (180 * 4), 90, 
    #                         lines=None, srn=0, stn=0, min_theta=np.pi * 0 / 180, max_theta=np.pi * 8 / 180)
    if left_lines is None:
        raise Exception("No left-lines detected")
    left_lines = left_lines[left_lines[:, 0, 0] < left_cutoff]
    debug_draw_rt_lines(img, left_lines, "Hough left-lines before filtering", 1)
    left_lines = keep_best(left_lines)
    debug_draw_rt_lines(img, left_lines, "Hough left-lines after filtering", 1)

    # Now detect remaining lines: 
    # aggressive blur, picky Canny, ruthless Hough, more vertical lines, middle-lines pruning
    blur = cv2.GaussianBlur(img, (7, 7), 0)
    edges = cv2.Canny(blur, 10, 25, apertureSize=3, L2gradient=False)
    debug_show(edges, "Canny edges", 1)

    # Clean the edges a little
    kernel = np.array([ [1],
                        [1],
                        [1]], dtype=np.uint8)
    erosion = cv2.erode(edges, kernel, iterations = 1)
    debug_show(erosion, "Erosion of horizontal edges", 1)

    right_lines = cv2.HoughLines(erosion, 1, np.pi / (180 * 4), 80, 
                            lines=None, srn=0, stn=0, min_theta=np.pi * 0 / 180, max_theta=np.pi * 8 / 180)
    # TODO Consider using accumulator version below
    # zzz = cv2.HoughLinesWithAccumulator(erosion, 1, np.pi / (180 * 4), 90, 
    #                         lines=None, srn=0, stn=0, min_theta=np.pi * 0 / 180, max_theta=np.pi * 8 / 180)
    if not right_lines is None:
        right_lines = right_lines[right_lines[:, 0, 0] >= left_cutoff]
        
        debug_draw_rt_lines(img, right_lines, "Hough right-lines before filtering", 1)
        right_lines = keep_best(right_lines)
        right_lines = filter_middle(right_lines, img)
        debug_draw_rt_lines(img, right_lines, "Hough right-lines after filtering", 1)

        lines = np.vstack((left_lines, right_lines))
    else:
        lines = left_lines
    debug_draw_rt_lines(img, lines, "All Hough lines after filtering", 1)

    return lines


def get_band(img: np.ndarray, lines: np.ndarray) -> np.ndarray:
    """
    Construct a mask by looking at edge-lines. Uses floodfill as main method.
    """
    WIDTH_THRESHOLD = 500

    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for line in lines:
        rt_line(mask, line[0], 128)
    debug_show(mask, "Hough lines on a mask", 1)

    left_rho = np.min(lines[:, 0, 0])
    right_rho = np.max(lines[:, 0, 0])
    mid_rho = int((left_rho + right_rho) / 2)
    if right_rho - left_rho < WIDTH_THRESHOLD:
        if mid_rho < img.shape[1] / 2:
            mid_rho += 300
        else:
            mid_rho -= 300
    cv2.floodFill(mask, None, (mid_rho, 0), 255)

    return mask


def mask_glow(masked_img: np.ndarray) -> None:
    """
    Maks out bright glow spots on the image.
    """
    low_area_threshold = 80
    high_area_threshold = 20000
    largest_area = 0
    glow_theshold = 240
    while largest_area < high_area_threshold:
        glow_mask = np.zeros(masked_img.shape[: 2], dtype=np.uint8)
        indices = np.all(masked_img > glow_theshold, axis=2)
        glow_theshold -= 10
        if not np.any(indices):
            return
        glow_mask[indices] = 255
        debug_show(glow_mask, "Glow mask", 1)
        contours, _ = cv2.findContours(glow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        # Find 3 largest contours
        area_cnt = []
        largest_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            largest_area = max(area, largest_area)
            if area > low_area_threshold:
                area_cnt.append([area, cnt])
        
    area_cnt.sort(key=lambda x: x[0], reverse=True)
    for i in range(min(3, len(area_cnt))):
        dx, dy, x, y = cv2.fitLine(area_cnt[i][1], cv2.DIST_L2, 0, 0.01, 0.01)
        if np.abs(dx) > np.abs(dy):
            center = np.array([x[0], y[0]], dtype=np.float32)
            direction = 800 * np.array([dx[0], dy[0]], dtype=np.float32)
            p1 = (center + direction).astype(np.int32)
            p2 = (center - direction).astype(np.int32)
            cv2.line(masked_img, p1, p2, (255, 255, 255), 40)
    

def get_black_dots(masked_img: np.ndarray) -> np.ndarray:
    """
    Finds defects of black-dot type and creates a mask for them.
    """
    gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
    debug_show(gray, "Grayscale masked image", 2)
    # Grayscale closing to remove smallest black dots
    closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    debug_show(closing, "Grayscale closing", 2)
    blur = cv2.GaussianBlur(closing, (5, 5), 0)
    debug_show(blur, "Gaussian blur", 2)
    
    black_hat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)), 
                                        borderType=cv2.BORDER_CONSTANT, borderValue=0)
    _, black_hat = cv2.threshold(black_hat, 13, 255, cv2.THRESH_BINARY)
    black_hat = cv2.morphologyEx(black_hat, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)))
    debug_show(black_hat, "Black hats", 2)

    return black_hat


def get_white_dots(masked_img: np.ndarray) -> np.ndarray:
    """
    Finds defects of white-dot type and creates a mask for them.
    """
    gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
    debug_show(gray, "Grayscale masked image", 2)
    # Grayscale opening to remove smallest white dots
    opening = gray
    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    debug_show(opening, "Grayscale opening", 2)
    blur = opening
    blur = cv2.GaussianBlur(opening, (5, 5), 0)
    debug_show(blur, "Gaussian blur", 2)

    top_hat = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)), 
                                        borderType=cv2.BORDER_CONSTANT, borderValue=255)
    debug_show(top_hat)
    _, top_hat = cv2.threshold(top_hat, 13, 255, cv2.THRESH_BINARY)
    top_hat = cv2.morphologyEx(top_hat, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)))
    debug_show(top_hat, "Top hats", 2)

    return top_hat


def prep_for_inference(orig: np.ndarray, 
                    mean: torch.tensor, std: torch.tensor, DEVICE: str='cuda') -> list[Any]:
    """
    Reads an image and prepares it for inference.
    """
    new_shape = (960, 768)
    orig = cv2.resize(orig, new_shape)
    prep = orig.copy()
    # apply preprocessing
    prep = cv2.cvtColor(prep, cv2.COLOR_BGR2RGB)
    prep = torch.from_numpy(prep).to(DEVICE)
    prep = prep.float()
    prep = prep.permute(2, 0, 1)
    prep = prep.unsqueeze(0)
    prep /= 255.0
    prep -= mean
    prep /= std

    return orig, prep


def decode_file_name(file_name: str) -> list[Any]:
    """
    Decodes parameters from a checkpoint file_name.
    """
    args = file_name.split('_')
    model_name = args[2]
    if args[3] == 'se':
        encoder_name = f"{args[3]}_{args[4]}_{args[5]}"
    else:
        encoder_name = args[3]
    if args[-3][0] == 'a':
        augment = True if args[-3][4] == 'T' else False
        out_classes = int(args[-4][-1])
    else:
        augment = False
        out_classes = int(args[-3][-1])
    loss = args[-2]

    return model_name, encoder_name, augment, out_classes, loss
