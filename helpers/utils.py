import math
from unicodedata import digit
import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
from timeit import default_timer as timer
from typing import List, Tuple

try:
    from debug import *
except ImportError:
    from helpers.debug import *


class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.prev = None


@debug
def log(msg: str, debug_lvl: int=1):
    """
    Log a message.
    Debug enabled.
    """
    print(msg)


@debug
def show(img: np.ndarray, msg: str="Debug image", debug_lvl: int=1):
    """
    Shows resized window with a message.
    Controls debug levels.
    """
    cv2.namedWindow(msg, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(msg, 864, 648)
    cv2.imshow(msg, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


@debug
def show_submask_box(mask: np.ndarray, rect: np.ndarray, debug_lvl: int=1) -> None:
    """
    Draws and shows a box for a submask.
    Debug enabled.
    """
    debug_img = cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2BGR)
    cv2.rectangle(debug_img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 2)
    show(debug_img, "Submask box", debug_lvl=debug_lvl)


@debug
def show_majority_contours(clean_mask: np.ndarray, majority_cnt: np.ndarray, debug_lvl: int=1):
    """
    Shows separated majority contours.
    Debug enabled.
    """
    debug_img = cv2.cvtColor(clean_mask.copy(), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(debug_img, majority_cnt, -1, (0, 255, 0), cv2.FILLED)
    show(debug_img, "Majority contours", debug_lvl=debug_lvl)


@debug
def show_rect(mask: np.ndarray, rect: np.ndarray, debug_lvl: int=1) -> None:
    """
    Shows rectangle on a mask.
    Debug enabled.
    """
    debug_img = cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2BGR)
    for i in range(4):
        p1 = rect[i - 1].astype(int)
        p2 = rect[i].astype(int)
        cv2.line(debug_img, p1, p2, (0, 0, 255), 2)
    show(debug_img, "Min area rectangle", debug_lvl=debug_lvl)


@debug
def show_dirs(mask: np.ndarray, rect: np.ndarray, dirs: np.ndarray, debug_lvl: int=1) -> None:
    """
    Shows direction of the surrounding rectangle.
    Debug enabled.
    """
    debug_img = cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2BGR)
    center = (rect[0] + rect[2]) / 2
    p1 = (center + dirs[0] * 128).astype(int)
    p2 = (center + dirs[2] * 128).astype(int)
    center = center.astype(int)
    cv2.line(debug_img, center, p1, (0, 255, 0), 2)
    cv2.line(debug_img, center, p2, (0, 255, 0), 2)
    show(debug_img, "Min area rectangle direction", debug_lvl=debug_lvl)


@debug
def show_middle_contours(mask: np.ndarray, dirs: np.ndarray, 
                            top_boundary: np.float32, bottom_boundary: np.float32, 
                            middle_cnts: List[np.ndarray], discarded_cnts: List[np.ndarray], debug_lvl: int=1) -> None:
    """
    Shows filtered middle contours with boundary lines and discarded contours.
    Debug enabled.
    """
    tmp = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    # Display discarded contours
    cv2.drawContours(tmp, discarded_cnts, -1, (0, 0, 255), cv2.FILLED)
    # Dispay bounding lines
    p1 = (np.array([0, top_boundary], dtype=np.float32) + dirs[0] * 1024).astype(int)
    p2 = (p1 - dirs[0] * 2048).astype(int)
    cv2.line(tmp, p1, p2, (0, 0, 255), 2)
    p1 = (np.array([0, bottom_boundary], dtype=np.float32) + dirs[1] * 1024).astype(int)
    p2 = (p1 - dirs[1] * 2048).astype(int)
    cv2.line(tmp, p1, p2, (0, 0, 255), 2)
    cv2.drawContours(tmp, middle_cnts, -1, (255, 255, 255), cv2.FILLED)
    show(tmp, "Contours after middle-filter", debug_lvl=debug_lvl)


@debug
def show_regression(mask: np.ndarray, 
                    fit_lines: List[List[np.ndarray]], median_line: List[np.ndarray], good_lines: List[List[np.ndarray]],
                    left_centroid: np.ndarray, right_centroid: np.ndarray, edge_dirs: np.ndarray, debug_lvl: int=1) -> None:
    """
    Shows regression lines for digit contours.
    Debug enabled.
    """
    # Prepare canvas for debug imgs
    debug_img = cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2BGR)
    
    # Add fitLine to contour display
    for p1, p2 in fit_lines:
        cv2.line(debug_img, p1, p2, (0, 0, 255), 2)

    # Display median direction
    cv2.line(debug_img, median_line[0], median_line[1], (0, 255, 0), 2)

    # Color good lines
    for p1, p2 in good_lines:
        cv2.line(debug_img, p1, p2, (255, 0, 0), 2)

    # Display edge lines
    p1 = (left_centroid - 64 * edge_dirs[0]).astype(int)
    p2 = (left_centroid + 64 * edge_dirs[0]).astype(int)
    cv2.line(debug_img, p1, p2, (128, 0, 255), 2)
    p1 = (right_centroid - 64 * edge_dirs[1]).astype(int)
    p2 = (right_centroid + 64 * edge_dirs[1]).astype(int)
    cv2.line(debug_img, p1, p2, (128, 0, 255), 2)
    show(debug_img, "Inclination of digits", debug_lvl=debug_lvl)


@debug
def show_sides(img: np.ndarray, 
                top: List[np.ndarray], bottom: List[np.ndarray], left: List[np.ndarray], right: List[np.ndarray],
                debug_lvl: int=1) -> None:
    """
    Shows edges split into top, bottom, left, right groups.
    Debug enabled.
    """
    debug_img = img.copy()
    for line_set, color in [[top, (0, 0, 255)], [bottom, (0, 0, 128)], 
                                        [left, (0, 255, 0)], [right, (0, 128, 0)]]:
        for i in range(len(line_set)):
            p1, p2 = line_set[i]
            cv2.line(debug_img, p1, p2, color, 2)
    show(debug_img, "Split sides", debug_lvl=debug_lvl)


@debug
def show_lines(mask: np.ndarray, lines_ordered: List[List[np.ndarray]], debug_lvl: int=1) -> None:
    """
    Shows straight lines for edges.
    Debug enabled.
    """
    debug_img = cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2BGR)
    for p1, p2 in lines_ordered:
        cv2.line(debug_img, p1, p2, (0, 0, 255), 2)
    show(debug_img, "Edge points and dirs", debug_lvl=debug_lvl)


@debug
def show_corners(mask: np.ndarray, corners: List[np.ndarray], debug_lvl: int=1):
    """
    Shows corners on the mask.
    Debug enabled.
    """
    debug_img = cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2BGR)
    for point in corners:
        cv2.circle(debug_img, point.astype(np.int32), 1, (0, 255, 0), 2)
    show(debug_img, "Corners", debug_lvl=debug_lvl)


def parallel(angle1, angle2, eps):
    diff = abs(angle1 - angle2)
    if diff < eps or np.pi - diff < eps:
        return True
    
    return False


def lies_on(p1, p2, angle, new_p):
    max_dist = 10
    lam_limit = 2
    if abs(angle) < np.pi / 4:
        i = 0
    else:
        i = 1
    lam = (new_p[i] - p1[i]) / (p2[i] - p1[i])
    if lam > lam_limit + 1 or lam < -lam_limit:
        return False

    proj = p1[1 - i] + lam * (p2[1 - i] - p1[1 - i])
    
    if abs(new_p[1 - i] - proj) < max_dist:
        return True
    return False


def get_angle(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dx > 0:
        return math.atan2(dy, dx)
    return math.atan2(-dy, -dx)


def get_vec_angle(vec):
    dx = vec[0]
    dy = vec[1]
    if dx > 0:
        return math.atan2(dy, dx)
    return math.atan2(-dy, -dx)


def choose(p1, p2, angle, new_p1, new_p2):
    if abs(angle) < np.pi / 4:
        i = 0
    else:
        i = 1
    lam1 = (new_p1[i] - p1[i]) / (p2[i] - p1[i])
    lam2 = (new_p2[i] - p1[i]) / (p2[i] - p1[i])
    choices = [[0.0, p1], [1.0, p2], [lam1, new_p1], [lam2, new_p2]]
    p1[0], p1[1] = min(choices)[1]
    p2[0], p2[1] = max(choices)[1]


def intersect(l1: List[np.ndarray], l2: List[np.ndarray]) -> np.ndarray:
    """
    Intersects two lines given as pairs of points.
    """
    p1, p2 = l1
    p3, p4 = l2
    dir1 = p2 - p1
    dir2 = p4 - p3
    mu = p1[0] * dir1[1] - p1[1] * dir1[0] - p3[0] * dir1[1] + p3[1] * dir1[0]
    mu /= dir2[0] * dir1[1] - dir1[0] * dir2[1]
    return p3 + mu * dir2


def dist_sq(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + dy * dy


def dist(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return (dx * dx + dy * dy) ** 0.5


def order_lines(bundle, img=None):
    """
    Deprecated.
    """
    lines_by_dist = []
    for p1, p2 in bundle:
        lines_by_dist.append([dist_sq(p1, p2), p1, p2])
    lines_by_dist.sort(reverse=True, key=lambda x: x[0])

    lines_ordered = []
    _, p1, p2 = lines_by_dist[0]
    _, p3, p4 = lines_by_dist[1]
    # top first
    if min(p1[1], p2[1]) < min(p3[1], p4[1]):
        lines_ordered.append([p1, p2])
        lines_ordered.append([p3, p4])
    else:
        lines_ordered.append([p3, p4])
        lines_ordered.append([p1, p2])
    _, p5, p6 = lines_by_dist[2]
    _, p7, p8 = lines_by_dist[3]
    # left first
    if min(p5[0], p6[0]) < min(p7[0], p8[0]):
        lines_ordered.append([p5, p6])
        lines_ordered.append([p7, p8])
    else:
        lines_ordered.append([p7, p8])
        lines_ordered.append([p5, p6])
    
    if show_lines:
        for p1, p2 in lines_ordered:
            cv2.line(img, p1, p2, (0, 0, 255), 2)
        show(img)

    return lines_ordered


def calc_angle_median(angle_dist_list, threshold):
    partial_dist = 0
    for angle, seg in angle_dist_list:
        partial_dist += seg
        if partial_dist > threshold:
            break
    return angle


def calc_dist_median(dist_angle_list, low_thresh=2/5, high_thresh=3/5):
    partial_dist = 0
    weighted_angle = 0
    count_dist = 0
    count = False
    for seg, angle in dist_angle_list:
        partial_dist += seg
        if partial_dist > low_thresh:
            count = True
        if count:
            weighted_angle += angle * seg
            count_dist += seg
        if partial_dist > high_thresh:
            break
    return weighted_angle / count_dist


def calc_mode(dist_angle_list, threshold):
    partial_dist = 0
    mod_angle = 0
    for seg, angle in dist_angle_list:
        partial_dist += seg
        mod_angle += angle * seg
        if partial_dist > threshold:
            break
    mod_angle /= partial_dist
    return mod_angle


def calc_error(direction, side):
    # Get a perpendicular unit vector to the direction
    unit_perp = np.array([direction[1], -direction[0]], dtype=np.float32)
    unit_perp /= np.linalg.norm(unit_perp)
    total_error = 0
    # Add up absolute perpendicular components of each side vector
    for p1, p2 in side:
        total_error += abs(np.dot(unit_perp, p2 - p1))
    
    return total_error


def vector_sum(side):
    """
    Calculates vector sum along the side
    """
    vector_sum = np.zeros([2, ], dtype=np.float32)
    for p1, p2 in side:
        vector_sum += p2 - p1
    return vector_sum


def median_vector_sum(side, threshold=0.2):
    """
    Calculates vector sum of the median segment of a side, ignoring edge segments.
    Works best for top and bottom segments.
    Good combination for digits regression on left and right sides.
    """
    # Determine the total length of a side
    dist_vec = []
    total_len = 0
    for p1, p2 in side:
        vec = p2 - p1
        dist = np.linalg.norm(vec)
        dist_vec.append([dist, vec])
        total_len += dist
        
    
    # Add to vector sum only between thresholds
    low_threshold = threshold * total_len
    high_threshold = total_len - low_threshold
    vector_sum = np.zeros([2, ], dtype=float)
    cur_len = 0
    for dist, vec in dist_vec:
        cur_len += dist
        if cur_len > low_threshold:
            vector_sum += vec
        if cur_len > high_threshold:
            break
    
    return vector_sum




def process_edge(side, median_func):
    # Edge case
    if len(side) == 0:
        raise Exception("Side len is 0")

    direction = median_func(side)
    
    # Calculate error along the side
    return direction, calc_error(direction, side)


def edge_out(sides):
    dirs = np.ones([4, 2], dtype=np.float32)
    total_error = 0
    for i in range(len(sides)):
        direction, error = process_edge(sides[i], median_vector_sum)
        dirs[i] = direction
        total_error += error
    
    # Normalize dirs
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    
    return dirs, total_error


def fast_bounding_boxes(mask):
    """
    Scales the mask down to compute contours and their bounding boxes.
    Returns bounding box rects with their areas scaled back to original size.
    """
    scaling_factor = max(mask.shape[1] // 324, 1)
    area_threshold = 300 / scaling_factor**2

    lower_res = cv2.resize(mask, (mask.shape[1] // scaling_factor, mask.shape[0] // scaling_factor))
    contours, _ = cv2.findContours(lower_res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    box_area = []
    for cnt in (contours):
        area = cv2.contourArea(cnt)
        if area > area_threshold:
            rect = np.array(cv2.boundingRect(cnt), dtype=int)
            # Enlarge rect to compensate for loss of resolution
            rect[: 2] -= 1
            rect[2: ] += 2
            box_area.append([rect * scaling_factor, area * scaling_factor**2])
    
    return box_area


def read_imgs(path, file):
    """
    Read img and mask from path and filename without extension
    """
    # Read image by trying different extensions
    img = None
    for ext in ['png', 'jpg']:
        if isfile(path + file + "_frame." + ext):
            img = cv2.imread(path + file + "_frame." + ext)
            break
    show(img, "Original image", debug_lvl=1)
    
    # Read mask as GRAY
    mask = None
    for ext in ['png', 'jpg']:
        if isfile(path + file + "_mask." + ext):
            mask = cv2.imread(path + file + "_mask." + ext, cv2.IMREAD_GRAYSCALE)
            break
    # Display mask if true
    show(mask, "Original mask", debug_lvl=1)
    
    return img, mask


def slice_imgs(img: np.ndarray, mask: np.ndarray) -> List[List[Any]]:
    """
    Slice img and mask into subimages.
    Then resizes them to approximately fit [256, 128] shape.
    """
    # Get subimage and submask for every large-enough mask blob
    slices_list = []
    for rect, area in fast_bounding_boxes(mask):
        # Draw bounding box for each valid contour
        show_submask_box(mask, rect, debug_lvl=1)

        # Copy and enlarge bounding box part of the image
        # Enlargement is done because all further parameters were tuned for about 256*128 size plates
        scaling_factor = int((256 * 128 / area)**0.5) + 1
        img_slice = cv2.resize(img[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]], 
                            (rect[2] * scaling_factor, rect[3] * scaling_factor))
        mask_slice = cv2.resize(mask[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]],
                            (rect[2] * scaling_factor, rect[3] * scaling_factor))
        
        # Hard treshold sumask
        mask_slice[mask_slice < 255] = 0
        
        # Clear out parts of other masks
        contours, _ = cv2.findContours(mask_slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_area = 0
        largest_cnt = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > largest_area:
                largest_area = area
                largest_cnt = cnt
        mask_slice = np.zeros(mask_slice.shape, dtype=np.uint8)
        cv2.drawContours(mask_slice, [largest_cnt], -1, 255, cv2.FILLED)

        # Display subimage and submask
        show(img_slice, "Image slice", debug_lvl=1)
        show(mask_slice, "Mask slice", debug_lvl=1)

        # Append a pair of cropped img and its mask
        slices_list.append([img_slice, mask_slice, rect[: 2], scaling_factor])
    
    return slices_list


def digits_regression(digits, digit_areas, mask):
    """
    Calculates the inclination for every digit.
    Then does regression to predict inclinations for edge contours.
    """
    log(f"digits_regression()", debug_lvl=1)
    
    # Threshold for filtering angles out when compared with median direction
    cos_threshold = 0.9
    tan_threshold = 0.001
    tan_max = 1000.0
    # Collect data of each contour inclination
    xy_vals = np.zeros([len(digits), 2], dtype=np.float32)
    dxdy_vals = np.zeros([len(digits), 2], dtype=np.float32)
    tan_vals = np.zeros([len(digits), ], dtype=np.float32)
    area_vals = np.zeros([len(digits), ], dtype=np.float32)
    left_centroid = np.array([mask.shape[1], 0], dtype=np.float32)
    right_centroid = np.array([0, 0], dtype=np.float32)
    fit_lines = []
    for i in range(len(digits)):
        cnt = digits[i]
        # fitLine obtains the same direction as PCA
        dx, dy, x, y = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
        # Make values into scalars for new np version
        dx = dx[0]
        dy = dy[0]
        x = x[0]
        y = y[0]
        xy_vals[i] = [x, y]
        dxdy_vals[i] = [dx, dy]
        # Record tan as tan_max if contour's direction is horizontal
        tan_vals[i] = dx / dy if abs(dy) > tan_threshold else tan_max
        area_vals[i] = digit_areas[i]
        # Update left and right centroids
        if x < left_centroid[0]:
            left_centroid[: ] = [x, y]
        if x > right_centroid[0]:
            right_centroid[: ] = [x, y]
        # For debug display
        # TODO: find a way to remove
        p1 = (int(x - dx * 64), int(y - dy * 64))
        p2 = (int(x + dx * 64), int(y + dy * 64))
        fit_lines.append([p1, p2])
        
    # Find median direction
    median_dir = np.array([np.median(tan_vals), 1.0], dtype=np.float32)
    median_dir /= np.linalg.norm(median_dir)
    # For debug display
    # TODO: find a way to remove
    center = np.array([mask.shape[1], mask.shape[0]], dtype=np.float32) / 2
    p1 = (center + median_dir * 64).astype(int)
    p2 = (center - median_dir * 64).astype(int)
    median_line = [p1, p2]
        

    good_indices = []
    good_lines = []
    for i in range(len(digits)):
        # Only consider lines more or less parallel to median dir, using cos of the angle between them
        if abs(np.dot(median_dir, dxdy_vals[i])) > cos_threshold:
            good_indices.append(i)
            # For debug display
            # TODO: find a way to remove
            p1 = (xy_vals[i] + dxdy_vals[i] * 64).astype(int)
            p2 = (xy_vals[i] - dxdy_vals[i] * 64).astype(int)
            good_lines.append([p1, p2])
                
    # Check if anything left
    if len(good_indices) == 0:
        return None, 10000

    # Do weighted linear regression, with x-coordinate as input, areas as weight, tans as predicted variable
    result = np.polyfit(xy_vals[good_indices, 0], tan_vals[good_indices], 1, w=area_vals[good_indices], full=True)
    coefs = result[0]
    if len(good_indices) < 3:
        error = 10000
    else:
        error = np.sqrt(np.sum((area_vals * (np.polyval(coefs, xy_vals[:, 0]) - tan_vals))**2) / len(digits))

    # Calculate predicted inclination dirs for left and right contours
    edge_dirs = np.zeros([2, 2], dtype=np.float32)
    edge_dirs[0] = [coefs[0] + coefs[1] * left_centroid[0], 1.0]
    edge_dirs[1] = [coefs[0] + coefs[1] * right_centroid[0], 1.0]

    # Normalizing
    edge_dirs /= np.linalg.norm(edge_dirs, axis=1, keepdims=True)

    # Replacing extreme values with median, if needed
    # Needed when digits are not splitting nicely
    edge_dirs[abs(np.dot(edge_dirs, median_dir)) < cos_threshold] = median_dir
    # Change right direction to the opposite
    edge_dirs[1] *= -1

    show_regression(mask, fit_lines, median_line, good_lines, left_centroid, right_centroid, edge_dirs, debug_lvl=2)
    
    return edge_dirs, error


def expand_poly(corners, factor):
    """
    Make a factor enlargement of a polygon relative to its centeroid
    """
    center = np.average(corners, axis=0)
    corners = center + factor * (corners - center)
    return corners


def warp(corners, shape=[128, 64], padding=False):
    """
    Calculates perspective transformation matrix from given 4 points into a given rectangular shape.
    If padding is True, adds same shape-sized padding from each side.
    Expect corners in order top-left, top-right, bottom-left, bottom-right.
    """
    warp_width, warp_height = shape

    top = 0
    left = 0
    if padding:
        top = warp_height
        left = warp_width
    input_pts = corners
    bottom = top + warp_height
    right = left + warp_width
    output_pts = np.float32([[left, top], [right, top], [left, bottom], [right, bottom]])
    matrix = cv2.getPerspectiveTransform(input_pts, output_pts)
    return matrix
    

def intersect_to_corners(lines_ordered: List[List[np.ndarray]], mask=None) -> List[np.ndarray]:
    """
    Returns corners by finding intersections between lines.
    Should be top-left, top-right, bottom-left, bottom-right, if lines are provided in the correct order.
    """
    corners = []
    corners.append(intersect(lines_ordered[0], lines_ordered[2]))
    corners.append(intersect(lines_ordered[0], lines_ordered[3]))
    corners.append(intersect(lines_ordered[1], lines_ordered[2]))
    corners.append(intersect(lines_ordered[1], lines_ordered[3]))
    
    show_corners(mask, corners, debug_lvl=2)
    return np.array(corners, dtype=np.float32)


def get_rect_directions(hull: np.ndarray):
    """
    Obtain main directions of a hull as unit vector.
    Horizontal returned first, followed by vertical.
    """
    log(f"get_rect_directions()", debug_lvl=1)

    # Min area rectangle
    # TODO: Try PCA to obtain main directions, could be faster and give better approximation
    rect = cv2.boxPoints(cv2.minAreaRect(hull))
    
    dirs = np.zeros([4, 2], dtype=np.float32)
    # Determine the longest and shortest sides
    vec1 = rect[1] - rect[0]
    vec2 = rect[2] - rect[1]
    len1 = np.linalg.norm(vec1)
    len2 = np.linalg.norm(vec2)
    # Normalize
    vec1 /= len1
    vec2 /= len2
    if len1 > len2:
        dirs[0] = vec1
        dirs[2] = vec2
    else:
        dirs[0] = vec2
        dirs[2] = vec1
    dirs[1] = -dirs[0]
    dirs[3] = -dirs[2]
    
    return rect, dirs


def rectanglify(img, mask, show_lines=False, show_corners=False, shape=[256, 128]):
    """
    Deprecated.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_area = 0
    largest_cnt = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > largest_area:
            largest_area = area
            largest_cnt = cnt

    # Min area rectangle
    rect = cv2.boxPoints(cv2.minAreaRect(largest_cnt)).astype(int)
    
    bundle = []
    p1, p2, p3, p4 = list(rect)
    bundle.append([p1, p2])
    bundle.append([p2, p3])
    bundle.append([p3, p4])
    bundle.append([p4, p1])

    tmp = None
    if show_lines or show_corners:
        tmp = img.copy()
    lines_ordered = order_lines(bundle, img=tmp, show_lines=show_lines)
    corners = intersect_to_corners(lines_ordered, img=tmp, show_corners=show_corners)

    return warp(corners, shape=shape, padding=True)


def distill_digits(mask: np.ndarray, hull_mask: np.ndarray, negative: np.ndarray) -> np.ndarray:
    """
    Obtain digit contours, using different available masks.
    """
    log(f"distill_digits()", debug_lvl=1)

    # Get the positive to obtain digits
    digits = np.bitwise_xor(negative, hull_mask)
    show(digits, "XOR with hull mask", debug_lvl=2)

    # Refine digits using original mask
    digits[mask == 0] = 0
    show(digits, "Digits refined with original mask", debug_lvl=2)

    # Improve digits separation with erosion
    digits = cv2.erode(digits, np.ones((3, 3), np.uint8), iterations=2)
    show(digits, "Digits separated with erosion", debug_lvl=2)

    # Clear out some of the residue
    digits = cv2.morphologyEx(digits, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    show(digits, "Digits cleaned with morph", debug_lvl=2)
    
    return digits


def get_contours(mask: np.ndarray, area_threshold: np.float32=50.0) -> List[List[Any]]:
    """
    Get contours and cleans them up by removing ones below area_threshold.
    """
    log(f"get_contours()", debug_lvl=1)

    # Get contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by area
    large_cnts = []
    cnt_areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Keep only large enough contours
        if area > area_threshold:
            large_cnts.append(cnt)
            cnt_areas.append(area)
    # Check if anything left
    if len(large_cnts) == 0:
        raise Exception("No contours left after size-filtering")
    
    return large_cnts, cnt_areas


def middle_contours(contours: np.ndarray, cnt_areas: List[np.float32], dirs: np.ndarray, mask: np.ndarray) -> List[List[Any]]:
    """
    Clean up contours by removing ones with centroid in the top and bottom 10% along the given directions.
    """
    log(f"middle_contours()", debug_lvl=1)

    # Check if top or bottom direction is close to vertical
    if abs(dirs[0, 0]) < 0.001 or abs(dirs[1, 0]) < 0.001:
        raise Exception("Top or bottom dirs are vertical")
    
    # Find top and bottom boundaries
    cnt_centers = []
    top_boundary = 1000000
    bottom_boundary = -1000000
    # Calculate tans of top and bottom directions
    tan_top = dirs[0, 1] / dirs[0, 0]
    tan_bottom = dirs[1, 1] / dirs[1, 0]
    for cnt in contours:
        # Find top and bottom points and a center for each contour
        top_idx = np.argmin(cnt[:, 0, 1])
        bottom_idx = np.argmax(cnt[:, 0, 1])
        top_point = cnt[top_idx, 0]
        bottom_point = cnt[bottom_idx, 0]
        cnt_centers.append((top_point + bottom_point) / 2)

        # Project top and bottom points onto y-axis along given top and bottom directions
        top_boundary = min(top_boundary, top_point[1] - tan_top * top_point[0])
        bottom_boundary = max(bottom_boundary, bottom_point[1] - tan_bottom * bottom_point[0])
    
    # Filter out top-most and bottom-most horizontal blobs as well
    middle_cnts = []
    middle_areas = []
    discarded_cnts = []
    for i in range(len(cnt_centers)):
        x, y = cnt_centers[i]
        top_proj = top_boundary + tan_top * x
        bottom_proj = bottom_boundary + tan_bottom * x
        vert_threshold = 0.1 * (bottom_proj - top_proj)
        # Check if contour's approximate centroid is too high or too low
        if top_proj + vert_threshold < y < bottom_proj - vert_threshold:
            middle_cnts.append(contours[i])
            middle_areas.append(cnt_areas[i])
        else:
            discarded_cnts.append(contours[i])
                
    # Check if anything left
    if len(middle_cnts) == 0:
        raise Exception("No contours left after middle-filtering")
    show_middle_contours(mask, dirs, top_boundary, bottom_boundary, middle_cnts, discarded_cnts, debug_lvl=2)

    return middle_cnts, middle_areas


def enhance_mask(img: np.ndarray, mask: np.ndarray) -> List[np.ndarray]:
    """
    Clean up mask by combining thresholding, erosion, etc.
    """
    # Display current image
    show(img, "Image", debug_lvl=1)
    show(mask, "Hard thresholded mask", debug_lvl=1)
    # Make a gray image for the future
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    # Adaptive thresholding to select digits
    negative = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 0)
    show(negative, "Adaptive thresholding on gray to get negative", debug_lvl=2)

    # Do normal thresholding to mostly get rid of black values outside of plate
    _, threshold = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    # Improve the negative
    negative[threshold == 0] = 0
    show(negative, "Negative refined with normal thresholding", debug_lvl=2)

    clean_mask = mask.copy()
    clean_mask[negative == 0] = 0
    show(clean_mask, "Mask refined with negative", debug_lvl=2)
    
    # Take remaining contours
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Order contours by area
    largest_cnts = []
    total_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        total_area += area
        largest_cnts.append((area, cnt))
    largest_cnts.sort(key=lambda x: x[0], reverse=True)
    # Select several contours that form the majority
    majority_contrib_threshold = 0.05 * total_area
    abs_majority_contrib_threshold = 0.01 * total_area
    majority_cnt = []
    abs_majority_cnt = []
    for area, cnt in largest_cnts:
        # Check if next candidate component is too small
        if area < abs_majority_contrib_threshold:
            break
        # Append contour to majority
        if area > majority_contrib_threshold:
            majority_cnt.append(cnt)
        # Append contour to absolute majority
        abs_majority_cnt.append(cnt)
    show_majority_contours(clean_mask, majority_cnt, debug_lvl=2)
    show_majority_contours(clean_mask, abs_majority_cnt, debug_lvl=1)
        
    
    # Form absolute majority hull for later
    abs_majority_hull = cv2.convexHull(np.vstack(abs_majority_cnt))

    hull_mask = np.zeros(mask.shape, dtype=np.uint8)
    # Fill lacunas in tresholded mask with convex hull
    unified_cnt = np.vstack(majority_cnt)
    hull = cv2.convexHull(unified_cnt)
    cv2.drawContours(hull_mask, [hull], -1, 255, cv2.FILLED)
    show(hull_mask, "Hull mask", debug_lvl=2)

    # Refine negative with hull mask as well
    clean_mask[hull_mask == 0] = 0
    show(clean_mask, "Mask refined with hull mask", debug_lvl=2)
    
    return hull, abs_majority_hull, hull_mask, clean_mask


def split_sides(hull: np.ndarray, dirs: np.ndarray, img: np.ndarray) -> List[List[np.ndarray]]:
    top = []
    bottom = []
    left = []
    right = []
    # Form unified horizontal and vertical directions
    horiz_dir = dirs[0] - dirs[1]
    vert_dir = dirs[2]- dirs[3]
    for i in range(len(hull)):
        p1 = hull[i - 1][0]
        p2 = hull[i][0]
        vec = p2 - p1
        proj1 = np.abs(np.dot(horiz_dir, vec))
        proj2 = np.abs(np.dot(vert_dir, vec))
        # TODO: More flat lines for top edge, since it is usually better preserved
        if proj1 > proj2:
            # Use hulls clockwise orientation to differentiate between top/bottom
            if vec[0] > 0:
                top.append([p1, p2])
            else:
                bottom.append([p1, p2])
        # Everything else on the left or on the right, again using clockwise orientation
        elif vec[1] < 0:
            left.append([p1, p2])
        else:
            right.append([p1, p2])
    
    show_sides(img, top, bottom, left, right, debug_lvl=2)
    return [top, bottom, left, right]


def list_files(path):
    files_only = [f for f in listdir(path) if isfile(join(path, f))]
    unique_files = files_only
    unique_files = set()
    for file in files_only:
        # Get rid of moronic ".DS.DS_Store"
        if file.split('.')[0] == "":
            continue
        parts = file.split('_')
        parts.pop()
        name = "_".join(parts)
        unique_files.add(name)
    unique_files = list(unique_files)
    unique_files.sort()

    return unique_files


def reduce_to_corners(hull: np.ndarray, edge_hull: np.ndarray, dirs: np.ndarray, 
                        img: np.ndarray, mask: np.ndarray, edge_dirs_flag: bool=False) -> Any:
    """
    Reduces a given hull to 4 corner points
    """
    # Check if there is actual hull
    if hull is None:
        raise Exception("No hull was provided for reduce")
    
    # Separate top, bottom, left, right sides of the hull by looking at the directions provided
    sides = split_sides(hull, dirs, img=img)
    
    # Average across each side to get a straigh edge
    if edge_dirs_flag:
        num_sides = 2
    else:
        num_sides = 4
    new_dirs, error = edge_out(sides[: num_sides])
    # Update dirs
    dirs[: num_sides] = new_dirs[: num_sides]
    
    # Find edge points, using obtained directions and edge_hull from the original mask, if provided
    if edge_hull is not None:
        sides = split_sides(edge_hull, dirs, img=img)
    edge_points = get_edge_points(sides, dirs)

    # Find lines along each edge
    lines_ordered = []
    for i in range(4):
        if edge_points[i] is None:
            raise Exception("No edge point was found")
        p1 = (edge_points[i]).astype(int)
        p2 = (edge_points[i] + dirs[i] * 128).astype(int)
        lines_ordered.append([p1, p2])
    # Display edge points on the same image as contours, if true
    show_lines(mask, lines_ordered, debug_lvl=2)

    # Find intersection points of those edges and return
    return intersect_to_corners(lines_ordered, mask=mask), error


def reduce_to_dirs(hull: np.ndarray, dirs: np.ndarray, img: np.ndarray, mask: np.ndarray):
    """
    Reduces a given hull to 4 directions
    """
    # Check if there is actual hull
    if hull is None:
        raise Exception("No hull was provided for reduce")
    
    # Separate top, bottom, left, right sides of the hull by looking at the directions provided
    sides = split_sides(hull, dirs, img=img)
    
    # Average across each side to get a straight edge
    new_dirs, error = edge_out(sides)
    
    # Find edge points, using obtained directions
    edge_points = get_edge_points(sides, new_dirs)

    # Find lines along each edge
    lines_ordered = []
    for i in range(4):
        if edge_points[i] is None:
            raise Exception("No edge point was found")
        p1 = (edge_points[i]).astype(int)
        p2 = (edge_points[i] + new_dirs[i] * 128).astype(int)
        lines_ordered.append([p1, p2])
    show_lines(mask, lines_ordered, debug_lvl=2)
    
    
    return new_dirs, error


def get_edge_points(sides, dirs):
    """
    Slides along the side to find the edge point along the given direction.
    """
     # Find top point along dirs[0]
    edge_points =[]
    
    # Top point calculation
    # Check if direction is close to vertical
    if abs(dirs[0, 0]) < 0.001:
        edge_points.append(sides[0][0][0])
    else:
        top_point = None
        top_proj = 1000000
        # Calculate tan of the direction
        tan_top = dirs[0, 1] / dirs[0, 0]
        for p1, _ in sides[0]:
            # Project the point onto y-axis along given direction
            proj = p1[1] - tan_top * p1[0]
            if proj < top_proj:
                top_point = p1
                top_proj = proj
        edge_points.append(top_point)
    
    # Bottom point calculation
    # Check if direction is close to vertical
    if abs(dirs[1, 0]) < 0.001:
        edge_points.append(sides[1][0][0])
    else:
        bottom_point = None
        bottom_proj = -1000000
        # Calculate tan of the direction
        tan_bottom = dirs[1, 1] / dirs[1, 0]
        for p1, _ in sides[1]:
            # Project the point onto y-axis along given direction
            proj = p1[1] - tan_bottom * p1[0]
            if proj > bottom_proj:
                bottom_point = p1
                bottom_proj = proj
        edge_points.append(bottom_point)

    # Left point calculation
    # Check if direction is close to horizontal
    if abs(dirs[2, 1]) < 0.001:
        edge_points.append(sides[2][0][0])
    else:
        left_point = None
        left_proj = 1000000
        # Calculate tan of the direction
        cotan_left = dirs[2, 0] / dirs[2, 1]
        for p1, _ in sides[2]:
            # Project the point onto y-axis along given direction
            proj = p1[0] - cotan_left * p1[1]
            if proj < left_proj:
                left_point = p1
                left_proj = proj
        edge_points.append(left_point)
    
    # Right point calculation
    # Check if direction is close to horizontal
    if abs(dirs[3, 1]) < 0.001:
        edge_points.append(sides[3][0][0])
    else:
        right_point = None
        right_proj = -1000000
        # Calculate tan of the direction
        cotan_right = dirs[3, 0] / dirs[3, 1]
        for p1, _ in sides[3]:
            # Project the point onto y-axis along given direction
            proj = p1[0] - cotan_right * p1[1]
            if proj > right_proj:
                right_point = p1
                right_proj = proj
        edge_points.append(right_point)
    
    return edge_points


