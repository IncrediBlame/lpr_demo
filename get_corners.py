from helpers.utils import *


def get_corners(img_slice: np.ndarray, mask_slice: np.ndarray, use_digits: bool = False) -> List[np.ndarray]:
    """
    Takes one img and mask slice and returns corners of the plate.
    Returns corners of the slice if no plate detected.
    """
    try:
        # Improve the mask by doing various filtering
        majority_hull, edge_hull, hull_mask, negative = enhance_mask(img_slice, mask_slice)
        # Get rough main directions for the hull
        rect, dirs = get_rect_directions(majority_hull)
        show_rect(hull_mask, rect, debug_lvl=1)
        show_dirs(hull_mask, rect, dirs, debug_lvl=1)
        
        if use_digits:
            # Get digits out of the image and make a convex hull out of them
            digits_mask = distill_digits(mask_slice, hull_mask, negative)
            # Clean up digit contours
            digits, digit_areas = get_contours(digits_mask, area_threshold=50.0)
            # Get a hull and main directions of contours
            hull = cv2.convexHull(np.vstack(digits))
            # Get rough main directions for the digits
            rect, dirs = get_rect_directions(hull)
            show_rect(digits_mask, rect, debug_lvl=1)
            show_dirs(digits_mask, rect, dirs, debug_lvl=1)
            # Some more contours cleaning
            digits, digit_areas = middle_contours(digits, digit_areas, dirs, digits_mask)

            # Do regression to find better left and right edge directions
            edge_dirs, reg_error = digits_regression(digits, digit_areas, digits_mask)
            log(f"Regression error: {reg_error}", debug_lvl=1)
            # Get a hull and main directions of contours
            hull = cv2.convexHull(np.vstack(digits))
            # Get rough main directions for the remaining digits
            rect, dirs = get_rect_directions(hull)
            show_rect(digits_mask, rect, debug_lvl=1)
            show_dirs(digits_mask, rect, dirs, debug_lvl=1)
            if reg_error < 1000:
                # If regression is consistent, replace edge directions
                dirs[2: ] = edge_dirs
            else:
                # If regression failed extract edge directions from majority_hull
                edge_dirs, _ = reduce_to_dirs(majority_hull, dirs, img_slice, hull_mask)
                dirs[2: ] = edge_dirs[2: ]
        else:
            # Use majority hull only
            hull = majority_hull
            edge_dirs, _ = reduce_to_dirs(majority_hull, dirs, img_slice, hull_mask)
            dirs[2: ] = edge_dirs[2: ]
            digits_mask = hull_mask
        
        # Reduce hull to corners
        corners, _ = reduce_to_corners(hull, edge_hull, dirs, img_slice, digits_mask, edge_dirs_flag=True)
        # Expand corners a little in all directions to compensate for warp's shrinking effect
        # 1.07 good for pure digits, 1.00 for edge_hull
        expand_factor = 1.07 if edge_hull is None else 1.00
        corners = expand_poly(corners, expand_factor)

    except Exception as instance:
        # Return corners of the original slice in case of error somewhere
        log(f"Exception: {instance}", debug_lvl=1)
        top_left = np.array([0.0, 0.0], dtype=np.float32)
        top_right = np.array([img_slice.shape[1]- 1.0, 0.0], dtype=np.float32)
        bottom_left = np.array([0.0, img_slice.shape[0]- 1.0], dtype=np.float32)
        bottom_right = np.array([img_slice.shape[1]- 1.0, img_slice.shape[0]- 1.0], dtype=np.float32)
        corners = np.array([top_left, top_right, bottom_left, bottom_right], dtype=np.float32)
        
    return corners