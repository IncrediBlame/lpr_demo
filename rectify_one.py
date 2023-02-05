from get_corners import *


def rectify_one(img: np.ndarray, mask: np.ndarray, warp_shape: np.ndarray=np.array([128, 64], dtype=int)) -> List[np.ndarray]:
    """
    Takes img and mask and produces a list of rectified plates.
    """
    slices = slice_imgs(img, mask)
    warped_list = []
    for img_slice, mask_slice, origin, scaling_factor in slices:
        corners = get_corners(img_slice, mask_slice)

        # Convert to original image coordinates
        corners /= scaling_factor
        corners += origin

        # Obtain perspective matrix
        matrix = warp(corners, shape=warp_shape, padding=False)
        
        # Warp the original image and its mask using combined transform
        warped = cv2.warpPerspective(img, matrix, warp_shape, flags=cv2.INTER_LINEAR)

        # Save results
        show(warped, "Warped plate", debug_lvl=3)
        warped_list.append(warped)

    return warped_list