import cv2
import numpy as np


def clean_mask(binary_mask, min_area=200, min_main_area_ratio=0.5):
    """
    Removes noise and keeps only the main connected component.
    min_area: small components below this area are removed.
    min_main_area_ratio: ignore components that are far too small relative to main body.
    """
    mask = binary_mask.astype(np.uint8)

    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:  # no object found
        return mask

    # Find the largest component (the real body)
    areas = stats[:, cv2.CC_STAT_AREA]
    largest = np.argmax(areas[1:]) + 1
    largest_area = areas[largest]

    # Clean small components
    cleaned = np.zeros_like(mask)
    for label in range(1, num_labels):
        area = areas[label]
        if area < min_area:
            continue
        if area < largest_area * min_main_area_ratio:
            continue
        cleaned[labels == label] = 1

    return cleaned.astype(bool)


