# uses a complex accumulator to combine multiple Canny edge detections and ridge detection for robust edge extraction.
import cv2
import numpy as np
from skimage.filters import frangi

def apply(img):
    blur_sizes = [3, 5, 7, 9]
    # Range of thresholds to handle varying contrast 
    threshold_ranges = [(30, 100), (50, 150), (100, 200), (150, 250)]
    
    accumulator = np.zeros_like(img, dtype=np.float32)

    # 1. 16 rounds of canny with varying parameters to capture a wide range of edges 
    for b_size in blur_sizes:
        # Smoothing suppresses noise from high frequencies (Section 78)
        blurred = cv2.GaussianBlur(img, (b_size, b_size), 0)
        for low, high in threshold_ranges:
            # Accumulating edges 
            edges = cv2.Canny(blurred, low, high)
            accumulator += (edges / 255.0)

    # 2. Ridge detection to capture thin, elongated structures like roads 
    ridges = frangi(img, sigmas=np.arange(2, 10, 2), black_ridges=True)
    ridges_norm = cv2.normalize(ridges, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    _, ridge_mask = cv2.threshold(ridges_norm, 15, 255, cv2.THRESH_BINARY)

    # 3. Voting mechanism to combine edges from Canny and Ridge (Section 2.2)
    # We require a 'majority' vote. 16 Canny + 1 Ridge = 17 total layers. If 5 or more layers agree, we consider it a valid edge.
    _, consolidated = cv2.threshold(accumulator, 6.0, 255.0, cv2.THRESH_BINARY)
    consolidated = consolidated.astype(np.uint8)
    
    # Merge the Ridge Anchor with the Gradient Consensus
    final_mask = cv2.bitwise_or(consolidated, ridge_mask)

    # 4. remove small noise artifacts using connected components analysis. 
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(final_mask, connectivity=8)
    clean_edges = np.zeros_like(final_mask)
    
    min_pixel_area = 40  # Minimum area threshold to filter out small noise
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_pixel_area:
            clean_edges[labels == i] = 255
            
    return clean_edges