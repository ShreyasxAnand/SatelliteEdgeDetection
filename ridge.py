# Uses frangi filter to detect ridges 
import cv2
import numpy as np
from skimage.filters import frangi

def apply(img):
    # Gaussian blur to reduce high-frequency texture noise
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Strict Parameter Tuning:
    # sigmas: Search only for structures between 1 and 4 pixels wide.
    # alpha: Deviation of plate-like structures. Lowering this makes it stricter.
    # beta: Deviation of blob-like structures. Lowering this penalizes houses/trees.
    ridges = frangi(
        blurred, 
        sigmas=np.arange(1, 5, 1), 
        alpha=0.3, 
        beta=0.3, 
        black_ridges=True
    )
    
    # Convert ridges to 8 bit image
    ridges_normalized = cv2.normalize(ridges, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Use threshold to remove weak ridges and retain strong ridge structures.
    _, final_roads = cv2.threshold(ridges_normalized, 30, 255, cv2.THRESH_BINARY)
    
    return final_roads