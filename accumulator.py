# Use accumulation of edges from multiple Canny configurations to enhance edge detection results.
import cv2
import numpy as np

 # 16 rounds of canny with varying parameters to capture a wide range of edges
def apply(img):
    accumulator = np.zeros_like(img, dtype=np.float32)
    
    blur_sizes = [3, 5, 7, 9]
    thresholds = [(50, 150), (100, 200), (150, 250), (200, 300)]
    
    for b_size in blur_sizes:
        blurred = cv2.GaussianBlur(img, (b_size, b_size), 0)
        for low, high in thresholds:
            edges = cv2.Canny(blurred, low, high)
            accumulator += (edges / 255.0)
            
    _, final_edges = cv2.threshold(accumulator, 2.0, 255.0, cv2.THRESH_BINARY)
    
    final_edges = final_edges.astype(np.uint8)
    
    return final_edges