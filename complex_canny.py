# Applies Bilateral filtering, strict Canny parameters, and connected components 
import cv2
import numpy as np

def apply(img):
    # 1. Edge-preserving blur
    filtered = cv2.bilateralFilter(img, 9, 75, 75)
    
    # 2. Strict Canny parameters
    edges = cv2.Canny(filtered, 100, 200)
    
    # 3. Noise removal via Connected Components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges, connectivity=8)
    clean_edges = np.zeros_like(edges)
    
    min_pixel_area = 60
    
    # Loop thru to remove small components
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_pixel_area:
            clean_edges[labels == i] = 255
            
    return clean_edges