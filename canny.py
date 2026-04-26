# simple canny edge detection pipeline with standard Gaussian blur and Canny parameters from openCV 
import cv2

def apply(img):
    blurred = cv2.GaussianBlur(img, (7, 7), 0)
    edges = cv2.Canny(blurred, 100, 200)
    
    return edges