import cv2
import numpy as np

# source: https://en.wikipedia.org/wiki/Kernel_(image_processing)
def sharpen(img):
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    return cv2.filter2D(img, -1, kernel=kernel)

def negative(img):
    return 1-img