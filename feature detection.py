import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('src/orange.jpg', 0)

# Initiate FAST object with default values
fast = cv2.FastFeatureDetector()

# find and draw the keypoints
kp = fast.detect(img, None)
