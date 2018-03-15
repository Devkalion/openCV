import cv2
import numpy as np

img = cv2.imread('src/1.jpg')
imgx = cv2.Sobel(img, cv2.CV_64F, dx=1, dy=0)
imgy = cv2.Sobel(img, cv2.CV_64F, dx=0, dy=1)
imgxy = cv2.Sobel(img, cv2.CV_64F, dx=1, dy=1)
laplacian = cv2.Laplacian(img, cv2.CV_64F)
#img = np.array(img, dtype='float64').astype('uint8')
cv2.imwrite('trgt/2 laplac.jpg', imgx)
cv2.imwrite('trgt/2 sobel x.jpg', imgx)
cv2.imwrite('trgt/2 sobel y.jpg', imgy)
cv2.imwrite('trgt/2 sobel xy.jpg', imgxy)
