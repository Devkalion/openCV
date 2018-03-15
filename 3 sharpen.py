import cv2

img = cv2.imread('src/1.jpg')
laplacian = cv2.Laplacian(img, cv2.CV_64F)
img = cv2.add(img, laplacian)
cv2.imwrite('trgt/sharpen.jpg', img)
