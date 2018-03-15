import cv2

img = cv2.imread('src/1.jpg')
img = cv2.GaussianBlur(img, (25, 25), 0)
cv2.imwrite('trgt/2 gauss.jpg', img)
