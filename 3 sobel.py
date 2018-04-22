import cv2

img = cv2.imread('src/1.jpg')
sobelx = cv2.Sobel(img, cv2.CV_64F, dx=1, dy=0)
sobely = cv2.Sobel(img, cv2.CV_64F, dx=0, dy=1)
sobelxy = cv2.Sobel(img, cv2.CV_64F, dx=1, dy=1)
laplacian = cv2.Laplacian(img, cv2.CV_64F)

cv2.imwrite('trgt/2 laplac.jpg', laplacian)
cv2.imwrite('trgt/2 sobel x.jpg', sobelx)
cv2.imwrite('trgt/2 sobel y.jpg', sobely)
cv2.imwrite('trgt/2 sobel xy.jpg', sobelxy)
