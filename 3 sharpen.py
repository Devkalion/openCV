from matplotlib import pyplot as plt
import cv2

img = cv2.imread('src/3.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (3, 3), 0)
img = img.astype('float64')

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input'), plt.xticks([]), plt.yticks([])
laplacian = cv2.Laplacian(img, cv2.CV_64F)

# laplacian = cv2.convertScaleAbs(laplacian)
# plt.subplot(132), plt.imshow(laplacian, cmap='gray')
# plt.title('Filter'), plt.xticks([]), plt.yticks([])

img = cv2.subtract(img, laplacian)
cv2.imwrite('trgt/sharpen.jpg', img)

plt.subplot(122), plt.imshow(img, cmap='gray')
plt.title('Result'), plt.xticks([]), plt.yticks([])

plt.show()
