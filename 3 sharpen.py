from matplotlib import pyplot as plt
import cv2

img = cv2.imread('src/3.jpg')

plt.subplot(121), plt.imshow(img)
plt.title('Input'), plt.xticks([]), plt.yticks([])

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (3, 3), 0)
img = img.astype('float64')

laplacian = cv2.Laplacian(img, cv2.CV_64F)

# laplacian = cv2.convertScaleAbs(laplacian)
# plt.subplot(132), plt.imshow(laplacian, cmap='gray')
# plt.title('Filter'), plt.xticks([]), plt.yticks([])

# img = cv2.subtract(img, laplacian)

rows, cols = img.shape
for i in range(rows):
    for j in range(cols):
        img[i, j] = max(min(img[i, j] - laplacian[i, j], 255), 0)


img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_GRAY2BGR)
cv2.imwrite('trgt/sharpen.jpg', img)

plt.subplot(122), plt.imshow(img)
plt.title('Result'), plt.xticks([]), plt.yticks([])

plt.show()
