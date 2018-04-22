from matplotlib import pyplot
import cv2

img = cv2.imread('src/3.jpg')

pyplot.subplot(121), pyplot.imshow(img)
pyplot.title('Input'), pyplot.xticks([]), pyplot.yticks([])

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (3, 3), 0)
img = img.astype('float64')

laplacian = cv2.Laplacian(img, cv2.CV_64F)

rows, cols = img.shape
for i in range(rows):
    for j in range(cols):
        img[i, j] = max(min(img[i, j] - laplacian[i, j], 255), 0)

img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_GRAY2BGR)
cv2.imwrite('trgt/sharpen.jpg', img)

pyplot.subplot(122), pyplot.imshow(img)
pyplot.title('Result'), pyplot.xticks([]), pyplot.yticks([])

pyplot.show()
