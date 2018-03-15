import cv2
import numpy as np

img1 = cv2.imread('src/cat1.jpg')
img2 = cv2.imread('src/cat2.jpg')

DEPTH = 6

tmp = int(img1.shape[1] / 2)
ls = np.hstack((img1[:, 0: tmp], img2[:, tmp:]))
cv2.imwrite('trgt/not blending.jpg', ls)

G = img1.copy()
gpA = [G]
for i in range(DEPTH):
    G = cv2.pyrDown(G)
    gpA.append(G)

G = img2.copy()
gpB = [G]
for i in range(DEPTH):
    G = cv2.pyrDown(G)
    gpB.append(G)

lpA = [gpA[DEPTH - 1]]
for i in range(DEPTH - 1, 0, -1):
    GE = cv2.pyrUp(gpA[i])
    L = cv2.subtract(gpA[i - 1], GE)
    lpA.append(L)

lpB = [gpB[DEPTH - 1]]
for i in range(DEPTH - 1, 0, -1):
    GE = cv2.pyrUp(gpB[i])
    L = cv2.subtract(gpB[i - 1], GE)
    lpB.append(L)

LS = []
for la, lb in zip(lpA, lpB):
    rows, cols, dpt = la.shape
    tmp = int(cols / 2)
    ls = np.hstack((la[:, 0: tmp], lb[:, tmp:]))
    LS.append(ls)

ls_ = LS[0]
for i in range(1, DEPTH):
    ls_ = cv2.pyrUp(ls_)
    ls_ = cv2.add(ls_, LS[i])

cv2.imwrite('trgt/blending.jpg', ls_)

