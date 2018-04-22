import cv2
import numpy as np

img1 = cv2.imread('src/cat1.jpg')
img2 = cv2.imread('src/cat2.jpg')

DEPTH = 6

tmp = int(img1.shape[1] / 2)
ls = np.hstack((img1[:, 0: tmp], img2[:, tmp:]))
cv2.imwrite('trgt/not blending.jpg', ls)

G1 = img1.copy()
G2 = img2.copy()
gpA = [G1]
gpB = [G2]
for i in range(DEPTH):
    G1 = cv2.pyrDown(G1)
    G2 = cv2.pyrDown(G2)
    gpA.append(G1)
    gpB.append(G2)

lpA = [gpA[DEPTH - 1]]
lpB = [gpB[DEPTH - 1]]
for i in range(DEPTH - 1, 0, -1):
    GA = cv2.pyrUp(gpA[i])
    GB = cv2.pyrUp(gpB[i])
    L1 = cv2.subtract(gpA[i - 1], GA)
    L2 = cv2.subtract(gpB[i - 1], GB)
    lpA.append(L1)
    lpB.append(L2)

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
