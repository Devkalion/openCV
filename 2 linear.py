import cv2
import numpy as np
from hist import draw_hist

img = cv2.imread('src/2.jpg') * 1.0
alpha, betta = 2, 1.0
img[:, :, :] *= alpha
img[:, :, :] += betta

img = np.array(img, dtype='float64').astype('uint8')
cv2.imwrite('trgt/2 new.jpg', img)

draw_hist(img)
