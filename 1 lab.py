import cv2
import numpy as np
from hist import draw_hist

img = cv2.imread('src/1.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)/255.0
img[:, :, 1:2] *= 1.1

img = np.array(img * 255.0, dtype='float64').astype('uint8')
draw_hist(img)
img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
cv2.imwrite('trgt/1 new.jpg', img)
