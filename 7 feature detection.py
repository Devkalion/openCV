import cv2
from matplotlib import pyplot

FEATURE_COUNTS = 30

image1 = cv2.imread('src/f1.jpg')
image2 = cv2.imread('src/f2.jpg')
pyplot.imshow(image1)
pyplot.imshow(image2)

orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(image1, None)
kp2, des2 = orb.detectAndCompute(image2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)[:FEATURE_COUNTS]

image3 = cv2.drawMatches(image1, kp1, image2, kp2, matches, None, flags=2)
image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)

pyplot.imshow(image3)
pyplot.show()
