from matplotlib import pyplot
import cv2
from sys import argv


def draw_hist(img):
    color = ('r', 'g', 'b')
    for i, col in enumerate(color):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        pyplot.plot(hist, color=col)
        pyplot.xlim([0, 256])
    pyplot.show()


if __name__ == "__main__":
    draw_hist(cv2.imread(argv[1]))
