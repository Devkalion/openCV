from matplotlib import pyplot as plt
import cv2
from sys import argv


def draw_hist(img):
    color = ('r', 'g', 'b')
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show()


if __name__ == "__main__":
    draw_hist(cv2.imread(argv[1]))
