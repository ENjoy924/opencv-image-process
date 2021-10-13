import cv2
import imutils
from imutils.perspective import four_point_transform
import numpy as np
from matplotlib import pyplot as plt

DIGITAL_MAP = {
    (1, 1, 1, 0, 1, 1, 1): '0',
    (0, 1, 0, 0, 1, 0, 0): '1',
    (1, 0, 1, 1, 1, 0, 1): '2',
    (1, 0, 1, 1, 0, 1, 1): '3',
    (0, 1, 1, 1, 0, 1, 0): '4',
    (1, 1, 0, 1, 0, 1, 1): '5',
    (1, 1, 0, 1, 1, 1, 1): '6',
    (1, 0, 1, 0, 0, 1, 0): '7',
    (1, 1, 1, 1, 1, 1, 1): '8',
    (1, 1, 1, 1, 0, 1, 1): '9'
}
img = cv2.imread('digit.jpg', cv2.IMREAD_GRAYSCALE)
blur = cv2.GaussianBlur(img, [3, 3], 0)
edge = cv2.Canny(blur, 100, 200)
cnts = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
digits_display = None
for c in cnts:
    L = cv2.arcLength(c, True)
    points = cv2.approxPolyDP(c, 0.02 * L, True)
    x, y, w, h = cv2.boundingRect(c)
    if len(points) == 4 and w > 10 and w < 300:
        digits_display = points

roiImage = four_point_transform(blur, digits_display.reshape(4, 2))
thresh = cv2.threshold(roiImage, 50, 255, cv2.THRESH_BINARY_INV)[1]

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, [3, 5])
morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

plt.imshow(morph,'gray')
plt.show()

cnts = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
digits = []
for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    if w > 20 and (h >= 30 and h <= 100):
        cv2.drawContours(roiImage, [c], -1, (255, 255, 255), 3)
        plt.imshow(roiImage, 'gray')
        plt.show()
