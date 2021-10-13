import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np

# image = cv2.imread('lena.jpg')
# # 旋转的时候会发生剪切
# for angle in np.arange(0, 360, 45):
#     rotate = imutils.rotate(image, angle)
#     cv2.imshow('', rotate)
#     cv2.waitKey(0)
# # 旋转的时候不会发生剪切
# for angle in np.arange(0, 360, 45):
#     rotate = imutils.rotate_bound(image, angle)
#     cv2.imshow('', rotate)
#     cv2.waitKey(0)

# 自己写的图片旋转不剪切
img = cv2.imread('pill_01.png')
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
gray = cv2.GaussianBlur(gray, [3, 3], 0)
edge = cv2.Canny(gray, 15, 100)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, [3, 5])
edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel)
cnts = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
if len(cnts) > 0:
    c = max(cnts, key=lambda x: cv2.arcLength(x, False))
    mask = np.zeros(shape=gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [c], -1, 255, -1)
    x, y, w, h = cv2.boundingRect(c)
    imageRoi = img[y:y + h, x:x + w]
    maskRoi = mask[y:y + h, x:x + w]
    imageROI = cv2.bitwise_and(imageRoi, imageRoi,mask=maskRoi)
    cv2.imshow('',imageROI)
    cv2.waitKey(0)
