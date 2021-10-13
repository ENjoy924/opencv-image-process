import cv2
import numpy as np


def callback(x):
    x = np.array([x], dtype=np.uint8).repeat(400, axis=0)[...,None]
    x = x.repeat(400, axis=1)
    img[:, :, 0] = x


img = np.zeros(shape=(400, 400, 3), dtype=np.uint8)
cv2.namedWindow('image')

cv2.createTrackbar('R', 'image', 0, 255, callback)

cv2.imshow('image', img)

cv2.waitKey(0)
