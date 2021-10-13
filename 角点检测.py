import cv2
import numpy as np

# img = cv2.imread('1102.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
# corner = cv2.cornerHarris(gray, 2, 3, 0.04)
# img[corner > 0.001 * corner.max()] = [0, 0, 255]
# cv2.imshow('', img)
# cv2.waitKey(0)

img = cv2.imread('1102.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
corner = cv2.goodFeaturesToTrack(gray, 10000, 0.01, 10)
for i in corner:
    x, y = i[0]
    cv2.circle(img, (int(x), int(y)), 1, (0, 0, 255))
cv2.imshow('',img)
cv2.waitKey(0)
