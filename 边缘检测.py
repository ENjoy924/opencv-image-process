import cv2
import matplotlib.pyplot as plt

img = cv2.imread('193.jpg', cv2.IMREAD_GRAYSCALE)
# Sobel检测算子
# gx = cv2.Sobel(img, -1, 1, 0)
# gy = cv2.Sobel(img, -1, 0, 1)
# sx = cv2.convertScaleAbs(gx)
# sy = cv2.convertScaleAbs(gy)
# sxy = cv2.add(sx,sy)
# plt.imshow(sxy, cmap=plt.cm.gray)
# plt.show()
# Candy检测算子
# gx = cv2.Canny(img, 5, 30)
# plt.imshow(gx, plt.cm.gray)
# plt.show()
# Lapance检测算子
# gx = cv2.Laplacian(img, -1,ksize=3)
# plt.imshow(gx,plt.cm.gray)
# plt.show()

ret, ths = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
cv2.imshow('',ths)
# gx = cv2.Canny(ths, 5, 10)
# cv2.imshow('',gx)
cv2.waitKey(0)
