import cv2

# 彩色图
# cv2.IMREAD_GRAYSCALE:灰度图
# cv2.IMREAD_UNCHANGED:带透明通道的图片
img = cv2.imread('lena.jpg', cv2.IMREAD_COLOR)
cv2.namedWindow('lena', cv2.WINDOW_NORMAL)
cv2.imshow('lena',img)
cv2.waitKey(0)
