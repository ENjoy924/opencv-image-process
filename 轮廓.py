import cv2
import matplotlib.pyplot as plt

img = cv2.imread('handwriting.jpg', cv2.IMREAD_GRAYSCALE)
ret, ths = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

# cv2.RETR_TREE,查找方法，cv2.CHAIN_APPROX_SIMPLE：轮廓的近似方法
# image:原图像,
# contours:轮廓
# hierarchy是轮廓间的层级关系
contours, hierarchy = cv2.findContours(ths, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# 表示找到几个轮廓
print(len(contours))
cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
plt.imshow(img,'gray')
plt.show()
