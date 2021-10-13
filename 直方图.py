import cv2
from matplotlib import pyplot as plt
import numpy as np

img = cv2.imread('color.png')
# 计算并显示直方图
# 图片，哪个通道，掩膜，只计算掩膜内的，分段数，范围
hist = cv2.calcHist(img, [0], None, [256], [0, 256])
# plt.plot(hist)
# plt.show()
# 形成掩膜
mask = np.zeros(shape=img.shape[:2], dtype=np.uint8)
mask[100:300, 200:400] = 1
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 获得掩膜后的图像
mask_img = cv2.bitwise_and(img_gray, img_gray, mask=mask)
# 计算直方图，这里输入必须是灰度图
hist = cv2.calcHist(img_gray, [0], mask, [256], [0, 256])

# 直方图均衡化
hist_img = cv2.equalizeHist(img_gray)
# 自适应直方图均衡化
# 第一个参数阈值，第二个参数每个方块的宽高
hist_func = cv2.createCLAHE(40, [10, 10])
hist = hist_func.apply(img_gray)
