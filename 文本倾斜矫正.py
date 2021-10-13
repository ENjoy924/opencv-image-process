import matplotlib.pyplot as plt
import numpy as np
import cv2

img = cv2.imread('imgs/pos_24.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_not(gray)
thres = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# 返回两个list，横大于0的索引和列大于0的索引
arr = np.where(thres > 0)
# 将二位数组按列进行合并
coords = np.column_stack(np.where(thres > 0))
# coords = np.stack(np.where(thres > 0), axis=1)
# 返回中心(x,y) (w,h)和旋转角度
angle = cv2.minAreaRect(coords)[-1]
rect = cv2.minAreaRect(coords)
# 将rect转化为四个点的坐标
box = cv2.boxPoints(rect)
print(angle)
# 返回的角度范围是-90到0
# 旋转的时候以y最大且最右边为起始点，顺时针旋转角度最近的边为目标边，求这个边与X轴的夹角
if angle < -45:
    angle = -(90 + angle)
else:
    angle = -angle

h, w = img.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1)
rotated = cv2.warpAffine(img, M, dsize=(w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
cv2.putText(rotated, str(angle), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
cv2.imshow('', rotated)
cv2.waitKey(0)
