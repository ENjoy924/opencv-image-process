import matplotlib.pyplot as plt
import numpy as np
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2

DIGITS_LOOKUP = {
    (1, 1, 1, 0, 1, 1, 1): 0,
    (0, 0, 1, 0, 0, 1, 0): 1,
    (1, 0, 1, 1, 1, 1, 0): 2,
    (1, 0, 1, 1, 0, 1, 1): 3,
    (0, 1, 1, 1, 0, 1, 0): 4,
    (1, 1, 0, 1, 0, 1, 1): 5,
    (1, 1, 0, 1, 1, 1, 1): 6,
    (1, 0, 1, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 1, 0, 1, 1): 9
}

image = cv2.imread('digit.jpg')

image = imutils.resize(image, height=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 50, 200, 255)
# plt.imshow(edged, 'gray')
# plt.show()
# 检测轮廓
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cv2,和cv3不一样，这里用兼容性提取轮廓
cnts = imutils.grab_contours(cnts)
# 根据轮廓的面积对轮廓进行排序
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
# -1表示画出所有轮廓,正数表示画出第几个轮廓
cv2.drawContours(gray, cnts, -1, (255, 255, 255), 3)
# plt.imshow(gray, 'gray')
# plt.show()
displayCnt = None
for c in cnts:
    # 计算轮廓的周长
    peri = cv2.arcLength(c, True)
    # 将给定点的曲线折线化,
    # 第二个参数表示阈值，距离小于此阈值则舍弃,参数值越小越接近曲线
    # 第三个参数是曲线是否闭合的标志位
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        displayCnt = approx
        break

# 利用四个点的仿射变换提取相关区域
warped = four_point_transform(gray, displayCnt.reshape(4, 2))
output = four_point_transform(image, displayCnt.reshape(4, 2))
# 进行阈值处理
thresh = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 3, 2)
# 进行形态学处理
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, [1, 2])
thresh = cv2.dilate(thresh, kernel, iterations=2)
thresh = cv2.erode(thresh, kernel, iterations=2)
# 寻找轮廓
cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
digitsCnts = []
for c in cnts:
    # 计算每个轮廓的外接矩形
    (x, y, w, h) = cv2.boundingRect(c)
    if w > 15 and (h >= 30 and h <= 40):
        digitsCnts.append(c)
# 从左到右进行排序
digitsCnts = contours.sort_contours(digitsCnts)[0]
for c in digitsCnts:
    (x, y, w, h) = cv2.boundingRect(c)
    roi = thresh[y:y + h, x:x + w]
    # plt.imshow(roi, 'gray')
    # plt.show()
    roiH, roiW = roi.shape
    (dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
    dHC = int(roiH * 0.05)
    # 二极管发光宽度（横一是竖一的二倍）
    # 所以横向分成四段，两边各一段，中间两端
    # 纵向分成7段，上一段，中上两端，中一段，中下两端，下一段
    segments = [
        ((0, 0), (w, dH)),  # top
        ((0, 0), (dW, h // 2)),  # top-left
        ((w - dW, 0), (w, h // 2)),  # top-right
        ((0, (h // 2) - dHC), (w, (h // 2) + dHC)),  # center
        ((0, h // 2), (dW, h)),  # bottom-left
        ((w - dW, h // 2), (w, h)),  # bottom-right
        ((0, h - dH), (w, h))  # bottom
    ]
    on = [0] * len(segments)
    for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
        segROI = roi[yA:yB, xA:xB]
        # 计算255像素的个数
        total = cv2.countNonZero(segROI)
        area = (xB - xA) * (yB - yA)
        if total / float(area) > 0.5:
            on[i] = 1
    digit = DIGITS_LOOKUP[tuple(on)]
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
    cv2.putText(output, str(digit), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
cv2.imshow('output', output)
cv2.waitKey(0)
