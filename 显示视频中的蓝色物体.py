import numpy as np
import cv2

cap = cv2.VideoCapture(r'C:\Users\jiangzhengquan\Desktop\5.mp4')
low_color = np.array([100, 110, 110])
upper_color = np.array([130, 255, 255])
while True:
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, low_color, upper_color)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow('', res)
    cv2.waitKey(0)
