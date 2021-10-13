import cv2
import imutils


class ShapeDetector():
    def __init__(self):
        pass

    def detect(self, c):
        shape = 'unidentified'
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) == 3:
            shape = 'triangle'
        elif len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            ar = w / float(h)
            shape = 'square' if ar > 0.95 and ar < 1.05 else 'rectangle'
        elif len(approx) == 5:
            shape = 'pentagon'
        else:
            shape = 'circle'
        return shape


def processImage():
    img = cv2.imread('img', cv2.IMREAD_GRAYSCALE)
    img = cv2.GaussianBlur(img, [3, 3], 0)
    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    sd = ShapeDetector()
    for c in cnts:
        # 得到轮廓的矩
        M = cv2.moments(c)
        # 计算轮廓的质心
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])
        shape = sd.detect(c)
        cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
        cv2.putText(img, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

