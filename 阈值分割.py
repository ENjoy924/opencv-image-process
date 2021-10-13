import cv2
from matplotlib import pyplot as plt

# 固定阈值分割
# img = cv2.imread('gradient.jpg', cv2.IMREAD_GRAYSCALE)
# thres_methods = [i for i in dir(cv2) if i.startswith('THRESH')]
#
# for i in range(len(thres_methods)):
#     plt.subplot(2, 4, i + 1)
#     ret, frame = cv2.threshold(img, 127, 255, getattr(cv2,thres_methods[i]))
#     plt.imshow(frame,cmap=plt.cm.gray)
#     plt.axis('off')
# plt.show()
# 自适应阈值分割
# img = cv2.imread('sudoku.jpg', cv2.IMREAD_GRAYSCALE)
# ret,th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 4)
# th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 6)
# titles = ['origal', 'global', 'adaptive mean', 'adaptive gaussian']
# imgs = [img, th1, th2, th3]
# for i in range(len(imgs)):
#     plt.subplot(2, 2, i + 1)
#     plt.axis('off')
#     plt.imshow(imgs[i],cmap=plt.cm.gray)
#     plt.title(titles[i])
# plt.show()

# img = cv2.imread('193.jpg', cv2.IMREAD_GRAYSCALE)
# th1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, 2)
# th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 6)
# plt.subplot(2, 2, 1)
# plt.imshow(th1, cmap=plt.cm.gray)
# plt.axis('off')
# plt.subplot(2, 2, 2)
# plt.imshow(th2, cmap=plt.cm.gray)
# plt.axis('off')
# plt.show()
# 大津阈值法
# 对图像直方图中有两个尖峰的情况比较好分割
img = cv2.imread('noisy.jpg', cv2.IMREAD_GRAYSCALE)
ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
ret, th2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
h1 = cv2.calcHist([img], [0], None, [256], [0, 256])
img2 = cv2.GaussianBlur(img, [3,3], 0)
ret, th3 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

h2 = cv2.calcHist([img2], [0], None, [256], [0, 256])

imgs = [img, img, img2]
hs = [h1, h1, h2]
ths = [th1, th2, th3]

for i in range(3):
    plt.subplot(3, 3, i * 3 + 1)
    plt.axis('off')
    plt.imshow(imgs[i],'gray')
    plt.subplot(3, 3, i * 3 + 2)
    plt.plot(hs[i])
    plt.subplot(3, 3, i * 3 + 3)
    plt.axis('off')
    plt.imshow(ths[i],'gray')
plt.show()
