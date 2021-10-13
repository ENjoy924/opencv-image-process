import cv2
import matplotlib.pyplot as plt
import numpy as np

# 均值滤波
img = cv2.imread('lena.jpg')
# # 中值滤波
# img1 = cv2.blur(img, [3, 3])
# # 盒滤波
# img2 = cv2.boxFilter(img, -1, [3, 3], normalize=True)
# # 高斯滤波
# img3 = cv2.GaussianBlur(img, [3, 3], 0)
# # 双边滤波
# img4 = cv2.bilateralFilter(img, 9, 75, 75)

# 进行高通滤波
# 先将图像从时间域转化为频率域,
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 传入的图像需要float32格式的
gray = gray.astype(np.float32)
filter_img = cv2.dft(gray, flags=cv2.DFT_COMPLEX_OUTPUT)
# 低频部分移入到图像的中心位置
filter_img = np.fft.fftshift(filter_img)
# 获得图像的大小,从中心位置进行截取，以获得低通滤波器
h, w = gray.shape
filter_mask = np.zeros(shape=(h, w, 2), dtype=np.uint8)
# 图像低频成分的掩膜
filter_mask[h // 2 - 30:h // 2 + 30, w // 2 - 30:w // 2 + 30] = 1
# 将频率域的图像和掩膜进行相乘
filter_result = filter_mask * filter_img
# 将图像的低频部分移回去
shift_img = np.fft.ifftshift(filter_result)
# 将图像映射回时间域
time_img = cv2.idft(shift_img)
# 转化后的图像有实部和虚部，使用cv2.magnitude转化为空间域内
img_dft = cv2.magnitude(time_img[:, :, 0], time_img[:, :, 1])
# 转化后的img_dft数值比较大,将数值映射到0,255区间
img_dft = (img_dft - np.min(img_dft)) / (np.max(img_dft) - np.min(img_dft))
img_dft = (img_dft * 255).astype(np.uint8)
plt.imshow(img_dft, 'gray')
plt.show()

img = cv2.imread('water.jpg', cv2.IMREAD_GRAYSCALE)
img = img.astype(np.float32)
f_img = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)
s_img = np.fft.fftshift(f_img)
# 将频率图像显示出来
show_img = 20 * np.log(cv2.magnitude(s_img[:, :, 0], s_img[:, :, 1]))
# plt.imshow(show_img,'gray')
# plt.show()
h, w = img.shape
m_mask = np.zeros_like(s_img, dtype=np.uint8)
m_mask[h // 2 - 15:h // 2 + 15, w // 2 - 15:w // 2 + 15] = 1
r_img = m_mask * s_img
s_r_img = np.fft.ifftshift(r_img)
f_img = cv2.idft(s_r_img)
result = cv2.magnitude(f_img[:, :, 0], f_img[:, :, 1])

plt.imshow(result, 'gray')
plt.show()
