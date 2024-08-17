import cv2
import numpy as np
from matplotlib import pyplot as plt

img0 = cv2.imread("gradient.png")
# plt.imshow(img0)
# plt.show()
print("img0", img0.shape)
img = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
print("img", img.shape)

# 単純な閾値処理
ret, thresh1 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img, 100, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img, 100, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img, 100, 255, cv2.THRESH_TOZERO_INV)

titles = ["Original Image", "BINARY", "BINARY_INV", "TRUNC", "TOZERO", "TOZERO_INV"]
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

# for i in range(6):
#     plt.subplot(2, 3, i + 1), plt.imshow(images[i], "gray")
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])
# plt.tight_layout()
# plt.savefig("1_thresh.jpg")
# plt.show()

# 適応的閾値処理-----------------------------------------------
# 近傍領域から閾値を決定することで、より正確に二値化できる
img = cv2.imread("Flower.jpg", 0)
img = cv2.medianBlur(img, 5)

ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# MEAN_C: 近傍領域の中央値を閾値とする
th2 = cv2.adaptiveThreshold(
    img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
)
# GAUSSIAN_C: 近傍領域の重み付け平均値を閾値とする（重みはGaussian分布）
th3 = cv2.adaptiveThreshold(
    img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
)

titles = [
    "Original Image",
    "Global Thresholding (v = 127)",
    "Adaptive Mean Thresholding",
    "Adaptive Gaussian Thresholding",
]
images = [img, th1, th2, th3]

# for i in range(4):
#     plt.subplot(2, 2, i + 1), plt.imshow(images[i], "gray")
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])
# plt.tight_layout()
# plt.savefig("2_AdaptiveThreshold.jpg")
# plt.show()

# 大津の二値化-----------------------------------------------
# ２つのピークの間の値を閾値として選ぶ方法
img = cv2.imread("noise.png", 0)
# global thresholding
ret1, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# Otsu's thresholding
ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Otsu's thresholding after Gaussian filtering
# 平滑化処理によって、ノイズの影響が軽減して、より良い二値化ができる
blur = cv2.GaussianBlur(img, (5, 5), 0)
ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# plot all the images and their histograms
images = [img, 0, th1, img, 0, th2, blur, 0, th3]
titles = [
    "Original Noisy Image",
    "Histogram",
    "Global Thresholding (v=127)",
    "Original Noisy Image",
    "Histogram",
    "Otsu's Thresholding",
    "Gaussian filtered Image",
    "Histogram",
    "Otsu's Thresholding",
]

for i in range(3):
    plt.subplot(3, 3, i * 3 + 1), plt.imshow(images[i * 3], "gray")
    plt.title(titles[i * 3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3, 3, i * 3 + 2), plt.hist(images[i * 3].ravel(), 256)
    plt.title(titles[i * 3 + 1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3, 3, i * 3 + 3), plt.imshow(images[i * 3 + 2], "gray")
    plt.title(titles[i * 3 + 2]), plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.savefig("3_Histogram.jpg")
plt.show()
