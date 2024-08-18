# エッジやノイズは画像の高周波成分に対応している
# 振幅の変化が大きくなければ低周波成分
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

img = cv2.imread("Fox.jpg", 0)

# スペクトルの大きさ------------------------------------------------
# np.fft.fft2(): 複素数型の配列を出力
# 第一引数として、グレースケール画像を入力する
f = np.fft.fft2(img)
# 周波数領域の原点（直流成分）を画像の中心に移動させる
# スペクトル全体を N/2 両方向にずらす
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20 * np.log(np.abs(fshift))

# plt.subplot(211), plt.imshow(img, cmap="gray")
# plt.title("Input Image"), plt.xticks([]), plt.yticks([])
# plt.subplot(212), plt.imshow(magnitude_spectrum, cmap="gray")
# plt.title("Magnitude Spectrum"), plt.xticks([]), plt.yticks([])
# plt.tight_layout()
# plt.savefig("1_MagnitudeSpectrum.jpg")
# plt.show()

# ハイパスフィルタ（低周波領域での処理）------------------------------
# 低周波成分に対して矩形windowを使ったマスク処理をすることによって低周波成分を取り除くことができる
# ハイパスフィルタによって、画像中のエッジ検出（エッジ：高周波成分）
rows, cols = img.shape
crow, ccol = int(rows / 2), int(cols / 2)
fshift[crow - 30 : crow + 30, ccol - 30 : ccol + 30] = 0
# np.fft.ifftshift(): 直流成分の位置を画像の左上に戻す
f_ishift = np.fft.ifftshift(fshift)
# np.fft.ifft2(): 逆フーリエ変換を適用
img_back = np.fft.ifft2(f_ishift)
# 複素数型の配列が返ってくるため、絶対値を取る
img_back = np.abs(img_back)

# plt.subplot(211), plt.imshow(img, cmap="gray")
# plt.title("Input Image"), plt.xticks([]), plt.yticks([])
# # plt.subplot(312), plt.imshow(img_back, cmap="gray")
# # plt.title("Image after HPF"), plt.xticks([]), plt.yticks([])
# plt.subplot(212), plt.imshow(img_back)
# plt.title("Result in JET"), plt.xticks([]), plt.yticks([])
# plt.tight_layout()
# plt.savefig("2_HighpassFilter.jpg")
# plt.show()

# openCVを使ったフーリエ変換-----------------------------------
# cv2.dft()やcv2.idft()という関数が用意されてる

dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

# plt.subplot(211), plt.imshow(img, cmap="gray")
# plt.title("Input Image"), plt.xticks([]), plt.yticks([])
# plt.subplot(212), plt.imshow(magnitude_spectrum, cmap="gray")
# plt.title("Magnitude Spectrum"), plt.xticks([]), plt.yticks([])
# plt.tight_layout()
# plt.savefig("3_opencv_Magnitude.jpg")
# plt.show()

# IDFT: ローパスフィルタ（高周波成分の除去）----------------------
# ローパスフィルタでは画像にボケを加える
rows, cols = img.shape
crow, ccol = int(rows / 2), int(cols / 2)

# create a mask first, center square is 1, remaining all zeros
# 低周波領域に高い値を持ち、高周波領域が0となるマスクを作成
mask = np.zeros((rows, cols, 2), np.uint8)
mask[crow - 30 : crow + 30, ccol - 30 : ccol + 30] = 1

# apply mask and inverse DFT
fshift = dft_shift * mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

# plt.subplot(211), plt.imshow(img, cmap="gray")
# plt.title("Input Image"), plt.xticks([]), plt.yticks([])
# plt.subplot(212), plt.imshow(img_back, cmap="gray")
# plt.title("Magnitude Spectrum"), plt.xticks([]), plt.yticks([])
# plt.tight_layout()
# plt.savefig("4_LowpassFilter.jpg")
# plt.show()

# DFTのパフォーマンス最適化------------------------------------------
# 配列のサイズが2の累乗のときに最も高速に動く
# 配列のサイズが2, 3, 5の積で表されるときに最も効率的に計算される
# DFTを適用する前に配列のサイズをゼロパディング等で最適なサイズに変更すると良い
# OpenCV：ゼロパディングは手動で行う必要あり
# Numpy：FFTを計算するときの配列のサイズを指定すれば自動的にゼロパディングが行なわれる
img = cv2.imread("Fox.jpg", 0)
rows, cols = img.shape
print(rows, cols)
# 最適な値を計算するためにcv2.getOptimalDFTSize()を使用
nrows = cv2.getOptimalDFTSize(rows)
ncols = cv2.getOptimalDFTSize(cols)
print(nrows, ncols)

# ゼロパディング
# 方法１
nimg = np.zeros((nrows, ncols))
nimg[:rows, :cols] = img
# 方法２
right = ncols - cols
bottom = nrows - rows
bordertype = cv2.BORDER_CONSTANT  # just to avoid line breakup in PDF file
nimg = cv2.copyMakeBorder(img, 0, bottom, 0, right, bordertype, value=0)

# DFTの計算
start_time = time.time()  # 元の配列サイズ
fft1 = np.fft.fft2(img)
end_time = time.time()
print(end_time - start_time)

start_time = time.time()  # ２つ目の配列サイズのほうが高速
fft2 = np.fft.fft2(img, [nrows, ncols])
end_time = time.time()
print(end_time - start_time)
#
start_time = time.time()  # openCVの関数のほうが高速
dft1 = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
end_time = time.time()
print(end_time - start_time)

start_time = time.time()  # openCVかつ新しい配列サイズが最速
dft2 = cv2.dft(np.float32(nimg), flags=cv2.DFT_COMPLEX_OUTPUT)
end_time = time.time()
print(end_time - start_time)

# なぜLaplacianがハイパスフィルタなのか--------------------------------
# simple averaging filter without scaling parameter
mean_filter = np.ones((3, 3))
# creating a guassian filter
x = cv2.getGaussianKernel(5, 10)
gaussian = x * x.T
print(gaussian)

# different edge detecting filters
# scharr in x-direction
scharr = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
# sobel in x direction
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
# sobel in y direction
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
# laplacian
laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

filters = [mean_filter, gaussian, laplacian, sobel_x, sobel_y, scharr]
filter_name = ["mean_filter", "gaussian", "laplacian", "sobel_x", "sobel_y", "scharr_x"]
fft_filters = [np.fft.fft2(x) for x in filters]
fft_shift = [np.fft.fftshift(y) for y in fft_filters]
mag_spectrum = [np.log(np.abs(z) + 1) for z in fft_shift]

for i in range(6):
    plt.subplot(2, 3, i + 1), plt.imshow(mag_spectrum[i], cmap="gray")
    plt.title(filter_name[i]), plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.savefig("5_Filters.jpg")
plt.show()
