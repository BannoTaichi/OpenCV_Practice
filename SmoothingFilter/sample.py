import cv2
import numpy as np
from matplotlib import pyplot as plt

img0 = cv2.imread("Flower.jpg")
img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
# 2D Convolution（カーネルフィルタ）--------------------------
kernel = np.ones((5, 5), np.float32) / 25
dst = cv2.filter2D(img, -1, kernel)
dst0 = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
cv2.imwrite("1_convolution.jpg", dst0)

# plt.subplot(211), plt.imshow(img), plt.title("Original")
# plt.xticks([]), plt.yticks([])
# plt.subplot(212), plt.imshow(dst), plt.title("Averaging")
# plt.xticks([]), plt.yticks([])
# plt.tight_layout()
# plt.savefig("1_Conv&Origin.jpg")
# plt.show()

# 画像のぼかし（平滑化）--------------------------------------
# ノイズ除去、高周波成分（エッジ・ノイズ）を消す

# 平均（箱型フィルタ）
# 正規化された箱型フィルタを使いたくない場合は、cv2.boxFilter()の引数にnomalize=Falseを指定
blur = cv2.blur(img, (5, 5))
blur0 = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)
cv2.imwrite("2_meanFilter.jpg", blur0)

# ガウシアンフィルタ
# 注目画素との距離に応じて重みを変える、ガウシアンの標準偏差値を指定する必要がある（0にしたら、カーネルサイズから自動的に計算）
# ガウシアンフィルタは白色雑音の除去に適している
blur = cv2.GaussianBlur(img, (5, 5), 0)
blur0 = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)
cv2.imwrite("2_GaussianFilter.jpg", blur0)

# plt.subplot(211), plt.imshow(img), plt.title("Original")
# plt.xticks([]), plt.yticks([])
# plt.subplot(212), plt.imshow(blur), plt.title("Blurred")
# plt.xticks([]), plt.yticks([])
# plt.tight_layout()
# plt.savefig("2_Smooth&Origin.jpg")
# plt.show()

# 中央値フィルタ
# カーネル内の全画素の中央値を計算、ごま塩ノイズに対して効果的
# 箱型やガウシアンは原画像中には存在しない画素値を出力とするのに対して、中央値フィルタは常に原画像中から出力を選ぶ
salt0 = cv2.imread("salt.jpg")
salt = cv2.cvtColor(salt0, cv2.COLOR_BGR2RGB)
median = cv2.medianBlur(salt, 5)
blur0 = cv2.cvtColor(median, cv2.COLOR_BGR2RGB)
cv2.imwrite("3_MedianFilter.jpg", blur0)
plt.imshow(median)
plt.show()

# バイラテラルフィルタ
# cv2.bilateralFilter()はエッジを保存しながら画像をぼかせる
# 処理速度が遅いが、一方でガウシアンフィルタはエッジの劣化が不可避
# バイラテラルフィルタはガウシアンフィルタを2つ使用
# 1つ目: 空間的に近い位置にあることを保証
# 2つ目: 注目画素に似た画素値を持つ画素の値のみ考慮してフィルタリング
blur = cv2.bilateralFilter(img, 9, 75, 75)
blur0 = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)
cv2.imwrite("4_BilateralFilter.jpg", blur0)
