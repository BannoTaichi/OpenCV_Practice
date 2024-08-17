# 画像の幾何変換
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("Fox.jpg")
print("img", img.shape)
rows, cols, ch = img.shape

img0 = cv2.imread("Fox.jpg", 0)
print("img0", img0.shape)

# 画像のサイズ変更（拡大）----------------------------------
# res = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
# # OR
# height, width = img.shape[:2]
# res = cv2.resize(img, (2 * width, 2 * height), interpolation=cv2.INTER_CUBIC)

# 並進----------------------------------------------------
# 並進を表す変換行列
M_trans = np.float32([[1, 0, 100], [0, 1, 50]])
dst = cv2.warpAffine(img0, M_trans, (cols, rows))
print("dst", dst.shape)
dst0 = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
# plt.imshow(dst0)
# plt.show()
# cv2.imwrite("1_translation.jpg", dst)
# cv2.imshow("img0", dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 回転(スケーリングせずに90°回転させる変換)-----------------------------
# 回転を表す変換行列
M_rotate = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
dst = cv2.warpAffine(img, M_rotate, (cols, rows))
dst0 = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
# plt.imshow(dst0)
# plt.show()
# cv2.imwrite("2_rotation.jpg", dst)

# アフィン変換（変換前後で並行性を保つ変換）
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

M_affine = cv2.getAffineTransform(pts1, pts2)
dst = cv2.warpAffine(img, M_affine, (cols, rows))
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
dst0 = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
# plt.subplot(121), plt.imshow(img1), plt.title("Input")
# plt.subplot(122), plt.imshow(dst0), plt.title("Output")
# # 複数の図を画像として保存(tight_layout(), savefig()を使用)
# plt.tight_layout()
# plt.savefig("3_AffineOriginal.jpg")
# plt.show()
cv2.imwrite("3_Affine.jpg", dst)

# 射影変換------------------------------------------------
pts1 = np.float32([[650, 0], [1600, 0], [500, 900], [1500, 700]])
pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

M_project = cv2.getPerspectiveTransform(pts1, pts2)
dst = cv2.warpPerspective(img, M_project, (300, 300))
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
dst0 = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
plt.subplot(121), plt.imshow(img1), plt.title("Input")
plt.subplot(122), plt.imshow(dst0), plt.title("Output")
plt.tight_layout()
plt.savefig("4_ProjectOriginal.jpg")
plt.show()

dst0 = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
cv2.imwrite("4_project.jpg", dst)
