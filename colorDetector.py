# 特性の色検出
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while 1:

    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    # HSV = [Hue(色相), Saturation(彩度), Value(明度)]
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])

    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    lower_green = np.array([45, 50, 50])
    upper_green = np.array([75, 255, 255])
    # Threshold the HSV image to get only blue colors
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Bitwise-AND mask and original image
    res_blue = cv2.bitwise_and(frame, frame, mask=mask_blue)
    res_red = cv2.bitwise_and(frame, frame, mask=mask_red)
    res_green = cv2.bitwise_and(frame, frame, mask=mask_green)

    cv2.imshow("frame", frame)
    # cv2.imshow("mask_blue", mask_blue)
    cv2.imshow("res_blue", res_blue)
    # cv2.imshow("mask_red", mask_red)
    cv2.imshow("res_red", res_red)
    # cv2.imshow("mask_green", mask_green)
    cv2.imshow("res_green", res_green)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
