import numpy as np
import cv2

img = cv2.imread('+img.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (5, 5), cv2.BORDER_DEFAULT)

average_pixel = np.mean(blur)  # считаем среднее значение массива пикселей

ret, thresh = cv2.threshold(blur, average_pixel, 255, cv2.THRESH_BINARY_INV)

cv2.imshow('img', thresh)  # для отладки просматриваем контраст из-за порогового значения
cv2.waitKey(0)
cv2.destroyAllWindows()

contours, h = cv2.findContours(thresh, 1, 2)

for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
    # print(len(approx))
    if len(approx) == 3:
        print("triangle")
        # cv2.drawContours(img, [cnt], 0, 255, -1)
    elif len(approx) == 4:
        print("rectangle")
    elif len(approx) == 5:
        print("pentagon")
    elif len(approx) > 10:
        print("circle")
