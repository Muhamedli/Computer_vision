import cv2 as cv
import numpy as np

image = cv.imread('Circle/Triangle_0b0f6ede-2a96-11ea-8123-8363a7ec19e6.png')
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# blur = cv.GaussianBlur(gray, (5, 5), cv.BORDER_DEFAULT)  # применяем сглаживание (но оно негативно влияет на определение типа фигуры)

average_pixel = np.mean(gray)  # считаем среднее значение массива пикселей

ret, thresh = cv.threshold(gray, average_pixel, 255, cv.THRESH_BINARY_INV)

cv.imshow('img', thresh)  # для отладки просматриваем контраст из-за порогового значения
cv.waitKey(0)
cv.destroyAllWindows()

contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)  # поиск контуров

for i in contours:
    M = cv.moments(i)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        if (cx == 99 & cy == 99):
            break
        cv.circle(image, (cx, cy), 3, (0, 0, 255), -1)

    rect = cv.minAreaRect(i)
    box = cv.boxPoints(rect)
    box = np.intp(box)
    cv.drawContours(image, [box], 0, (0, 0, 255), 2)

    print(f"x = {cx}    y = {cy}")  # вывод центра фигуры

    for cnt in contours:
        approx = cv.approxPolyDP(cnt, 0.02 * cv.arcLength(cnt, True), True)
        print(len(approx)) # вывод количества полигонов(ребер) у фигуры
        if len(approx) == 3:
            print("triangle")
            break
        elif len(approx) == 4:
            print("rectangle")
            break
        elif len(approx) == 5:
            print("pentagon")
            break
        elif len(approx) == 6:
            print("hexagon")
            break
        elif len(approx) > 7:
            print("circle")
            break

cv.imshow('img', image)  # для отладки просматриваем контраст из-за порогового значения
cv.waitKey(0)
cv.destroyAllWindows()
