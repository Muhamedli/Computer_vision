import os
import shutil
import glob
import cv2 as cv
import numpy as np

num = 0

cx = 0
cy = 0

# создание папки, где будут храниться результаты обработки картинок
if not os.path.isdir("picture_storage"):
    os.mkdir("picture_storage")
else:
    shutil.rmtree("picture_storage")
    os.mkdir("picture_storage")

# проход по всем картинкам из директории
for file in glob.glob("Circle/*.png"):

    # привязка картинки к файлу
    image = cv.imread(file)

    # перевод в ч/б градацию
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # применяем сглаживание (но оно негативно влияет на определение типа фигуры)
    # blur = cv.GaussianBlur(gray, (5, 5), cv.BORDER_DEFAULT)

    # считаем среднее значение массива пикселей
    average_pixel = np.mean(gray)

    # применяем функцию порогового разграничения групп пикселей
    ret, thresh = cv.threshold(gray, average_pixel, 255,
                               cv.THRESH_BINARY_INV)

    # поиск контуров
    contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    # проходимся по контурам
    for i in contours:
        M = cv.moments(i)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            if (cx == 99 & cy == 99):
                break
            cv.circle(image, (cx, cy), 3, (0, 0, 255), -1)

        x, y, w, h = cv.boundingRect(i)
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        weight = w
        height = h

        # вывод центра фигуры
        print(f"x = {cx}    y = {cy}")

        # определение типа фигуры
        for cnt in contours:
            approx = cv.approxPolyDP(cnt, 0.02 * cv.arcLength(cnt, True), True)
            if len(approx) == 3:
                # Печатаем тип фигуры в консоль и выводим текст на картинку
                print("triangle")
                cv.putText(image, "triangle", (cx - 30, cy - height // 2 - 6),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                break
            elif len(approx) == 4:
                print("rectangle")
                cv.putText(image, "rectangle", (cx - 30, cy - height // 2 - 6),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                break
            elif len(approx) == 5:
                print("pentagon")
                cv.putText(image, "pentagon", (cx - 30, cy - height // 2 - 6),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                break
            elif len(approx) == 6:
                print("hexagon")
                cv.putText(image, "hexagon", (cx - 30, cy - height // 2 - 6),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                break
            elif len(approx) > 7:
                print("circle")
                cv.putText(image, "circle", (cx - 20, cy - height // 2 - 6),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                break

        cv.imwrite("picture_storage/image_" + str(num) + ".png", image)
        num = num + 1
