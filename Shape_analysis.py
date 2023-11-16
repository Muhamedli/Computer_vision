import cv2 as cv
import numpy as np


class ShapeAnalysis:  # Определить класс анализа формы
    def __init__(self):
        self.shapes = {'triangle': 0, 'rectangle': 0, 'polygons': 0, 'circles': 0}

    def analysis(self, frame):
        h, w, ch = frame.shape
        result = np.zeros((h, w, ch), dtype=np.uint8)
        # Бинаризованное изображение
        # print("start to detect lines...\n")
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        cv.imshow("input image", gray)
        ret, thresh = cv.threshold(gray, 50, 255, cv.THRESH_BINARY)
        # cv.imshow("input image", thresh)

        contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for cnt in range(len(contours)):
            # Извлечь и нарисовать контуры
            cv.drawContours(result, contours, cnt, (0, 255, 0), 2)

            # Аппроксимация контура
            epsilon = 0.01 * cv.arcLength(contours[cnt], True)
            approx = cv.approxPolyDP(contours[cnt], epsilon, True)

            # Анализировать геометрию
            corners = len(approx)
            shape_type = ""
            if corners == 3:
                count = self.shapes['triangle']
                count = count + 1
                self.shapes['triangle'] = count
                shape_type = "Треугольник"
            if corners == 4:
                count = self.shapes['rectangle']
                count = count + 1
                self.shapes['rectangle'] = count
                shape_type = "Прямоугольник"

            if corners >= 10:
                count = self.shapes['circles']
                count = count + 1
                self.shapes['circles'] = count
                shape_type = "Круг"

            if 4 < corners < 10:
                count = self.shapes['polygons']
                count = count + 1
                self.shapes['polygons'] = count
                shape_type = "Многоугольник"

            p = cv.arcLength(contours[cnt], True)
            area = cv.contourArea(contours[cnt])
            print("Форма:% s" % (shape_type))

        # cv.imshow("Analysis Result", self.draw_text_info(result))
        return self.shapes


if __name__ == "__main__":
    image = cv.imread("Circle\Circle_0a6ed4de-2a95-11ea-8123-8363a7ec19e6.png")
    ld = ShapeAnalysis()
    ld.analysis(image)
    cv.waitKey(0)
    cv.destroyAllWindows()
