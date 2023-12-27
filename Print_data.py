import serial
from tkinter import *

# Настраиваем COM порт
port = "COM6"
baudrate = 9600

# Создаем объект gui
gui = Tk()
# Создаем наименование окна
gui.title('Aquarobotics')
# Указываем габариты и координаты окна
gui.geometry("630x120+100+100")
# Создаем объект label
label = Label(gui, text="")
# Настраиваем цвета фона окна и текста
gui['background'] = '#C8FFC8'
label['background'] = '#C8FFC8'

# Принимаем данные из COM порта
input_data = serial.Serial(port, baudrate=baudrate)

while True:
    # Декодируем данные из COM порта
    temp_depth = input_data.readline().decode().strip()
    data = temp_depth.split('$')
    # Выводим данные в окно
    if (len(data) == 2):
        label.config(text="Temperature: " + data[0] + " deg\tDepth: " + data[1] + " cm",
                     font=("Comic Sans MS", 18, "roman"))
        label.place(relx=0.5, rely=0.5, anchor='center')
    # Обновляем окно
    gui.update()
