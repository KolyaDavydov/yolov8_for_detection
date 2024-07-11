import os
import cv2
import numpy as np
import json
from ultralytics import YOLO

TRESHOLD = 0.5
IMG_SIZE = 640


def predict(image):
    # Загружаем в модель обученные веса
    path_to_model = os.getcwd() + '/best.pt'
    model = YOLO(path_to_model)  # путь к обученым весам инницирум в модельке
    img = image.copy()
    # Делаем предсказание всех изображений в папке
    results = model.predict(img,
                            save_dir="runs/detect/predict/",
                            save=False,
                            conf=TRESHOLD)
    result = results[0]

    data = dict()  # для хранения информации по обнаруженым или не обнаруженым символам

    # Заносим в переменную обнаружен ли символ и если обнаружен то какая это нога
    left_or_right = 'not detected'
    if len(result.boxes.cls.tolist()) > 0:
        if result.boxes.cls.tolist()[0] == 0:
            left_or_right = 'left'
        else:
            left_or_right = 'right'

    data['left_or_right'] = left_or_right

    if left_or_right != 'not detected':
        # Извлекаем координаты боундингбокса для дальнейшей отрисовки на изображении
        xyxy = result.boxes.xyxy.tolist()[0]
        x1 = int(xyxy[0])
        y1 = int(xyxy[1])
        x2 = int(xyxy[2])
        y2 = int(xyxy[3])
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Извлекаем координаты центра прямоуголника и наносим информацию на изображение
        x0 = int((x1 + x2) / 2)
        y0 = int((y1 + y2) / 2)

        data['x_center'] = x0
        data['y_center'] = y0
    else:
        data['x_center'] = None
        data['y_center'] = None

    # Наносим полученную информацию на изображение
    y_draw = img.shape[0] // 20
    for_draw = (20, y_draw)
    cv2.putText(img, f'Symbol: {left_or_right}', for_draw, cv2.FONT_HERSHEY_TRIPLEX, img.shape[0] / 1200,
                    (200, 255, 5), 1)

    if left_or_right != 'not detected':
        y_draw = img.shape[0] // 20 + y_draw
        for_draw = (20, y_draw)
        cv2.putText(img, f'Coordinate: {str(x0)}, {str(y0)}', for_draw, cv2.FONT_HERSHEY_TRIPLEX,
                        img.shape[0] / 1200, (200, 255, 5), 1)
    # cv2.imwrite(filename='runs/detect/predict/' + data['img'],
    #                 img=img)

    return img, data




