import cv2
import numpy as np
import easyocr
import matplotlib.pyplot as plt
from typing import List, Tuple

def detect_text_boxes_cv2(image_path: str) -> List[Tuple[int, int, int, int]]:
    img = cv2.imread(image_path)
    if img is None: return []

    # 1. Перевод в градации серого и улучшение контраста
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 2. Использование морфологического градиента для выделения контуров объектов
    sq_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, sq_kernel)

    # 3. Бинаризация
    _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 4. Объединение символов в слова
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, rect_kernel)

    # 5. Поиск контуров для блоков
    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = img.shape[:2]
    boxes = []

    for contour in contours:
        # Фильтрация шумовых артефактов по площади контура. Порог можно поменять, если много
        # ненужной информации на фото
        area = cv2.contourArea(contour)
        if area < 150:
            continue

        x, y, box_w, box_h = cv2.boundingRect(contour)

        x1, y1 = x, y
        x2, y2 = x + box_w, y + box_h

        # Отступы чуть больше блока, чтобы весь текст был
        pad_w = max(int((x2 - x1) * 0.05), 7)
        pad_h = max(int((y2 - y1) * 0.05), 7)

        boxes.append((
            max(0, x1 - pad_w),
            max(0, y1 - pad_h),
            min(w, x2 + pad_w),
            min(h, y2 + pad_h)
        ))

    return boxes
