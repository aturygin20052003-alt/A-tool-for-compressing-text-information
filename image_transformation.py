import cv2
import numpy as np
import easyocr
import matplotlib.pyplot as plt
from typing import List, Tuple

# Инициализация
reader = easyocr.Reader(['ru', 'en'], gpu=False, verbose=False)

def detect_text_boxes(image_path: str) -> List[Tuple[int, int, int, int]]:
    img = cv2.imread(image_path)
    if img is None: return []

    # Препроцессинг для детектора
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    processed = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)

    # Детекция
    results = reader.readtext(processed, detail=1, mag_ratio=1.5)
    
    h, w = img.shape[:2]
    boxes = []
    for res in results:
        bbox = res[0]
        xs = [p[0] for p in bbox]; ys = [p[1] for p in bbox]
        x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        
        # отступы: 7 пикселей минимум + 2% от размера блока
        pad_w = max(int((x2 - x1) * 0.05), 7)
        pad_h = max(int((y2 - y1) * 0.05), 7)
        
        boxes.append((
            max(0, x1 - pad_w), 
            max(0, y1 - pad_h), 
            min(w, x2 + pad_w), 
            min(h, y2 + pad_h)
        ))
    return boxes

def extract_text_only(image_path: str, boxes: List[Tuple[int, int, int, int]], padding: int = 2) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None: return np.array([])
    
    # Оригинальные оттенки серого
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Создание маски
    mask = np.zeros((h, w), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    
    # Расширение маски, чтобы текст не сьедался
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    #черный фон
    final = np.zeros_like(gray)
    
    # Копия пикселей, где был только белый
    final[mask == 255] = gray[mask == 255]
    
    return final

