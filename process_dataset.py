#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для поиска оптимального размера блока BSR-матрицы
для каждого изображения в датасете.
"""

import os
import cv2 
import glob
import math
import pandas as pd
import numpy as np
from pathlib import Path

# Импорт функций из репозитория
# Убедитесь, что файлы находятся в той же директории или в PYTHONPATH
from bsr_matrix import to_bsr
from image_transformation import detect_text_boxes, extract_text_only


def adjust_to_multiple_of_16(h: int, w: int, round_up: bool = True) -> tuple[int, int]:
    """
    Подгоняет размеры изображения до ближайших чисел, кратных 16.
    
    Параметры:
    ----------
    round_up : bool
        True  → округление ВВЕРХ (сохраняем информацию, добавляем падинг)
        False → округление ВНИЗ (обрезка, но чуть лучше сжатие)
    """
    if round_up:
        # 🔹 Округление ВВЕРХ до кратного 16: (x + 15) // 16 * 16
        adj_h = ((h + 15) // 16) * 16
        adj_w = ((w + 15) // 16) * 16
    else:
        # 🔹 Округление ВНИЗ до кратного 16: x // 16 * 16
        adj_h = (h // 16) * 16
        adj_w = (w // 16) * 16
    
    # Защита от нулевых размеров
    return max(adj_h, 16), max(adj_w, 16)


def get_common_divisors(h: int, w: int, max_divisor: int = None) -> list[int]:
    """
    Находит все общие делители чисел h и w.
    """
    if max_divisor is None:
        max_divisor = int(math.sqrt(min(h, w)))
    
    common = []
    for d in range(1, min(max_divisor + 1, min(h, w) + 1)):
        if h % d == 0 and w % d == 0:
            common.append(d)
    return common


def calculate_compression_ratio(original_matrix: np.ndarray, block_size: int) -> float:
    """
    Вычисляет коэффициент сжатия при использовании BSR-формата.
    Матрица уже имеет размеры, кратные 16.
    """
    original_size = original_matrix.nbytes
    bsr = to_bsr(original_matrix, b=block_size)
    
    # Оценка размера BSR в памяти
    n_blocks = bsr.data.shape[0]
    block_elements = block_size * block_size
    data_size = n_blocks * block_elements * original_matrix.itemsize
    indices_size = n_blocks * 4
    indptr_size = (bsr.shape[0] // block_size + 1) * 4
    
    compressed_size = data_size + indices_size + indptr_size
    return original_size / compressed_size if compressed_size > 0 else 0


def find_optimal_block_size(
    image_path: str, 
    metric: str = 'compression',
    padding: int = 2,
    round_up: bool = True
) -> tuple[int, float, int, int]:
    """
    Находит оптимальный размер блока после подгонки изображения под кратность 16.
    
    Возвращает:
    -----------
    (optimal_block, best_score, adjusted_h, adjusted_w)
    """
    # Шаг 1: Детекция текста на ОРИГИНАЛЬНОМ изображении
    boxes = detect_text_boxes(image_path)
    text_image = extract_text_only(image_path, boxes, padding=padding)
    
    h_orig, w_orig = text_image.shape
    
    # 🔹 Шаг 2: Подгоняем размеры до кратных 16
    h_adj, w_adj = adjust_to_multiple_of_16(h_orig, w_orig, round_up=round_up)
    
    # 🔹 Шаг 3: Обрезаем или паддим матрицу до новых размеров
    if round_up:
        # Падинг нулями (сохраняем информацию)
        if h_orig >= h_adj and w_orig >= w_adj:
            text_image_adj = text_image[:h_adj, :w_adj]
        else:
            text_image_adj = np.zeros((h_adj, w_adj), dtype=text_image.dtype)
            h_copy = min(h_orig, h_adj)
            w_copy = min(w_orig, w_adj)
            text_image_adj[:h_copy, :w_copy] = text_image[:h_copy, :w_copy]
    else:
        # Обрезка (проще, но теряем края)
        text_image_adj = text_image[:h_adj, :w_adj]
    
    # 🔹 Шаг 4: Находим все общие делители подогнанных размеров
    valid_blocks = get_common_divisors(h_adj, w_adj)
    
    if not valid_blocks:
        return 1, 0.0, h_adj, w_adj
    
    best_score = -np.inf
    optimal_block = valid_blocks[0]
    
    # 🔹 Шаг 5: Перебор ТОЛЬКО валидных блоков
    for b in valid_blocks:
        try:
            if metric == 'compression':
                score = calculate_compression_ratio(text_image_adj, b)
            elif metric == 'sparsity':
                bsr = to_bsr(text_image_adj, b=b)
                total_blocks = (h_adj // b) * (w_adj // b)
                nonzero_blocks = bsr.data.shape[0]
                score = 1.0 - (nonzero_blocks / total_blocks) if total_blocks > 0 else 0
            else:
                raise ValueError(f"Неизвестная метрика: {metric}")
            
            if score > best_score:
                best_score = score
                optimal_block = b
                
        except Exception as e:
            print(f"  ⚠️ Пропущен блок b={b}: {e}")
            continue
    
    return optimal_block, best_score, h_adj, w_adj


def process_dataset(
    dataset_dir: str, 
    output_csv: str,
    image_extensions: list[str] = None,
    metric: str = 'compression',
    padding: int = 2,
    round_up: bool = True  # 🔹 Округление: True=вверх, False=вниз
) -> pd.DataFrame:
    if image_extensions is None:
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    
    results = []
    
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(dataset_dir, f'*{ext}')))
    
    image_paths = [
        path for path in image_paths 
        if os.path.isfile(path) and Path(path).suffix.lower() in image_extensions
    ]
    
    print(f"📂 Найдено изображений: {len(image_paths)}")
    
    if len(image_paths) > 0:
        print(f"📋 Примеры файлов:")
        for path in image_paths[:5]:
            print(f"   - {Path(path).name}")
        if len(image_paths) > 5:
            print(f"   ... и ещё {len(image_paths) - 5} файлов")
    
    for idx, img_path in enumerate(image_paths, 1):
        image_id = Path(img_path).stem
        print(f"[{idx}/{len(image_paths)}] Обработка: {image_id}")
        
        try:
            optimal_block, best_score, adj_h, adj_w = find_optimal_block_size(
                img_path, 
                metric=metric, 
                padding=padding,
                round_up=round_up
            )
            
            orig_img = cv2.imread(img_path)
            orig_h, orig_w = orig_img.shape[:2] if orig_img is not None else (None, None)
            
            results.append({
                'image_id': image_id,
                'optimal_block': optimal_block,
                'best_score': round(best_score, 4),
                'original_h': orig_h,
                'original_w': orig_w,
                'adjusted_h': adj_h,
                'adjusted_w': adj_w,
                'size_change_h': adj_h - orig_h if orig_h else None,
                'size_change_w': adj_w - orig_w if orig_w else None
            })
            
        except Exception as e:
            print(f"⚠️  Ошибка при обработке {image_id}: {e}")
            results.append({
                'image_id': image_id,
                'optimal_block': None,
                'best_score': None,
                'original_h': None, 'original_w': None,
                'adjusted_h': None, 'adjusted_w': None,
                'size_change_h': None, 'size_change_w': None
            })
    
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"\n✅ Результат сохранён в: {output_csv}")
        
        # 🔹 Статистика по изменению размеров
        if 'size_change_h' in df.columns and df['size_change_h'].notna().any():
            print(f"\n📏 Статистика подгонки размеров (кратность 16):")
            print(f"   Изменение высоты:  мин={df['size_change_h'].min():+.0f}, "
                  f"макс={df['size_change_h'].max():+.0f}, "
                  f"сред={df['size_change_h'].mean():+.1f}")
            print(f"   Изменение ширины:  мин={df['size_change_w'].min():+.0f}, "
                  f"макс={df['size_change_w'].max():+.0f}, "
                  f"сред={df['size_change_w'].mean():+.1f}")
        
        return df
    else:
        print("❌ Нет данных для сохранения!")
        return pd.DataFrame(columns=['image_id', 'optimal_block', 'best_score',
                                     'original_h', 'original_w', 'adjusted_h', 'adjusted_w'])


# ==================== Пример использования ====================
if __name__ == "__main__":
    # Настройки
    DATASET_PATH = "Data"  # Путь к папке с изображениями
    OUTPUT_FILE = "optimal_blocks.csv"  # Имя выходного CSV
    METRIC = "compression"  # или "sparsity"
    PADDING = 2  # Отступ вокруг текста в пикселях
    
    # Запуск обработки
    result_df = process_dataset(
        dataset_dir=DATASET_PATH,
        output_csv=OUTPUT_FILE,
        metric=METRIC,
        padding=PADDING
    )
    
    # Краткая статистика
    print("\n📊 Статистика результатов:")
    print(result_df['optimal_block'].describe())