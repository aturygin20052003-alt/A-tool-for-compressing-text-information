#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для поиска оптимального размера блока BSR-матрицы
для каждого изображения в датасете.
"""

import os
import glob
import math
import pandas as pd
import numpy as np
from pathlib import Path

# Импорт функций из репозитория
# Убедитесь, что файлы находятся в той же директории или в PYTHONPATH
from bsr_matrix import to_bsr
from image_transformation import detect_text_boxes, extract_text_only


def calculate_compression_ratio(original_matrix: np.ndarray, block_size: int) -> float:
    """
    Вычисляет коэффициент сжатия при использовании BSR-формата.
    
    Возвращает: отношение размера исходной матрицы к размеру сжатого представления.
    """
    original_size = original_matrix.nbytes
    
    # Преобразуем в BSR и оцениваем размер хранения
    bsr = to_bsr(original_matrix, b=block_size)
    
    # Приблизительный размер BSR: данные + индексы + указатели
    # data: количество ненулевых блоков * (block_size^2) * размер элемента
    # indices: количество ненулевых блоков * 4 байта (int32)
    # indptr: (число строк блоков + 1) * 4 байта
    n_blocks = bsr.data.shape[0]
    block_elements = block_size * block_size
    data_size = n_blocks * block_elements * original_matrix.itemsize
    indices_size = n_blocks * 4  # int32 для индексов столбцов блоков
    indptr_size = (bsr.shape[0] // block_size + 1) * 4
    
    compressed_size = data_size + indices_size + indptr_size
    
    return original_size / compressed_size if compressed_size > 0 else 0


def find_optimal_block_size(
    image_path: str, 
    metric: str = 'compression',
    padding: int = 2
) -> tuple[int, float]:
    """
    Находит оптимальный размер блока для данного изображения.
    
    Параметры:
    ----------
    image_path : str
        Путь к изображению
    metric : str
        Метрика оптимизации: 'compression' (макс. коэффициент сжатия) 
        или 'sparsity' (макс. разреженность)
    padding : int
        Отступ вокруг детектированных текстовых областей
    
    Возвращает:
    -----------
    (optimal_block, best_score) : tuple[int, float]
    """
    # Шаг 1: Детекция текста и извлечение только текстовой области
    boxes = detect_text_boxes(image_path)
    text_image = extract_text_only(image_path, boxes, padding=padding)
    
    # Шаг 2: Получение размеров матрицы
    h, w = text_image.shape
    min_dim = min(h, w)
    max_block = int(math.sqrt(min_dim))
    
    if max_block < 1:
        return 1, 0.0
    
    # 🔧 Шаг 3: Находим все допустимые блоки (которые делят h и w нацело)
    valid_blocks = [
        b for b in range(1, max_block + 1)
        if h % b == 0 and w % b == 0
    ]
    
    if not valid_blocks:
        # Если ни один блок не подходит, используем b=1 (всегда делится)
        valid_blocks = [1]
    
    best_score = -np.inf
    optimal_block = valid_blocks[0]
    
    # 🔧 Шаг 4: Перебираем ТОЛЬКО подходящие блоки (БЕЗ обрезки матрицы!)
    for b in valid_blocks:
        try:
            if metric == 'compression':
                # Используем ПОЛНУЮ матрицу, без обрезки
                score = calculate_compression_ratio(text_image, b)
            elif metric == 'sparsity':
                bsr = to_bsr(text_image, b=b)
                # Доля ненулевых блоков (чем меньше, тем лучше разреженность)
                total_blocks = (h // b) * (w // b)
                nonzero_blocks = bsr.data.shape[0]
                score = 1.0 - (nonzero_blocks / total_blocks) if total_blocks > 0 else 0
            else:
                raise ValueError(f"Неизвестная метрика: {metric}")
            
            if score > best_score:
                best_score = score
                optimal_block = b
                
        except Exception as e:
            # Пропускаем блоки, которые вызвали ошибку
            print(f"  ⚠️ Пропущен блок b={b}: {e}")
            continue
    
    return optimal_block, best_score


def process_dataset(
    dataset_dir: str, 
    output_csv: str,
    image_extensions: list[str] = None,
    metric: str = 'compression',
    padding: int = 2
) -> pd.DataFrame:
    if image_extensions is None:
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    
    results = []
    
    # Поиск всех изображений в директории
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(dataset_dir, f'*{ext}')))
    
    # 🔧 ФИЛЬТРАЦИЯ: убираем файлы без расширений и служебные файлы
    image_paths = [
        path for path in image_paths 
        if os.path.isfile(path) and Path(path).suffix.lower() in image_extensions
    ]
    
    print(f"📂 Найдено изображений: {len(image_paths)}")
    
    # 🔍 Выводим список найденных файлов для отладки
    if len(image_paths) == 0:
        print(f"⚠️  Папка '{dataset_dir}' пуста или не содержит изображений!")
        print(f"💡  Проверьте, что файлы .jpg и .png действительно находятся в папке Data/")
    else:
        print(f"📋 Список файлов:")
        for path in image_paths[:5]:  # Показываем первые 5
            print(f"   - {Path(path).name}")
        if len(image_paths) > 5:
            print(f"   ... и ещё {len(image_paths) - 5} файлов")
    
    for idx, img_path in enumerate(image_paths, 1):
        image_id = Path(img_path).stem  # Имя файла без расширения
        
        print(f"[{idx}/{len(image_paths)}] Обработка: {image_id}")
        
        try:
            optimal_block, best_score = find_optimal_block_size(
                img_path, 
                metric=metric, 
                padding=padding
            )
            
            results.append({
                'image_id': image_id,
                'optimal_block': optimal_block,
                'best_score': round(best_score, 4)
            })
            
        except Exception as e:
            print(f"⚠️  Ошибка при обработке {image_id}: {e}")
            results.append({
                'image_id': image_id,
                'optimal_block': None,
                'best_score': None
            })
    
    # Создание DataFrame и сохранение
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"\n✅ Результат сохранён в: {output_csv}")
        return df
    else:
        print("❌ Нет данных для сохранения!")
        return pd.DataFrame(columns=['image_id', 'optimal_block', 'best_score'])


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