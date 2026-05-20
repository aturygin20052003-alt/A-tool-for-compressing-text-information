#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random Forest для предсказания оптимального размера блока BSR.
Быстро, стабильно, интерпретируемо — идеально для малого датасета (~300 изображений).
"""
from pathlib import Path

import os
import math
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')


def read_image_safe(path: str) -> np.ndarray:
    """Читает изображение, обходя баг OpenCV с кириллицей в путях Windows."""
    with open(path, 'rb') as f:
        file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


# ==================== 1. Извлечение признаков ====================

def extract_features(image_path: str) -> dict:
    """Извлекает признаки, которые реально влияют на оптимальный блок."""
    img = read_image_safe(image_path)
    if img is None:
        return None
    
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 🔹 1. Анализ текста (самое важное!)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Плотность текста
    text_density = np.mean(thresh == 0)  # Доля чёрных пикселей
    
    # 🔹 2. Морфологический анализ (размер символов)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    # Горизонтальные линии (строки текста)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
    horizontal_density = np.mean(horizontal_lines > 0)
    
    # Вертикальные линии (колонки)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)
    vertical_density = np.mean(vertical_lines > 0)
    
    # 🔹 3. Подсчёт контуров (количество текстовых блоков)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_contours = len(contours)
    
    # Средний размер контура
    if num_contours > 0:
        contour_areas = [cv2.contourArea(c) for c in contours]
        avg_contour_area = np.mean(contour_areas)
        median_contour_area = np.median(contour_areas)
    else:
        avg_contour_area = 0
        median_contour_area = 0
    
    # 🔹 4. Анализ плотности на разных масштабах
    small = cv2.resize(gray, (w//4, h//4))
    _, small_thresh = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    small_density = np.mean(small_thresh == 0)
    
    # 🔹 5. Статистика яркости
    mean_int = np.mean(gray)
    std_int = np.std(gray)
    
    # 🔹 6. Края (текст = много границ)
    edges = cv2.Laplacian(gray, cv2.CV_64F)
    edge_var = np.var(edges)
    edge_mean = np.mean(np.abs(edges))
    
    # 🔹 7. Сложность изображения
    _, thresh_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    complexity = cv2.countNonZero(thresh_inv) / (h * w)
    
    return {
        # Геометрия
        'height': h,
        'width': w,
        'aspect_ratio': w / h if h > 0 else 1.0,
        'area': h * w,
        'min_dim': min(h, w),
        'max_dim': max(h, w),
        'sqrt_min_dim': math.sqrt(min(h, w)),
        
        # 🔹 ТЕКСТОВЫЕ ПРИЗНАКИ (самые важные!)
        'text_density': text_density,
        'num_text_blocks': num_contours,
        'avg_contour_area': avg_contour_area,
        'median_contour_area': median_contour_area,
        'horizontal_structure': horizontal_density,
        'vertical_structure': vertical_density,
        'small_scale_density': small_density,
        
        # Статистика
        'mean_intensity': mean_int,
        'std_intensity': std_int,
        'intensity_range': np.max(gray) - np.min(gray),
        
        # Края
        'edge_variance': edge_var,
        'edge_mean_abs': edge_mean,
        
        # Сложность
        'complexity': complexity,
        
        # Комбинированные
        'text_to_area': text_density * (h * w) / 1e6,
        'blocks_per_area': num_contours / (h * w) * 1e6,
    }


# ==================== 2. Ограничение кратности ====================

def find_valid_blocks(h: int, w: int, max_block: int = None) -> list:
    """Возвращает все b, которые делят h и w нацело."""
    if max_block is None:
        max_block = int(math.sqrt(min(h, w)))
    return [b for b in range(1, max_block + 1) if h % b == 0 and w % b == 0]

def enforce_divisibility(pred_b: float, h: int, w: int) -> int:
    """Корректирует предсказание до ближайшего валидного делителя."""
    valid = find_valid_blocks(h, w)
    if not valid:
        return 1
    return min(valid, key=lambda b: abs(b - pred_b))


# ==================== 3. Подготовка данных ====================

def prepare_data(csv_path: str, data_dir: str = "Data", 
                 test_size: float = 0.2, val_size: float = 0.15, random_state: int = 42):
    """Загружает CSV, извлекает признаки, делит на train/val/test."""
    
    # 🔹 Используем Path для корректной работы с путями
    csv_path = Path(csv_path)
    data_dir = Path(data_dir)
    
    # Проверка существования
    if not csv_path.exists():
        raise FileNotFoundError(f"❌ CSV файл не найден: {csv_path.absolute()}")
    if not data_dir.exists():
        raise FileNotFoundError(f"❌ Папка с данными не найдена: {data_dir.absolute()}")
    
    df = pd.read_csv(csv_path)
    df['image_id'] = df['image_id'].astype(int).astype(str)  # Убираем ".0"
    
    X_list, y_list, h_list, w_list, paths = [], [], [], [], []
    
    print(f"🔍 Извлечение признаков из {len(df)} изображений...")
    print(f"📂 Путь к данным: {data_dir.absolute()}")
    
    for _, row in df.iterrows():
        img_id = row['image_id']
        path = None
        
        # 🔹 Ищем файл с разными расширениями
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']:
            p = data_dir / f"{img_id}{ext}"
            if p.exists():
                path = str(p)  # 🔹 Конвертируем в строку для cv2
                break
        
        if path is None:
            continue
        
        # 🔹 Пробуем прочитать изображение
        img = read_image_safe(path)
        if img is None:
            print(f"⚠️ Не удалось прочитать: {path}")
            continue
        
        features = extract_features(path)
        if features is None:
            continue
        
        h, w = img.shape[:2]
        
        X_list.append(features)
        y_list.append(row['optimal_block'])
        h_list.append(h)
        w_list.append(w)
        paths.append(path)
    
    if len(X_list) == 0:
        # 🔹 Подробная диагностика
        print(f"\n❌ ОШИБКА: Не найдено ни одного изображения!")
        print(f"   CSV файл: {csv_path.absolute()}")
        print(f"   Папка Data: {data_dir.absolute()}")
        print(f"   Записей в CSV: {len(df)}")
        
        # Проверяем первые несколько ID
        if len(df) > 0:
            print(f"\n   Примеры ID из CSV:")
            for i, row in df.head(5).iterrows():
                img_id = str(int(row['image_id']))
                print(f"     - {img_id}")
                
                # Проверяем, существует ли файл
                found = False
                for ext in ['.png', '.jpg', '.jpeg']:
                    test_path = data_dir / f"{img_id}{ext}"
                    if test_path.exists():
                        print(f"       ✅ Найден: {test_path.name}")
                        found = True
                        break
                if not found:
                    print(f"       ❌ Не найден ни один файл")
        
        raise ValueError("❌ Не найдено ни одного изображения! Проверьте пути.")
    
    X = pd.DataFrame(X_list)
    y = np.array(y_list)
    
    print(f"✅ Извлечено {len(X)} образцов с {X.shape[1]} признаками\n")
    
    # Стратифицированный сплит по бинам целевой переменной
    y_bins = pd.cut(y, bins=min(5, len(np.unique(y))), labels=False)
    
    X_train, X_temp, y_train, y_temp, h_train, h_temp, w_train, w_temp, paths_train, paths_temp = train_test_split(
        X, y, h_list, w_list, paths,
        test_size=test_size + val_size, 
        random_state=random_state  # stratify убран
    )
    
    X_val, X_test, y_val, y_test, h_val, h_test, w_val, w_test, paths_val, paths_test = train_test_split(
        X_temp, y_temp, h_temp, w_temp, paths_temp,
        test_size=test_size / (test_size + val_size), 
        random_state=random_state  # stratify убран
    )
    
    print(f"📊 Размеры выборок: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    
    data = {
        'train': (X_train, y_train, h_train, w_train, paths_train),
        'val': (X_val, y_val, h_val, w_val, paths_val),
        'test': (X_test, y_test, h_test, w_test, paths_test)
    }
    
    return data


# ==================== 4. Метрики ====================

def compute_metrics(preds, targets):
    """Метрики для классификации (предсказываем точное число)."""
    preds = np.array(preds)
    targets = np.array(targets)
    
    # Точное совпадение классов
    acc_exact = accuracy_score(targets, preds)
    
    # Средняя абсолютная ошибка (насколько далеко угадали)
    mae = mean_absolute_error(targets, preds)
    
    # Попадание в ±1 блок (для классификации это просто разница <= 1)
    acc_within_1 = np.mean(np.abs(preds.astype(int) - targets.astype(int)) <= 1)
    
    return {'mae_corr': mae, 'acc_exact': acc_exact, 'acc_1': acc_within_1}


# ==================== 5. Обучение ====================

def train_model(data, model_type='rf', **kwargs):
    """Обучает RandomForestClassifier для предсказания дискретных блоков."""
    from sklearn.ensemble import RandomForestClassifier
    
    X_train, y_train, h_train, w_train, _ = data['train']
    X_val, y_val, h_val, w_val, _ = data['val']
    X_test, y_test, h_test, w_test, _ = data['test']
    
    # Преобразуем цели в целые числа (классы)
    y_train = y_train.astype(int)
    y_val = y_val.astype(int)
    y_test = y_test.astype(int)
    
    print(f"\n🔄 Обучение RandomForestClassifier...")
    
    # Используем Классификатор вместо Регрессора
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
        **kwargs
    )
    
    try:
        model.fit(X_train, y_train)
    except ValueError as e:
        print(f"❌ Ошибка обучения: {e}")
        print("   Возможно, в тесте есть классы, которых нет в трейне.")
        return None, {}

    # Оценка
    results = {}
    for name, (X, y, h, w, _) in [('Train', data['train']), ('Val', data['val']), ('Test', data['test'])]:
        preds = model.predict(X)
        
        # Используем новую функцию метрик (без enforce_divisibility)
        metrics = compute_metrics(preds, y)
        results[name.lower()] = metrics
        
        print(f"\n📊 {name}:")
        print(f"  MAE (blocks)      : {metrics['mae_corr']:.3f}")
        print(f"  Accuracy (exact)  : {metrics['acc_exact']:.1%}")
        print(f"  Accuracy (±1)     : {metrics['acc_1']:.1%}")
        
    return model, results


# ==================== 6. Визуализация ====================

def plot_feature_importance(model, feature_names, save_path='feature_importance.png', top_n=15):
    """Строит график важности признаков."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    
    plt.figure(figsize=(10, 6))
    plt.title('Важность признаков (Top 15)')
    plt.barh(range(len(indices)), importances[indices], color='steelblue')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Важность')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Важность признаков сохранена в {save_path}")


def plot_predictions(y_true, y_pred_corr, split_name='Test', save_path=None):
    """Scatter-plot: предсказание vs истинное значение."""
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred_corr, alpha=0.6, edgecolors='black')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label='Идеальное предсказание')
    plt.xlabel('Истинный оптимальный блок')
    plt.ylabel('Предсказанный блок (скорректированный)')
    plt.title(f'{split_name}: Предсказания модели')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 График предсказаний сохранён в {save_path}")
    plt.close()


# ==================== 7. Запуск ====================

if __name__ == "__main__":
    # Настройки

    # 🔹 Определяем пути относительно расположения этого файла
    PROJECT_ROOT = Path(__file__).parent.parent
    CSV_PATH = PROJECT_ROOT / "optimal_blocks.csv"
    DATA_DIR = PROJECT_ROOT / "Data"
    
    # Настройки
    MODEL_TYPE = 'rf'  # 'rf' или 'gb'
    SAVE_MODEL = True
    MODEL_SAVE_PATH = PROJECT_ROOT / "models" / "rf_block_predictor.pkl"
    
    print("🌲 Random Forest для предсказания оптимального блока BSR\n")
    print(f"📂 Корень проекта: {PROJECT_ROOT}")
    print(f"📄 CSV: {CSV_PATH}")
    print(f"📁 Data: {DATA_DIR}\n")
    
    # Проверка существования файлов
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"❌ Не найден файл: {CSV_PATH}\n"
                               f"   Запустите сначала process_dataset.py!")
    
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"❌ Не найдена папка: {DATA_DIR}")
    
    # 1. Подготовка данных
    data = prepare_data(str(CSV_PATH), str(DATA_DIR))
    
    # 2. Обучение
    model, results = train_model(data, model_type=MODEL_TYPE)
    
    # 3. Сохранение модели
    if SAVE_MODEL:
        joblib.dump(model, MODEL_SAVE_PATH)
        print(f"\n💾 Модель сохранена: {MODEL_SAVE_PATH}")
    
    # 4. Визуализация
    print("\n📈 Генерация графиков...")
    plot_feature_importance(model, data['train'][0].columns.tolist())
    
    # График предсказаний на тесте
    X_test, y_test, h_test, w_test, _ = data['test']
    preds_test = model.predict(X_test)
    preds_corr = [enforce_divisibility(p, int(h), int(w)) for p, h, w in zip(preds_test, h_test, w_test)]
    plot_predictions(y_test, preds_corr, split_name='Test', save_path='predictions_scatter.png')
    
    # 5. Финальная сводка
    print("\n" + "="*52)
    print("🏁 ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("="*52)
    print(f"{'Выборка':<10} | {'Acc (exact)':<12} | {'Acc (±1)':<10} | {'MAE (corr)'}")
    print("-"*52)
    for split in ['train', 'val', 'test']:
        m = results[split]
        print(f"{split.upper():<10} | {m['acc_exact']:<12.1%} | {m['acc_1']:<10.1%} | {m['mae_corr']:.3f}")
    print("="*52)
    
    # 6. Топ-5 важных признаков
    print(f"\n🔍 Топ-5 важных признаков:")
    importances = model.feature_importances_
    feat_names = data['train'][0].columns.tolist()
    top_idx = np.argsort(importances)[-5:][::-1]
    for i, idx in enumerate(top_idx, 1):
        print(f"  {i}. {feat_names[idx]}: {importances[idx]:.3f}")
    
    print("\n✅ Всё готово! Модель можно использовать для предсказания на новых изображениях.")
    