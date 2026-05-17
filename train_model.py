#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random Forest для предсказания оптимального размера блока BSR.
Быстро, стабильно, интерпретируемо — идеально для малого датасета (~300 изображений).
"""

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


# ==================== 1. Извлечение признаков ====================

def extract_features(image_path: str) -> dict:
    """
    Извлекает простые, но эффективные признаки из изображения.
    Без OCR — быстро и стабильно.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Статистика яркости
    mean_int = np.mean(gray)
    std_int = np.std(gray)
    
    # Оценка "текстовости": доля тёмных пикселей (текст обычно чёрный)
    dark_ratio = np.mean(gray < 128)
    
    # Плотность краёв (текст = много границ)
    edges = cv2.Laplacian(gray, cv2.CV_64F)
    edge_var = np.var(edges)
    edge_mean = np.mean(np.abs(edges))
    
    # Простая оценка "плотности текста" через порогование
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text_density = np.mean(thresh == 0)  # Доля чёрных пикселей после бинаризации
    
    return {
        # Геометрия
        'height': h,
        'width': w,
        'aspect_ratio': w / h if h > 0 else 1.0,
        'area': h * w,
        'min_dim': min(h, w),
        'max_dim': max(h, w),
        'sqrt_min_dim': math.sqrt(min(h, w)),
        
        # Статистика яркости
        'mean_intensity': mean_int,
        'std_intensity': std_int,
        'intensity_range': np.max(gray) - np.min(gray),
        
        # Текстовые признаки
        'dark_pixel_ratio': dark_ratio,
        'text_density_otsu': text_density,
        'edge_variance': edge_var,
        'edge_mean_abs': edge_mean,
        
        # Комбинированные
        'text_to_area': text_density * (h * w) / 1e6,  # Нормализованная "масса" текста
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
    
    df = pd.read_csv(csv_path)
    df['image_id'] = df['image_id'].astype(int).astype(str)  # Убираем ".0"
    
    X_list, y_list, h_list, w_list, paths = [], [], [], [], []
    
    print(f"🔍 Извлечение признаков из {len(df)} изображений...")
    for _, row in df.iterrows():
        img_id = row['image_id']
        path = None
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']:
            p = os.path.join(data_dir, f"{img_id}{ext}")
            if os.path.exists(p):
                path = p
                break
        
        if path is None:
            continue
            
        features = extract_features(path)
        if features is None:
            continue
        
        img = cv2.imread(path)
        h, w = img.shape[:2]
        
        X_list.append(features)
        y_list.append(row['optimal_block'])
        h_list.append(h)
        w_list.append(w)
        paths.append(path)
    
    if len(X_list) == 0:
        raise ValueError("❌ Не найдено ни одного изображения! Проверьте пути.")
    
    X = pd.DataFrame(X_list)
    y = np.array(y_list)
    
    print(f"✅ Извлечено {len(X)} образцов с {X.shape[1]} признаками\n")
    
    # Стратифицированный сплит по бинам целевой переменной
    y_bins = pd.cut(y, bins=min(5, len(np.unique(y))), labels=False)
    
    X_train, X_temp, y_train, y_temp, h_train, h_temp, w_train, w_temp, paths_train, paths_temp = train_test_split(
        X, y, h_list, w_list, paths,
        test_size=test_size + val_size, random_state=random_state, stratify=y_bins
    )
    
    y_temp_bins = pd.cut(y_temp, bins=min(5, len(np.unique(y_temp))), labels=False)
    X_val, X_test, y_val, y_test, h_val, h_test, w_val, w_test, paths_val, paths_test = train_test_split(
        X_temp, y_temp, h_temp, w_temp, paths_temp,
        test_size=test_size / (test_size + val_size), random_state=random_state, stratify=y_temp_bins
    )
    
    print(f"📊 Размеры выборок: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    
    data = {
        'train': (X_train, y_train, h_train, w_train, paths_train),
        'val': (X_val, y_val, h_val, w_val, paths_val),
        'test': (X_test, y_test, h_test, w_test, paths_test)
    }
    
    return data


# ==================== 4. Метрики ====================

def compute_metrics(preds_raw, targets, h_dims, w_dims):
    """Считает метрики с учётом корректировки на кратность."""
    preds_raw = np.array(preds_raw)
    targets = np.array(targets)
    
    # Сырые метрики
    mae_raw = mean_absolute_error(targets, preds_raw)
    
    # Корректировка на кратность
    preds_corr = np.array([enforce_divisibility(p, int(h), int(w)) 
                           for p, h, w in zip(preds_raw, h_dims, w_dims)])
    
    mae_corr = mean_absolute_error(targets, preds_corr)
    acc_exact = accuracy_score(targets, preds_corr)
    acc_within_1 = np.mean(np.abs(preds_corr - targets) <= 1)
    max_err = np.max(np.abs(preds_corr - targets))
    
    return {
        'mae_raw': mae_raw, 'mae_corr': mae_corr,
        'acc_exact': acc_exact, 'acc_1': acc_within_1, 'max_err': max_err
    }


# ==================== 5. Обучение ====================

def train_model(data, model_type='rf', **kwargs):
    """Обучает модель и возвращает метрики по всем сплитам."""
    
    # Выбор модели
    if model_type == 'rf':
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            **kwargs
        )
    elif model_type == 'gb':
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            **kwargs
        )
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")
    
    X_train, y_train, h_train, w_train, _ = data['train']
    X_val, y_val, h_val, w_val, _ = data['val']
    X_test, y_test, h_test, w_test, _ = data['test']
    
    print(f"\n🔄 Обучение {model_type.upper()}...")
    model.fit(X_train, y_train)
    
    # Оценка
    results = {}
    for name, (X, y, h, w, _) in [('Train', data['train']), ('Val', data['val']), ('Test', data['test'])]:
        preds = model.predict(X)
        metrics = compute_metrics(preds, y, h, w)
        results[name.lower()] = metrics
        
        print(f"\n📊 {name}:")
        print(f"  MAE (raw)       : {metrics['mae_raw']:.3f}")
        print(f"  MAE (corrected) : {metrics['mae_corr']:.3f}")
        print(f"  Accuracy (exact): {metrics['acc_exact']:.1%}")
        print(f"  Accuracy (±1)   : {metrics['acc_1']:.1%}")
        print(f"  Max Error       : {metrics['max_err']}")
    
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
    CSV_PATH = "optimal_blocks.csv"
    DATA_DIR = "Data"
    MODEL_TYPE = 'rf'  # 'rf' или 'gb'
    SAVE_MODEL = True
    
    print("🌲 Random Forest для предсказания оптимального блока BSR\n")
    
    # 1. Подготовка данных
    data = prepare_data(CSV_PATH, DATA_DIR)
    
    # 2. Обучение
    model, results = train_model(data, model_type=MODEL_TYPE)
    
    # 3. Сохранение модели
    if SAVE_MODEL:
        joblib.dump(model, 'rf_block_predictor.pkl')
        print(f"\n💾 Модель сохранена: rf_block_predictor.pkl")
    
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