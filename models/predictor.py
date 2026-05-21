import cv2
import numpy as np
import joblib
import math
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "rf_block_predictor.pkl"
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"❌ Модель не найдена: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

def read_image_safe(path: str) -> np.ndarray:
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)

def extract_features_from_image(image_path: str):
    img = read_image_safe(image_path)
    if img is None: return None, 0, 0
    
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text_density = np.mean(thresh == 0)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
    horizontal_density = np.mean(horizontal_lines > 0)
    
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)
    vertical_density = np.mean(vertical_lines > 0)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_contours = len(contours)
    contour_areas = [cv2.contourArea(c) for c in contours] if num_contours > 0 else [0]
    avg_contour_area = np.mean(contour_areas)
    median_contour_area = np.median(contour_areas)
    
    small = cv2.resize(gray, (max(w//4, 1), max(h//4, 1)))
    _, small_thresh = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    small_density = np.mean(small_thresh == 0)
    
    mean_int = np.mean(gray)
    std_int = np.std(gray)
    edges = cv2.Laplacian(gray, cv2.CV_64F)
    edge_var = np.var(edges)
    edge_mean = np.mean(np.abs(edges))
    _, thresh_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    complexity = cv2.countNonZero(thresh_inv) / (h * w)
    
    return {
        'height': h, 'width': w, 'aspect_ratio': w / h if h > 0 else 1.0,
        'area': h * w, 'min_dim': min(h, w), 'max_dim': max(h, w),
        'sqrt_min_dim': math.sqrt(min(h, w)), 'text_density': text_density,
        'num_text_blocks': num_contours, 'avg_contour_area': avg_contour_area,
        'median_contour_area': median_contour_area, 'horizontal_structure': horizontal_density,
        'vertical_structure': vertical_density, 'small_scale_density': small_density,
        'mean_intensity': mean_int, 'std_intensity': std_int,
        'intensity_range': np.max(gray) - np.min(gray), 'edge_variance': edge_var,
        'edge_mean_abs': edge_mean, 'complexity': complexity,
        'text_to_area': text_density * (h * w) / 1e6, 'blocks_per_area': num_contours / (h * w) * 1e6,
    }, h, w

def enforce_divisibility_safe(pred_b: float, h: int, w: int) -> int:
    """Ограничивает предсказание диапазоном [4, 20] и ищет ближайший делитель."""
    pred_b = np.clip(pred_b, 4, 20)  # Ограничиваем реалистичным диапазоном
    max_b = int(math.sqrt(min(h, w)))
    valid = [b for b in range(4, min(max_b, 20) + 1) if h % b == 0 and w % b == 0]
    if valid:
        return min(valid, key=lambda b: abs(b - pred_b))
    # Fallback: ближайшая степень двойки
    return min([4, 8, 16], key=lambda b: abs(b - round(pred_b)))

def predict_block_size(image_path: str):
    feat_dict, h, w = extract_features_from_image(image_path)
    if feat_dict is None: return None, None
    
    X = pd.DataFrame([feat_dict])
    raw_pred = model.predict(X)[0]
    final_pred = enforce_divisibility_safe(raw_pred, h, w)
    return final_pred

if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).parent
    DATA_DIR = SCRIPT_DIR.parent / "Data"
    print(f"📂 Данные: {DATA_DIR}\n")
    
    # Ищем первые 5 картинок в папке Data
    test_images = list(DATA_DIR.glob("*.[jp][pn]g"))[:5]
    if not test_images:
        print("❌ В папке Data не найдено изображений (.png/.jpg)")
        exit()
        
    print("🔍 ПРЕДСКАЗАНИЕ (Raw → Final)\n")
    for img_path in test_images:
        try:
            pred = predict_block_size(str(img_path))
            print(f"✅ {img_path.name:15} | Предсказание: {pred}")
        except Exception as e:
            print(f"❌ {img_path.name:15} | Ошибка: {e}")