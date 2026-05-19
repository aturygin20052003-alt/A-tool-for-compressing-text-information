import numpy as np

def pad_to_multiple_of_16(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    
    # Ближайшие размеры, кратные 16
    new_h = ((h + 15) // 16) * 16
    new_w = ((w + 15) // 16) * 16
    
    # Если уже кратно 16, то возвращается оригинал
    if new_h == h and new_w == w:
        return img
        
    # массив нулей 
    padded = np.zeros((new_h, new_w) + img.shape[2:], dtype=img.dtype)
    
    # оригинал в левый верхний угол
    padded[:h, :w] = img
    return padded
