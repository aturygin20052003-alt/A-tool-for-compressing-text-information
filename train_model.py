#!/usr/bin/env python3
import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используется: {DEVICE}")

BATCH_SIZE = 16
EPOCHS = 30
IMAGE_SIZE = 128
DATA_DIR = "Data"
CSV_PATH = "optimal_blocks.csv"
MODEL_PATH = "block_size_model.pth"


def load_data():
    """ данные, сопоставляя CSV с файлами в папке Data"""
    df = pd.read_csv(CSV_PATH)
    print(f"CSV строк: {len(df)}")
    
    # Получаем список всех файлов в Data
    if not os.path.exists(DATA_DIR):
        print(f"❌ Папка {DATA_DIR} не найдена")
        return [], []
    
    # Создаём словарь: имя файла -> путь
    file_dict = {}
    for filename in os.listdir(DATA_DIR):
        name, ext = os.path.splitext(filename)
        if ext.lower() in ['.png', '.jpg', '.jpeg']:
            file_dict[name] = os.path.join(DATA_DIR, filename)
    
    print(f" Найдено изображений в Data: {len(file_dict)}")
    print(f"   Примеры: {list(file_dict.keys())[:10]}")
    
    # Сопоставляем
    image_paths = []
    labels = []
    matched = 0
    missing = []
    
    for _, row in df.iterrows():
        block = row['optimal_block']
        if pd.isna(block):
            continue
        
        
        img_id = str(int(row['image_id']))
        
        if img_id in file_dict:
            image_paths.append(file_dict[img_id])
            labels.append(int(block))
            matched += 1
        else:
            missing.append(img_id)
    
    print(f" Найдено соответствий: {matched}")
    if missing:
        print(f" Не найдено {len(missing)} файлов, примеры: {missing[:10]}")
    
    return image_paths, labels


class BlockDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx])
        if img is None:
            img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            img = self.transform(img)
        
        label = self.labels[idx] - 1  # классы от 0 до 19
        return img, label


class BlockSizePredictor(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def main():
    
    print("ОБУЧЕНИЕ МОДЕЛИ ДЛЯ ПРЕДСКАЗАНИЯ РАЗМЕРА БЛОКА")
    
    
    # Загрузка данных
    paths, labels = load_data()
    
    if len(paths) == 0:
        print("\n Нет данных для обучения!")
        return
    
    # Разделение на train/val/test
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels, test_size=0.3, random_state=42
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, random_state=42
    )
    
    print(f"\n Размеры выборок:")
    print(f"   Train: {len(train_paths)}")
    print(f"   Val:   {len(val_paths)}")
    print(f"   Test:  {len(test_paths)}")
    
    # Трансформации
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # DataLoader
    train_loader = DataLoader(BlockDataset(train_paths, train_labels, transform), 
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(BlockDataset(val_paths, val_labels, transform), 
                            batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(BlockDataset(test_paths, test_labels, transform), 
                             batch_size=BATCH_SIZE, shuffle=False)
    
    # Модель
    model = BlockSizePredictor().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"\n Модель: {sum(p.numel() for p in model.parameters()):,} параметров")
    
    # Обучение
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0
        for imgs, lbls in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                outputs = model(imgs)
                val_loss += criterion(outputs, lbls).item()
                _, pred = torch.max(outputs, 1)
                total += lbls.size(0)
                correct += (pred == lbls).sum().item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        accuracy = correct / total
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={accuracy:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f" Сохранена лучшая модель")
    
    # Тестирование
   
    print("ТЕСТИРОВАНИЕ")
    
    
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            _, pred = torch.max(outputs, 1)
            predictions.extend(pred.cpu().numpy())
            true_labels.extend(lbls.numpy())
    
    pred_blocks = np.array(predictions) + 1
    true_blocks = np.array(true_labels) + 1
    
    mae = mean_absolute_error(true_blocks, pred_blocks)
    acc = accuracy_score(true_labels, predictions)
    
    print(f"Точность (Accuracy): {acc:.4f}")
    print(f"Средняя абсолютная ошибка (MAE): {mae:.2f}")
    
    # Сохраняем
    results = pd.DataFrame({
        'true_block': true_blocks,
        'pred_block': pred_blocks,
        'error': abs(pred_blocks - true_blocks)
    })
    results.to_csv('test_predictions.csv', index=False)
    
    # График
    plt.figure(figsize=(10, 5))
    plt.hist(pred_blocks - true_blocks, bins=20, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Разница (предсказание - истина)')
    plt.ylabel('Количество')
    plt.title('Гистограмма ошибок предсказаний')
    plt.axvline(x=0, color='red', linestyle='--')
    plt.grid(True, alpha=0.3)
    plt.savefig('prediction_errors.png', dpi=150)
    plt.show()
    
    print(f"\n Модель сохранена: {MODEL_PATH}")
    print(f" Предсказания: test_predictions.csv")


if __name__ == "__main__":
    main()