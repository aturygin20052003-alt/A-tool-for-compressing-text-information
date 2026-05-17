import torch
import cv2
import numpy as np
from torchvision import transforms

#  класс модели ( такой же, как при обучении)
class BlockSizePredictor(torch.nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(128, 256, 3, padding=1), torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Загрузка модели
model = BlockSizePredictor()
model.load_state_dict(torch.load('block_size_model.pth', map_location='cpu'))
model.eval()

# Трансформации
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_block_size(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transform(img)
    img = img.unsqueeze(0)
    
    with torch.no_grad():
        output = model(img)
        pred = torch.argmax(output, dim=1).item()
    return pred + 1

# Тест на нескольких изображениях
test_images = ["Data/1.png", "Data/100.png", "Data/103.png"]


print("ПРЕДСКАЗАНИЕ РАЗМЕРОВ БЛОКОВ")


for img_path in test_images:
    block = predict_block_size(img_path)
    if block:
        print(f"{img_path}: {block}x{block}")
    else:
        print(f"{img_path}: не найден")