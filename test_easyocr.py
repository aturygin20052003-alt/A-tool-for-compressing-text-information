import cv2
import easyocr

print("🔄 Загрузка модели EasyOCR...")
reader = easyocr.Reader(['ru', 'en'], gpu=False, verbose=False)
print("✅ Модели загружены!")

# Тест на простом изображении (если есть)
import os
if os.path.exists("test.png"):
    result = reader.readtext("Data/test.png")
    print(f"📝 Найдено текстовых блоков: {len(result)}")
    for bbox, text, score in result[:3]:  # Первые 3 результата
        print(f"   • {text} (confidence: {score:.2f})")
else:
    print("💡 Положите тестовое изображение в Data/test.png для полной проверки")