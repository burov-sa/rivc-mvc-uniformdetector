from ultralytics import YOLO

# Загрузка модели
model = YOLO("yolov8n.yaml")  # построение новой модели

# Использование модели
results = model.train(data="config.yaml", epochs=50)  # Обучение модели на размеченных данных