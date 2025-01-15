from ultralytics import YOLO

# Carregar o modelo base YOLOv8
model = YOLO('yolov8n.pt')

# Treinar no dataset personalizado
model.train(data='overwatch.yaml', epochs=50, imgsz=640, batch=16)
