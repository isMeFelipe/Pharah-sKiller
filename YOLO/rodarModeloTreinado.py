from ultralytics import YOLO

# Carregar o modelo treinado
model = YOLO('runs/detect/train/weights/best.pt')

# Detectar no vídeo
results = model.predict(source="path/to/video.mp4", save=True, conf=0.5)

# O vídeo processado será salvo automaticamente
