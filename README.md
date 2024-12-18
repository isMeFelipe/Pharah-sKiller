# Criar Venv
python3 -m venv novo_venv

# Rodar Venv
source venv/bin/activate

# Sair do Venv
deactivate

# LabelImg
Ferramenta para rotular imagens para treinamento do YOLO

    # Rodar
     python labelImg.py





# Pharah-sKiller
Projeto da disciplina de Visão Computacional em que se tem como objetivo: 

-- Detectar Pharah
-- Prever Movimentação de Pharah
                

# Treinar
yolo task=detect \
mode=train \
model=yolov8s.pt \
data=/var/www/html/Pharah-sKiller/YOLO/datasetv2/data.yaml \
epochs=100 \
imgsz=416
batch=8

yolo task=detect \
mode=train \
model=yolov8s.pt \
data=/var/www/html/Pharah-sKiller/YOLO/datasetv3/data.yaml \
epochs=100 \
imgsz=416 \
batch=8



# Rodar
yolo task=detect \
mode=predict \
model=/var/www/html/Pharah-sKiller/runs/detect/train9/weights/best.pt \
source=/var/www/html/Pharah-sKiller/YOLO/test-dataset/pharaVideoTest.mkv \
imgsz=640 \
conf=0.25 \
save=True


yolo task=detect \
mode=predict \
model=/var/www/html/Pharah-sKiller/runs/detect/train16/weights/best.pt \
source=/var/www/html/Pharah-sKiller/YOLO/test-dataset/v1.mp4 \
imgsz=416 \
conf=0.25 \
save=True


