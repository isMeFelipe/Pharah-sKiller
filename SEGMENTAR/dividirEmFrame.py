import cv2
import os

# Caminho do vídeo de entrada
video_path = 'v3.mp4'

# Caminho para a pasta onde os frames serão salvos
frames_folder = 'frames'

# Cria a pasta "frames" se não existir
if not os.path.exists(frames_folder):
    os.makedirs(frames_folder)

# Carrega o vídeo
cap = cv2.VideoCapture(video_path)

# Verifica se o vídeo foi carregado corretamente
if not cap.isOpened():
    print(f"Erro ao abrir o vídeo {video_path}")
    exit()

# Obtém a taxa de frames por segundo (FPS) do vídeo
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"FPS do vídeo: {fps}")

frame_count = 0
frame_interval = 30  # Intervalo para capturar frames (1 segundo)

# Loop para ler o vídeo frame por frame
while True:
    ret, frame = cap.read()

    if not ret:
        break  # Fim do vídeo

    # Verifica se o frame atual é o próximo a ser salvo
    if frame_count % frame_interval == 0:
        # Define o caminho para salvar o frame
        frame_filename = os.path.join(frames_folder, f'frame_{frame_count // frame_interval:04d}.jpg')

        # Salva o frame como imagem
        cv2.imwrite(frame_filename, frame)

    frame_count += 1

# Libera o objeto de captura do vídeo
cap.release()

print(f"Frames extraídos e salvos na pasta {frames_folder}")
