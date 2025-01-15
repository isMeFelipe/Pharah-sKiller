from ultralytics import YOLO
import cv2
import numpy as np
import mss
import time
import os
import threading
import torch
from queue import Queue

# Carregar o modelo YOLO
model = YOLO('../runs/detect/train18/weights/best.pt')

# Verificar se há GPU disponível e mover o modelo para a GPU com FP16
if torch.cuda.is_available():
    model = model.half().cuda()

# Configurar captura de tela
sct = mss.mss()
screen_region = sct.monitors[1]  # Captura da tela principal

# Definir o diretório de saída
output_dir = "frames_output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)  # Cria o diretório se não existir

# Definir resolução inicial para captura (pode ser ajustada dinamicamente)
screen_width, screen_height = 1280, 720

# Variáveis de controle
frame_lock = threading.Lock()
frame_queue = Queue(maxsize=1)  # Fila para compartilhar frames entre threads
save_interval = 5  # Intervalo de salvamento em segundos
last_saved_time = time.time()
target_fps = 30  # FPS desejado

# Função para capturar a tela (sem thread)
def capture_screen():
    screenshot = sct.grab(screen_region)
    frame = np.array(screenshot)
    frame = cv2.resize(frame, (screen_width, screen_height))
    
    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # Converte para BGR

    return frame

# Função para processar o frame com YOLO e obter a posição dos objetos
def process_frame(frame):
    results = model(frame, verbose=False)

    for result in results:
        for box in result.boxes:
            # Coordenadas da caixa delimitadora
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            label = result.names[int(box.cls[0])]

            # Cálculo do centro da caixa delimitadora
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Exibir as coordenadas no console
            print(f'Objeto: {label} | Confiança: {conf:.2f} | Centro: ({center_x}, {center_y})')

            # Desenhar a caixa e a posição central no frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)  # Marca o centro
            cv2.putText(frame, f'{label} ({center_x}, {center_y})', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    return frame

# Função para salvar o frame processado
def save_frame(frame):
    global last_saved_time
    if time.time() - last_saved_time >= save_interval:
        frame_filename = f'{output_dir}/output_frame_{int(time.time())}.jpg'
        cv2.imwrite(frame_filename, frame)
        last_saved_time = time.time()

# Ajustar a resolução conforme o FPS
def adaptive_resize(current_fps):
    global screen_width, screen_height
    if current_fps < target_fps - 5:
        screen_width = max(640, screen_width - 100)
        screen_height = max(360, screen_height - 56)
    elif current_fps > target_fps + 5:
        screen_width = min(1920, screen_width + 100)
        screen_height = min(1080, screen_height + 56)

# Função principal para captura e processamento
def main():
    global frame_queue
    while True:
        # Captura a tela na thread principal
        frame = capture_screen()

        # Coloca o frame na fila para processamento
        if not frame_queue.full():
            frame_queue.put(frame)

        # Verifica se há um frame na fila para processar
        if not frame_queue.empty():
            frame = frame_queue.get()
            start_time = time.time()

            # Processar e salvar o frame
            processed_frame = process_frame(frame)
            save_frame(processed_frame)

            # Calcular FPS
            if start_time != time.time():
                fps = 1 / (time.time() - start_time)
                adaptive_resize(fps)
                print(f"FPS: {fps:.2f}")

            # Aguarda um pouco para não sobrecarregar a CPU
            time.sleep(0.01)

if __name__ == "__main__":
    main()
