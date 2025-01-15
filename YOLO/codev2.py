import tkinter as tk
import time
import threading
import mss
import numpy as np
import torch
import cv2  # Importando OpenCV
from ultralytics import YOLO
from queue import Queue

# Carregar o modelo YOLO
model = YOLO('../runs/detect/train18/weights/best.pt')

# Variáveis de controle
screen_width, screen_height = 2560, 1080
frame_queue = Queue(maxsize=1)

# Função para capturar a tela
def capture_screen():
    with mss.mss() as sct:
        screen_region = sct.monitors[1]  # Captura da tela principal
        screenshot = sct.grab(screen_region)
        frame = np.array(screenshot)
        frame = cv2.resize(frame, (screen_width, screen_height))  # Redimensiona a imagem
        
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # Converte para BGR

        return frame

# Função para processar o frame com YOLO
def process_frame(frame):
    results = model(frame, verbose=False)

    positions = []  # Lista para armazenar as posições dos objetos detectados

    for result in results:
        for box in result.boxes:
            # Coordenadas da caixa delimitadora
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Nome do objeto detectado
            label = result.names[int(box.cls[0])]
            
            # Exibe no console a detecção e a posição
            print(f'Detecção: {label} | Posição: ({center_x}, {center_y})')

            # Adicionar a posição do centro ao resultado
            positions.append((center_x, center_y))

    return positions

# Função para desenhar as posições na janela overlay
# Função para desenhar as posições na janela overlay
def draw_overlay(canvas, positions):
    # Limpar a tela
    canvas.delete('all')  # Deleta todos os desenhos anteriores

    # Desenhar círculos maiores para cada posição detectada
    for center_x, center_y in positions:
        radius = 10  # Novo tamanho da bola (10 é o raio)

        # Mover a bola 15 pixels para cima manualmente
        adjusted_center_y = center_y - 15  # Ajusta a posição para cima (centralziar melhor)

        # Desenhar o círculo ajustado
        canvas.create_oval(center_x - radius, adjusted_center_y - radius, center_x + radius, adjusted_center_y + radius, fill="red")


# Função para exibir o overlay com as posições detectadas
def overlay_window():
    # Criação da janela principal
    root = tk.Tk()
    root.title("Tela Principal")
    root.geometry(f"{screen_width}x{screen_height}")

    # Criar a janela de overlay
    overlay = tk.Toplevel(root)
    overlay.title("Overlay")
    overlay.geometry(f"{screen_width}x{screen_height}")
    overlay.config(bg="black")
    overlay.attributes('-topmost', 1)  # Garante que a janela ficará no topo

    # Tornando o fundo transparente
    overlay.attributes('-transparentcolor', 'black')  # Definindo a cor preta como transparente no Windows

    overlay.withdraw()  # Começa com a janela oculta

    # Tela de desenho da sobreposição
    canvas = tk.Canvas(overlay, width=screen_width, height=screen_height, bg='black')
    canvas.pack()

    overlay.deiconify()  # Exibe a janela overlay

    # Função que vai capturar a tela, processar e desenhar a sobreposição
    def process_and_draw():
        while True:
            frame = capture_screen()
            positions = process_frame(frame)
            draw_overlay(canvas, positions)  # Atualiza a posição na sobreposição
            time.sleep(0.1)  # Aguarda um pouco para o próximo frame

    # Criar uma thread para processar a captura e desenhar a sobreposição
    thread = threading.Thread(target=process_and_draw)
    thread.daemon = True
    thread.start()

    # Inicia o loop de eventos do Tkinter
    root.mainloop()

if __name__ == "__main__":
    overlay_window()
