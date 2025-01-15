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

# Variável para armazenar as posições anteriores
object_positions = {}

# Tempo entre as capturas (50ms, já configurado no canvas.after)
TIME_BETWEEN_FRAMES = 0.5  # segundos

# Número de frames à frente para estimar
FRAMES_A_FRENTE = 5  # Estimar 5 frames à frente

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
            positions.append((center_x, center_y, label))

    return positions

# Função para calcular a distância entre duas posições
def calculate_speed(prev_x, prev_y, last_x, last_y):
    # Calculando a distância Euclidiana entre as posições
    distance = np.sqrt((last_x - prev_x) ** 2 + (last_y - prev_y) ** 2)
    
    # Velocidade = distância / tempo (em segundos)
    speed = distance / TIME_BETWEEN_FRAMES  # em pixels por segundo
    return speed

# Função para calcular a posição estimada *X* frames à frente
def estimate_position_ahead(last_x, last_y, prev_x, prev_y, frames_ahead):
    # Calcula o deslocamento entre as duas últimas posições
    delta_x = last_x - prev_x
    delta_y = last_y - prev_y

    # Estima a nova posição considerando o deslocamento multiplicado pelos frames à frente
    predicted_x = last_x + delta_x * frames_ahead
    predicted_y = last_y + delta_y * frames_ahead
    
    return predicted_x, predicted_y

# Função para calcular e desenhar a direção do movimento e a estimativa
def draw_direction(canvas, positions):
    global object_positions

    # Limpar a tela
    canvas.delete('all')  # Deleta todos os desenhos anteriores

    for center_x, center_y, label in positions:
        radius = 10  # Tamanho da bola

        # Mover a bola 15 pixels para cima manualmente
        adjusted_center_y = center_y - 15  # Ajusta a posição para cima

        # Verifica se o objeto já foi detectado antes
        if label not in object_positions:
            object_positions[label] = []

        # Adiciona a nova posição ao histórico
        object_positions[label].append((center_x, adjusted_center_y))

        # Limita o histórico a um número máximo de posições
        if len(object_positions[label]) > 10:  # Ajuste o número conforme necessário
            object_positions[label].pop(0)

        # Desenha a trajetória (linhas conectando as posições anteriores)
        for i in range(1, len(object_positions[label])):
            prev_x, prev_y = object_positions[label][i - 1]
            canvas.create_line(prev_x, prev_y, object_positions[label][i][0], object_positions[label][i][1], fill="blue", width=2)

        # Estimativa da próxima posição *X* frames à frente
        if len(object_positions[label]) > 1:
            # Pega as duas últimas posições
            prev_x, prev_y = object_positions[label][-2]
            last_x, last_y = object_positions[label][-1]
            
            # Estima a posição *X* frames à frente
            predicted_x, predicted_y = estimate_position_ahead(last_x, last_y, prev_x, prev_y, FRAMES_A_FRENTE)

            # Calcula a velocidade do objeto
            speed = calculate_speed(prev_x, prev_y, last_x, last_y)

            # Desenha a estimativa da próxima posição
            canvas.create_oval(predicted_x - radius, predicted_y - radius, predicted_x + radius, predicted_y + radius, outline="yellow", width=5)
            canvas.create_text(predicted_x, predicted_y - 15, text=f"{speed:.2f} px/s", fill="yellow", font=("Helvetica", 8))

        # Desenha o círculo ajustado (posição atual)
        canvas.create_oval(center_x - radius, adjusted_center_y - radius, center_x + radius, adjusted_center_y + radius, fill="red")

# Função que será chamada para atualizar a interface gráfica no loop principal
def update_canvas(canvas):
    frame = capture_screen()
    positions = process_frame(frame)
    draw_direction(canvas, positions)  # Atualiza a posição na sobreposição
    canvas.after(50, update_canvas, canvas)  # Chama a função novamente após 50ms

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

    # Inicia o processo de atualização do canvas no loop principal
    update_canvas(canvas)

    # Inicia o loop de eventos do Tkinter
    root.mainloop()

if __name__ == "__main__":
    overlay_window()
