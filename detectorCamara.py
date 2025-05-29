import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
import ttkbootstrap as tb
from ultralytics import YOLO
import threading
from PIL import Image, ImageTk

# Cargar modelo YOLOv8
modelo = YOLO("best.pt")  # Asegúrate de tener el modelo en la misma carpeta

# Variables globales
captura = None
deteniendo = False
video_thread = None

# Obtener cámaras disponibles
def listar_camaras():
    index = 0
    camaras = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            break
        camaras.append(f"Cámara {index}")
        cap.release()
        index += 1
    return camaras

# Función para iniciar la captura de video
def mostrar_video():
    global captura, deteniendo, video_thread

    cam_id = selected_camera.get()

    if isinstance(cam_id, str) and cam_id.isdigit():
        cam_id = int(cam_id)  # Convertir a entero si es necesario

    captura = cv2.VideoCapture(cam_id)

    if not captura.isOpened():
        status_label.config(text="❌ Error al abrir la cámara", foreground="red")
        return

    status_label.config(text="🟢 Cámara activa", foreground="green")

    # Iniciar el hilo para actualizar el video
    video_thread = threading.Thread(target=actualizar_frame, daemon=True)
    video_thread.start()

# Función para actualizar el frame en la interfaz
def actualizar_frame():
    global deteniendo, captura

    if captura is None or not captura.isOpened():
        status_label.config(text="❌ Cámara no iniciada", foreground="red")
        return
    print("Clases detectadas:", modelo.names)
    colores_personalizados = {
        " Maduro": (0, 0, 225),     
        "Inmaduro": (0, 255, 0),    
        "Mal estado": (0, 255, 255)  
    }

    while not deteniendo:
        ret, frame = captura.read()
        if not ret:
            break

        resultados = modelo(frame)
        conf_minima = confianza_var.get() / 100.0  

        for r in resultados:
            for box in r.boxes:
                conf = box.conf[0]
                if conf < conf_minima:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                clase = int(box.cls[0])
                nombre_clase = modelo.names[clase]

                color = colores_personalizados.get(nombre_clase, (255, 255, 255))

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                etiqueta = f"{nombre_clase}: {conf:.2f}"
                cv2.putText(frame, etiqueta, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imagen = Image.fromarray(frame)
        img_tk = ImageTk.PhotoImage(image=imagen)

        video_label.imgtk = img_tk
        video_label.config(image=img_tk)

    if captura is not None:
        captura.release()

    status_label.config(text="⏹ Cámara detenida", foreground="black")

# Función para detener la detección y liberar la cámara
def detener_deteccion():
    global deteniendo, captura
    deteniendo = True

    if captura is not None and captura.isOpened():
        captura.release()
        captura = None

    status_label.config(text="⏹ Cámara detenida", foreground="black")
    deteniendo = False

# Función para reiniciar el video
def reiniciar_video():
    global deteniendo, captura, video_thread

    detener_deteccion()

    deteniendo = False
    video_thread = None
    captura = None

    mostrar_video()

# Crear ventana principal con ttkbootstrap
root = tb.Window(themename="superhero")
root.title("Detección de Ají Rojo")
root.geometry("800x500")
root.resizable(False, False)

# Variables de control
selected_camera = tk.IntVar(value=0)
confianza_var = tk.IntVar(value=50)

# Crear marco principal
frame_principal = ttk.Frame(root)
frame_principal.pack(fill="both", expand=True, padx=10, pady=10)

# Controles de cámara
frame_columna1 = ttk.Frame(frame_principal, width=250, relief="sunken")
frame_columna1.pack(side="left", fill="y", padx=5)

frame_columna2 = ttk.Frame(frame_principal, relief="sunken")
frame_columna2.pack(side="left", fill="both", expand=True, padx=5)

# Selección de cámara
ttk.Label(frame_columna1, text="Selecciona la Cámara:", font=("Arial", 11)).pack(padx=10)
camaras_disponibles = listar_camaras()
camera_menu = ttk.Combobox(frame_columna1, textvariable=selected_camera, values=list(range(len(camaras_disponibles))), state="readonly")
camera_menu.pack(padx=10)
camera_menu.current(0)

# Slider de confianza
ttk.Label(frame_columna1, text="Confianza mínima:", font=("Arial", 11)).pack(padx=10)
slider_confianza = ttk.Scale(frame_columna1, from_=0, to=100, orient="horizontal", variable=confianza_var, command=lambda e: actualizar_confianza_label())
slider_confianza.pack(padx=10)

confianza_label = ttk.Label(frame_columna1, text=f"Confianza: {confianza_var.get()}%", font=("Arial", 10))
confianza_label.pack(padx=10)

def actualizar_confianza_label():
    confianza_label.config(text=f"Confianza: {confianza_var.get()}%")

# Estado de la cámara
status_label = ttk.Label(frame_columna1, text="⏹ Cámara detenida", font=("Arial", 12))
status_label.pack(pady=5)

# Botones de control
frame_botones = ttk.Frame(frame_columna1)
frame_botones.pack(pady=10)

btn_iniciar = tb.Button(frame_botones, text="▶ Iniciar Video", bootstyle="success", command=reiniciar_video)
btn_iniciar.pack(side="left", padx=10)

btn_detener = tb.Button(frame_botones, text="⏹ Detener Video", bootstyle="danger", command=detener_deteccion)
btn_detener.pack(side="left", padx=10)

# Etiqueta para el video
video_label = ttk.Label(frame_columna2)
video_label.pack(pady=10)

# Iniciar interfaz gráfica
root.mainloop()
