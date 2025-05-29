# 🌶️ Detección de Maduración de Ají Rojo con YOLOv8 y OpenCV

Este proyecto utiliza **YOLOv8**, **OpenCV** y **Tkinter** con `ttkbootstrap` para detectar en tiempo real el estado de maduración de ajíes rojos a través de la cámara del computador.  
Las clases detectadas incluyen:  
🔴 **Maduro**, 🟢 **Inmaduro**, 🟡 **Mal estado**

---

## 🧠 ¿Cómo funciona?

- El modelo `YOLOv8` (entrenado y guardado como `best.pt`) detecta los ajíes y clasifica su estado.
- Se utiliza la cámara web para realizar la detección en vivo.
- La interfaz gráfica te permite:
  - Seleccionar la cámara.
  - Ajustar el nivel de confianza.
  - Ver en tiempo real los resultados con colores personalizados por clase.

---

## 🚀 Requisitos e instalación

> 🐍 **Este proyecto requiere Python 3.12**

### 📦 Dependencias

Instálalas fácilmente con `pip`:

```bash
pip install opencv-python
pip install numpy
pip install pillow
pip install ttkbootstrap
pip install ultralytics
