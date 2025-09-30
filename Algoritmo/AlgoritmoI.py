# Importamos librerías necesarias
from ultralytics import YOLO  # Framework YOLOv8
import cv2 # Para cargar y manipular imágenes
import matplotlib.pyplot as plt

# 1. Cargar modelo preentrenado
# Usamos un modelo pre-entrenado
model = YOLO("yolov8n.pt")

# 2. Cargar imagen de entrada
image_path = "C:/Users/aggro/OneDrive/Desktop/Deteccion/Algoritmo/CarreraCaballos Detección ML.jfif"
image = cv2.imread(image_path)

# Convertimos a formato RGB (OpenCV carga en BGR por defecto)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 3. Realizar detección con YOLO
results = model(image_rgb)

# 4. Filtrar detecciones solo de "person" (jinete)
jinetes = []
for r in results[0].boxes:
    cls_id = int(r.cls[0])  # ID de la clase
    label = model.names[cls_id]  # Nombre de la clase
    if label == "person":
        # Guardamos coordenadas del bounding box (x1, y1, x2, y2)
        x1, y1, x2, y2 = r.xyxy[0].tolist()
        jinetes.append((x1, y1, x2, y2))

# 5. Seleccionar solo el jinete delantero
# Definimos al "jinete delantero" como el que está más abajo en la imagen (mayor y2)
if jinetes:
    jinete_delantero = max(jinetes, key=lambda box: box[3])  # mayor y2 (parte inferior del bounding box)
    x1, y1, x2, y2 = map(int, jinete_delantero)

    # Dibujar solo este bounding box en la imagen
    cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (255, 0, 0), 3)  # azul
    cv2.putText(image_rgb, "Jinete Delantero", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# 6. Mostrar resultado final
plt.figure(figsize=(10, 6))
plt.imshow(image_rgb)
plt.axis("off")
plt.title("Detección del Jinete Delantero con YOLOv8")
plt.show()

# 7. Guardar resultado en archivo
cv2.imwrite("resultado.jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
print("Imagen guardada como resultado.jpg")
