Actividad Deteccion Machine Learning

El algoritmo logró identificar al jinete que se encuentra en la delantera dentro de la carrera de caballos. Utilizando el modelo de detección YOLOv8, primero se detectaron todas las personas (jinetes) presentes en la imagen y, posteriormente, se aplicó un criterio adicional que selecciona únicamente al competidor que aparece más adelantado, definido como aquel cuya caja delimitadora (bounding box) llega más abajo en la imagen (coordenada y2 más grande).

El resultado final es una imagen procesada donde únicamente el jinete delantero está marcado con un rectángulo azul y la etiqueta “Jinete Delantero”, descartando al resto de los jinetes detectados.




Implementación del Código

Carga de librerías
Se utilizaron ultralytics para YOLOv8, cv2 para procesar imágenes y matplotlib para la visualización.

Modelo preentrenado
Se usó el modelo yolov8n.pt, una versión ligera de YOLOv8, capaz de detectar múltiples clases de objetos.

Lectura de la imagen
La imagen de la carrera se cargó con OpenCV y se transformó de formato BGR a RGB para su correcta visualización.

Detección de objetos con YOLOv8
El modelo predijo los objetos presentes en la imagen, incluyendo personas (jinetes).

Filtrado de la clase "person"
Se extrajeron únicamente las detecciones correspondientes a la clase person, que representan a los jinetes.

Selección del jinete delantero
Entre todos los jinetes detectados, se eligió al que está más adelante en la carrera, definido como el que tiene la coordenada vertical inferior (y2) más grande.

Visualización y guardado
Sobre el jinete delantero se dibujó un rectángulo azul con la etiqueta “Jinete Delantero”. La imagen final fue mostrada en pantalla y guardada como resultado.jpg.
