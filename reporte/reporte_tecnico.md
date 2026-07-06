# Sistema de Detección y Segmentación de Objetos en Tiempo Real mediante Fine-Tuning de YOLOv8

**Trabajo Práctico Final — Deep Learning, Maestría en Inteligencia Artificial**

**Autor:** Federico Morán

---

## Resumen

*(Redactar al final: 150–250 palabras que sinteticen el problema, el método —fine-tuning de YOLOv8n-seg sobre un subconjunto de 8 clases de COCO 2017—, los resultados principales —mAP@50-95 de cajas y máscaras en test, mejora sobre el baseline— y la conclusión sobre la viabilidad de la inferencia en tiempo real sobre CPU.)*

## 1. Introducción

*(Motivación del problema: reconocimiento de objetos de entorno doméstico/oficina en tiempo real. Objetivo general y objetivos específicos. Alcance: 8 clases, demo por webcam. Breve mención del enfoque adoptado y de la estructura del documento.)*

- Objetivo general: desarrollar un sistema completo de visión por computadora que detecte y segmente objetos en tiempo real.
- Objetivos específicos: (i) construir un conjunto de datos a partir de COCO 2017; (ii) ajustar un modelo de segmentación de instancias; (iii) evaluarlo con métricas estándar; (iv) implementar la inferencia en tiempo real sobre hardware de consumo.

## 2. Conjunto de datos

*(Basado en el Notebook 01.)*

- Fuente: COCO 2017, anotaciones de segmentación de instancias.
- Clases seleccionadas y criterio de selección (exhibibles ante una webcam).
- Cantidad de imágenes e instancias; distribución por clase (incluir figura del Notebook 01) y discusión del desbalance.
- Particiones train/val/test (60/15/25), semilla fija, verificación de proporcionalidad entre particiones.
- Conversión al formato YOLO de segmentación (máscaras → polígonos, tolerancia de simplificación) y verificación visual de la exportación.

## 3. Metodología

*(Basado en el Notebook 02.)*

### 3.1 Arquitectura

*(YOLOv8n-seg: backbone CSPDarknet/C2f, neck PAN-FPN, cabeza desacoplada con prototipos de máscara. Justificación de la variante nano por el requisito de tiempo real sobre CPU.)*

### 3.2 Preprocesamiento y aumento de datos

*(Letterbox 640×640 y normalización. Tabla de transformaciones con valores y justificación: mosaic con close_mosaic, volteo horizontal, traslación, escala, perturbación HSV; transformaciones omitidas y por qué.)*

### 3.3 Configuración de entrenamiento

*(Transferencia desde pesos COCO; fine-tuning completo. Épocas, early stopping, batch automático, optimizador, semilla. Hardware utilizado: GPU del servidor. Duración del entrenamiento.)*

## 4. Experimentos y resultados

*(Basado en los Notebooks 02 y 03.)*

### 4.1 Curvas de aprendizaje

*(Figura de pérdidas train/val y de evolución del mAP en validación. Discutir convergencia, ausencia/presencia de sobreajuste y efecto de close_mosaic.)*

### 4.2 Métricas sobre la partición de prueba

| Métrica | Cajas (detección) | Máscaras (segmentación) |
|---|---|---|
| Precisión media | *(completar)* | *(completar)* |
| Recall medio | *(completar)* | *(completar)* |
| mAP@50 | *(completar)* | *(completar)* |
| mAP@50-95 | *(completar)* | *(completar)* |

*(Tabla por clase + figura. Analizar mejores/peores clases y sus causas: tamaño del objeto, oclusión, cantidad de ejemplos.)*

### 4.3 Matriz de confusión

*(Figura. Confusiones sistemáticas entre clases y proporción de objetos no detectados.)*

### 4.4 Comparación con el modelo de referencia

*(Tabla y figura baseline vs. fine-tuned por clase. Cuantificar la mejora. Incluir la salvedad metodológica: el baseline resuelve un problema de 80 clases, por lo que parte de la mejora se debe a la reducción del vocabulario.)*

### 4.5 Análisis cualitativo

*(2–4 figuras: casos de éxito y casos de error representativos —oclusiones, objetos pequeños, instancias superpuestas—. Comentar cada figura.)*

## 5. Sistema de inferencia en tiempo real

*(Basado en src/demo_webcam.py.)*

- Descripción del sistema: captura por webcam (OpenCV), inferencia, superposición de cajas/máscaras/FPS.
- Latencia medida: ___ ms/imagen en GPU (servidor); ___ FPS en CPU local con pesos .pt; ___ FPS con la exportación OpenVINO. *(Completar con mediciones reales.)*
- Discusión del compromiso precisión–latencia y del efecto de la resolución de inferencia (`imgsz`).

## 6. Conclusiones y trabajo futuro - Aqui quiero evitar trabajos futuros, solo poner conclusiones. 

*(Síntesis de los aportes y resultados. Limitaciones observadas. Trabajo futuro: más clases, variantes de mayor capacidad, cuantización INT8, seguimiento de objetos (tracking), despliegue en dispositivos de borde.)*

## Referencias

1. Lin, T.-Y. et al. (2014). *Microsoft COCO: Common Objects in Context*. ECCV 2014.
2. Jocher, G., Chaurasia, A., & Qiu, J. (2023). *Ultralytics YOLOv8*. https://github.com/ultralytics/ultralytics
3. Bolya, D. et al. (2019). *YOLACT: Real-time Instance Segmentation*. ICCV 2019.
4. Redmon, J. et al. (2016). *You Only Look Once: Unified, Real-Time Object Detection*. CVPR 2016.

---

*Anexos: notebooks ejecutados (01–03) y código fuente en `src/`.*
