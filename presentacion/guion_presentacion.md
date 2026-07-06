# Guion de la presentación oral

**Duración sugerida:** 12–15 minutos de exposición + demo en vivo + preguntas.

## Estructura de diapositivas

1. **Portada** — Título del trabajo, autor, materia.
2. **Problema y objetivo** — Sistema de detección y segmentación en tiempo real de 8
   clases de objetos cotidianos. Por qué importa el "tiempo real" (aplicaciones).
3. **Detección vs. segmentación** — Una imagen ilustrativa con cajas vs. máscaras.
   El sistema resuelve ambas tareas con un único modelo.
4. **Dataset** — COCO 2017; criterio de selección de las 8 clases (exhibibles ante la
   webcam); figura de distribución de instancias por clase; particiones 60/15/25.
5. **Pipeline** — Diagrama de bloques: descarga selectiva → filtrado → formato YOLO →
   fine-tuning en GPU → evaluación → inferencia local en tiempo real.
6. **Modelo** — YOLOv8n-seg en una lámina: one-stage, anchor-free, prototipos de máscara.
   Justificar la variante *nano* (tiempo real sobre CPU).
7. **Data augmentation** — Tabla breve (mosaic, flips, HSV, escala) con una imagen de
   ejemplo del mosaico de entrenamiento (está en `runs/yolov8n_seg_tpf/train_batch0.jpg`).
8. **Entrenamiento** — Curvas de pérdida y de mAP en validación (Notebook 02).
9. **Resultados** — Tabla de mAP@50 / mAP@50-95 (cajas y máscaras) en test; figura por
   clase; matriz de confusión.
10. **Baseline vs. fine-tuned** — Figura comparativa; cuantificar la mejora.
11. **Casos de éxito y de error** — 2–3 imágenes cualitativas comentadas.
12. **DEMO EN VIVO** — (ver checklist abajo).
13. **Conclusiones y trabajo futuro** — 3–4 puntos.

## Checklist de la demo en vivo

Antes de la presentación:

- [ ] Probar la demo completa en la misma computadora y con la misma iluminación que se
      usará en la presentación: `python src/demo_webcam.py`
- [ ] Verificar los FPS; si son bajos, usar la variante OpenVINO o `--imgsz 480`.
- [ ] Preparar los objetos físicos: taza, botella, celular, libro, mouse (el teclado, la
      laptop y la persona ya están en escena).
- [ ] Cerrar aplicaciones que usen la cámara (Zoom, Teams) antes de la demo.
- [ ] **Grabar un video de respaldo** de la demo funcionando y tenerlo en el escritorio:
      si la cámara o la iluminación fallan el día de la presentación, se proyecta el video
      (`python src/demo_webcam.py --source respaldo.mp4` también sirve como demo).
- [ ] Ensayar el guion de la demo (2–3 min): mostrar un objeto por vez, acercarlo y
      alejarlo (robustez a escala), ocluirlo parcialmente (limitación), mostrar dos
      objetos superpuestos (separación de instancias por máscara).

Durante la demo, narrar lo que se ve: clase predicha, confianza, máscara siguiendo el
contorno, FPS en pantalla — conectarlo con las métricas del reporte.
