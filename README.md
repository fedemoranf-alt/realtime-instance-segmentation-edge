# TPF — Sistema de Detección y Segmentación de Objetos en Tiempo Real

**Maestría en Inteligencia Artificial — Deep Learning**

Sistema completo de visión por computadora: ajuste fino (*fine-tuning*) de **YOLOv8n-seg**
sobre un subconjunto de 8 clases de **COCO 2017** (`person`, `cell phone`, `cup`, `bottle`,
`laptop`, `keyboard`, `mouse`, `book`), con evaluación mediante métricas estándar
(mAP, IoU) y demostración de inferencia en tiempo real por webcam.

## Estructura del proyecto

```
TPF - Computer Vision/
├── Indicaciones.txt            Consigna del trabajo
├── README.md                   Este archivo
├── requirements.txt            Dependencias
├── notebooks/
│   ├── 01_dataset.ipynb        Descarga del subset COCO, EDA, particiones, formato YOLO
│   ├── 02_entrenamiento.ipynb  Augmentation + fine-tuning de YOLOv8n-seg  [GPU]
│   └── 03_evaluacion.ipynb     mAP/IoU en test, matriz de confusión, baseline  [GPU]
├── src/
│   ├── demo_webcam.py          Inferencia en tiempo real (webcam/video) con FPS
│   └── export_openvino.py      Exportación opcional a OpenVINO (CPU Intel)
├── data/                       Dataset en formato YOLO (generado; NO se sube ni entrega)
├── models/                     Pesos entrenados (yolov8n_seg_best.pt) — entregable
├── runs/                       Salidas de entrenamiento/evaluación (generado)
├── reporte/                    Reporte técnico
└── presentacion/               Guion de la presentación oral y checklist de la demo
```

## Flujo de trabajo

El proyecto se ejecuta en **dos entornos**: el servidor con GPU (entrenamiento) y la
computadora personal (demostración en tiempo real).

### A. En el servidor GPU (JupyterLab)

1. **Subir el proyecto completo** (carpeta entera, sin `data/` ni `runs/`, que aún no
   existen o se regeneran). Los notebooks usan rutas relativas a la raíz del proyecto,
   por lo que la estructura de carpetas debe conservarse.
2. Instalar dependencias: `pip install -r requirements.txt`
3. Ejecutar en orden:
   - `notebooks/01_dataset.ipynb` — descarga el subset de COCO **directamente en el
     servidor** (~1 GB; por eso no conviene subir datos desde la PC local).
   - `notebooks/02_entrenamiento.ipynb` — fine-tuning en GPU (~15–30 min en una RTX 5090).
   - `notebooks/03_evaluacion.ipynb` — métricas sobre test y comparación con el baseline.
4. **Descargar de vuelta** (lo demás se regenera y no hace falta):
   - `models/yolov8n_seg_best.pt` — el modelo entrenado (entregable),
   - `runs/` — curvas, matriz de confusión y figuras para el reporte,
   - los tres notebooks **ejecutados** (con sus salidas), que son parte del entregable.

### B. En la computadora local (demo)

```bash
pip install -r requirements.txt

# Demo en tiempo real con la webcam (tecla q para salir)
python src/demo_webcam.py

# Opcional: variante optimizada para CPU Intel (más FPS)
python src/export_openvino.py
python src/demo_webcam.py --model models/yolov8n_seg_best_openvino_model
```

Si los FPS resultan bajos, reducir la resolución de inferencia: `--imgsz 480` o `--imgsz 320`.

## Correspondencia con la consigna

| Requisito | Dónde se cumple |
|---|---|
| Detección y/o segmentación | YOLOv8n-seg ajustado: cajas + máscaras de instancia |
| Preprocesamiento y data augmentation | Notebook 01 (filtrado, particiones) y Notebook 02 §3 |
| Métricas estándar (mAP, IoU) | Notebook 03: mAP@50 y mAP@50-95 de cajas y máscaras |
| Inferencia en tiempo real | `src/demo_webcam.py` (FPS en pantalla) |
| Pipeline completo | Notebooks 01→03 + scripts de `src/` |
| Modelo entrenado | `models/yolov8n_seg_best.pt` |
| Reporte técnico | `reporte/reporte_tecnico.md` |
| Presentación oral con demo | `presentacion/guion_presentacion.md` + demo webcam |
