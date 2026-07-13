# Handoff — TPF Deep Learning: Detección y Segmentación en Tiempo Real

> **Propósito de esta nota.** Resumen de estado del proyecto para retomarlo con contexto
> completo (por Federico o por un agente como Claude). Reemplaza a la antigua
> `PENDIENTE_data_leakage.md`, cuyo problema ya está **resuelto** (ver más abajo).
> Última actualización: **2026-07-10**.

---

## 1. Qué es el proyecto

Sistema de visión por computadora que **detecta y segmenta 8 clases** de objetos de
escritorio/hogar en tiempo real, mediante **ajuste fino (fine-tuning) de YOLOv8n-seg**
sobre un subconjunto de COCO 2017.

- **Clases:** `person`, `cell phone`, `cup`, `bottle`, `laptop`, `keyboard`, `mouse`, `book`.
- **Criterio:** todas se pueden mostrar ante una webcam en la demo en vivo.
- **Entorno de trabajo:** servidor con GPU RTX 5090 para entrenar/evaluar; PC personal
  (CPU Intel Core Ultra 5, sin GPU) para la demo en tiempo real.

**Estado general: COMPLETO.** El reporte técnico y la presentación oral están terminados y
compilados. Queda un único paso opcional de tu lado (re-ejecutar el Notebook 03, ver §6).

---

## 2. Resultados finales (métricas mAP@50-95)

| | Cajas | Máscaras |
|---|---|---|
| **Modelo ajustado, test train2017** (partición reservada) | 0,292 | 0,246 |
| **Baseline preentrenado, val2017** (comparación justa) | 0,329 | 0,277 |
| **Modelo ajustado, val2017** (comparación justa) | 0,275 | 0,236 |

- **Velocidad (no depende de los pesos):** 17 FPS con PyTorch y **35 FPS con OpenVINO** en
  la CPU de la PC personal; ~198 FPS en la GPU del servidor.
- **Conclusión honesta:** sobre un vocabulario que es *subconjunto* del preentrenamiento y
  con 40× menos datos del mismo dominio, el modelo ajustado **iguala** al preentrenado en
  una comparación justa; su aporte real es la **especialización** (no dispara clases fuera
  de vocabulario), no un incremento de mAP. Ver §4.

---

## 3. Los tres problemas que se encontraron y cómo se resolvieron

Estos problemas se descubrieron y corrigieron durante el desarrollo. Los notebooks de
`notebooks/` ya contienen las correcciones y **guardas** que impiden que vuelvan a ocurrir
en silencio.

### 3.1 Fuga de datos (data leakage) — RESUELTO
- **Qué pasaba:** FiftyOne etiqueta cada muestra descargada con su *split* de origen
  (`train`, por venir de COCO train2017). Esa etiqueta colisionaba con la partición propia,
  así que `match_tags('train')` devolvía las 5.000 imágenes en vez de las 3.000 previstas.
  El modelo entrenaba sobre val y test.
- **Fix (en `notebooks/01_dataset.ipynb`):** `dataset.untag_samples([...])` antes de
  `random_split`.

### 3.2 Export fusionado de FiftyOne — RESUELTO
- **Qué pasaba:** el exportador `YOLOv5Dataset` **fusiona por defecto** con lo que haya en
  el directorio. Sin borrarlo antes, `images/train/` quedaba con las 3.000 nuevas + las
  2.000 viejas (= val + test). El notebook imprimía `3000/750/1250` (correcto a nivel de
  etiquetas de FiftyOne) pero YOLO leía 5.000 del disco. **Señal de alarma:** el log de
  Ultralytics dice `train: Scanning ... 5000 images`.
- **Fix:** `shutil.rmtree(DATA_DIR)` antes de exportar, **más un `assert` a nivel de
  ARCHIVOS** que verifica que las particiones exportadas sean disjuntas y midan
  `3000/750/1250` (que es lo que el entrenador realmente lee). También hay una guarda en
  `notebooks/02_entrenamiento.ipynb` que aborta si `train != 3000` y borra la corrida previa.

### 3.3 Cabeza de clasificación reinicializada — RESUELTO
- **Qué pasaba:** al reducir el vocabulario de 80 a 8 clases, Ultralytics **reinicializa**
  la rama de clasificación de la cabeza (`Transferred 381/417 items`: 36 tensores,
  ~372.000 parámetros desde cero). Un ajuste fino *completo* de esa cabeza sobre solo 3.000
  imágenes **degradaba** el modelo por debajo del preentrenado.
- **Fix (en `notebooks/02_entrenamiento.ipynb`):** congelar el *backbone* (`freeze=10`) para
  preservar las representaciones de COCO y concentrar el ajuste en el *neck* y la cabeza,
  con más épocas (`epochs=150`, early stopping `patience=50`, se detuvo en la 140).
- La corrida de ajuste completo se conservó como **ablación** (ver §5, `runs/*_full_ft`).

---

## 4. La pieza metodológica clave: evaluación imparcial sobre val2017

Aun con los datos limpios, comparar el modelo ajustado contra el preentrenado sobre nuestra
partición de prueba **favorece injustamente al baseline**: el modelo preentrenado
`yolov8n-seg.pt` se entrenó sobre COCO **train2017**, el mismo pozo del que salió nuestro
test. Es decir, el baseline **ya había visto nuestras imágenes de prueba** durante su
preentrenamiento.

**Corrección:** se construyó un test independiente desde **COCO val2017** (el set de
validación oficial de COCO, disjunto de train2017 y no visto por ninguno de los dos
modelos) y se evaluó a ambos ahí. Resultado: el baseline **cae de 0,342 a 0,277** (máscaras)
al pasar a datos no vistos, mientras que el ajustado apenas varía (0,246 → 0,236). **Dos
tercios de la brecha aparente eran la ventaja de fuga del baseline.** El código está en la
**sección 5 de `notebooks/03_evaluacion.ipynb`**.

---

## 5. Estado de los entregables

| Entregable | Ubicación | Estado |
|---|---|---|
| Reporte técnico (fuente) | `reporte/informe.tex` | ✅ Reescrito. Compila a 19 pág. con `pdflatex` (2 pasadas). |
| Reporte técnico (PDF entregable) | `reporte/TPF - DL - Federico Moran.pdf` | ✅ Copia manual de `informe.pdf` (ver gotcha §7). |
| Presentación oral (fuente) | `presentacion/presentacion.tex` | ✅ Beamer, 34 pág., 0 pendientes. |
| Presentación oral (PDF) | `presentacion/presentacion.pdf` | ✅ Compilada. |
| Figuras del reporte | `reporte/figuras/` | ✅ Todas regeneradas con la corrida limpia + eval val2017. |
| Modelo entrenado (entregable) | `models/yolov8n_seg_best.pt` | ✅ Backbone congelado, corrida limpia. |
| Notebooks del pipeline | `notebooks/01`, `02`, `03` | ✅ Con correcciones y guardas. **03 tiene la §5 sin ejecutar** (ver §6). |
| Demo en tiempo real | `src/demo_webcam.py`, `src/export_openvino.py` | ✅ Sin cambios. |
| Demo en la nube (Gradio/HF) | `demo_nube/` | ⚠️ Federico lo actualiza a mano. **No tocar.** |

**Artefactos de respaldo / ablación:**
- `models/yolov8n_seg_full_ft.pt` + `runs/yolov8n_seg_tpf_full_ft/` + `runs/eval_test_full_ft/`
  — corrida de ajuste fino *completo* (sin congelar), usada como ablación en el reporte.
- `runs/fair_baseline/`, `runs/fair_finetuned/` — evaluaciones sobre val2017.
- `data/coco_val2017_fair/`, `data/coco_val2017_fair_80/` — test imparcial (val2017),
  generado localmente (8 y 80 clases).

---

## 6. Qué queda pendiente / cómo continuar

**Único pendiente (opcional, del lado de Federico):** re-ejecutar `notebooks/03_evaluacion.ipynb`
para que su **sección 5 (evaluación imparcial en val2017)** quede con salidas ejecutadas. El
reporte y las figuras ya están hechos con una corrida local, así que esto es solo para que
el notebook entregado sea coherente. Corre en CPU (~5 min) o rápido en el servidor.

**Si hubiera que regenerar el pipeline completo desde cero (servidor GPU):**
1. Subir `notebooks/` corregidos. Borrar `runs/` a mano en el servidor.
2. Ejecutar `01 → 02 → 03`. El Notebook 01 **debe imprimir** `train 3000 / val 750 / test 1250`
   y su celda de verificación debe decir "OK — particiones disjuntas". Si no, parar.
3. El Notebook 02 debe imprimir `Particiones verificadas: train=3000` y el log de Ultralytics
   `train: Scanning ... 3000 images`. Si dice 5000, algo se rompió (ver §3.2).
4. Bajar `models/`, `runs/` y los notebooks ejecutados.
5. Regenerar figuras del reporte y recompilar (ver gotchas §7).
6. Analizar si hace falta cambiar el modelo que esta implementado en hugging face.
7. Volver a generar el modelo openvino con el ultimo modelo entrenado. 

**Ideas de trabajo futuro (documentadas en el reporte):** ampliar el dataset (clases raras
como `keyboard`/`mouse` tienen <150 ejemplos de train), probar YOLOv8s-seg, atacar el recall
de instancias pequeñas (tiling/mayor resolución), cuantización INT8 con OpenVINO. La única
vía para **superar** genuinamente al baseline sería entrenar con datos del dominio real
(imágenes de webcam), no con más COCO.

---

## 7. Gotchas y notas para el próximo agente

- **PDF entregable del reporte:** `reporte/TPF - DL - Federico Moran.pdf` es una **copia
  manual** de `informe.pdf`. Tras cada recompilación hay que re-copiarlo:
  `cp -f informe.pdf "TPF - DL - Federico Moran.pdf"`.
- **Compilar LaTeX:** `pdflatex` de MiKTeX, 2 pasadas. Beamer y `listings` funcionan.
- **Figuras:** no había script original; se regeneran con matplotlib desde `runs/` y desde
  el dataset. El *cualitativo* y la *verificación de etiquetas* se generaron sobre val2017
  (que está disponible localmente); las de datos, desde `results.csv` y las métricas.
- **`Entregables/`** (carpeta con `models/`, `notebooks/`, `src/`) es un empaquetado
  **anterior** a estas correcciones (del 2026-07-06) — está **desactualizado**; regenerarlo
  si se va a entregar de ahí.
- **Descarga de COCO:** `images.cocodataset.org` a veces falla por límite de tasa en
  descargas paralelas; reintentar suele bastar. La PC local **no** tiene las imágenes de
  train2017 (solo se bajaron en el servidor); sí tiene las de val2017 (fair eval).
- **No modificar `demo_nube/` ni el Space de Hugging Face** — Federico los gestiona a mano.
- **Números que NO deben volver a aparecer** (eran de la corrida con fuga): mAP 0,524 /
  0,452, "+30% / +32%", 86,5 min. Los correctos están en §2.
