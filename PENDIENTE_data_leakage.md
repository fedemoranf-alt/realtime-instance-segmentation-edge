# PENDIENTE — Fuga de datos (data leakage) en la corrida del 2026-07-05

> **Estado:** conocido y NO corregido en los resultados actuales. El informe técnico
> se redactó con las métricas de esta corrida por restricciones de tiempo.
> Corregir y re-entrenar cuando sea posible, antes de cualquier uso posterior del modelo.

## Qué pasó

FiftyOne etiqueta cada muestra descargada del zoo con su split de origen (`train`,
por provenir de COCO train2017). La partición propia del Notebook 01 usaba esa misma
palabra como etiqueta, por lo que `view.match_tags('train')` devolvió las **5.000
imágenes completas** en lugar de las 3.000 previstas (60 %). Evidencia en el notebook
ejecutado (`notebooks_ejecutados/01_dataset.ipynb`, celda de particiones):

```
train    5000     ← debería ser 3000
val       750
test     1250
```

Consecuencia: el conjunto de entrenamiento exportado **incluyó las 1.250 imágenes de
test y las 750 de val**. El modelo `models/yolov8n_seg_best.pt` vio durante el
entrenamiento las imágenes con las que luego fue evaluado.

## Impacto

- Las métricas de test reportadas (mAP@50-95: 0,524 cajas / 0,452 máscaras) están
  **sesgadas al alza** en magnitud desconocida. La comparación con el baseline sigue
  siendo direccionalmente válida (el baseline se evaluó sobre las mismas imágenes),
  pero la magnitud de la mejora también está inflada.
- Las mediciones de velocidad (GPU y CPU/OpenVINO) **no** están afectadas: dependen
  de la arquitectura, no de los pesos.

## Cómo corregirlo (ya implementado en `notebooks/01_dataset.ipynb`)

El notebook 01 de la carpeta `notebooks/` ya contiene el fix:
1. `dataset.untag_samples(['train', 'validation', 'test'])` antes de `random_split`.
2. Un `assert` que verifica que las particiones son disjuntas y cubren el total.
3. Borrado de `data/coco_subset*` antes de exportar (evita mezclar con corridas previas).

Pasos para regenerar resultados válidos:
1. Subir el `notebooks/01_dataset.ipynb` corregido al servidor y borrar allí `runs/`.
2. Ejecutar notebooks 01 → 02 → 03 completos (verificar que 01 imprima `train 3000 / val 750 / test 1250`).
3. Re-descargar `models/`, `runs/` y los notebooks ejecutados.
4. Re-exportar OpenVINO localmente (`python src/export_openvino.py`) y actualizar las capturas de la demo.
5. Actualizar las tablas y figuras del informe con las métricas nuevas (se esperan
   valores algo menores; el pipeline de figuras está en `reporte/figuras/`).
