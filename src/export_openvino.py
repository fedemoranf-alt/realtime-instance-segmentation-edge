"""Exportación del modelo a OpenVINO — TPF Computer Vision.

Convierte los pesos entrenados (.pt) al formato OpenVINO, optimizado para
inferencia sobre CPU y GPU integradas Intel. En equipos Intel esta variante
suele duplicar o triplicar los FPS de la demostración en tiempo real respecto
de ejecutar el .pt directamente sobre CPU.

Se ejecuta una única vez, en la computadora local, tras descargar los pesos
del servidor de entrenamiento:

    python src/export_openvino.py

El resultado es el directorio models/yolov8n_seg_best_openvino_model/, que se
pasa a la demo con:

    python src/demo_webcam.py --model models/yolov8n_seg_best_openvino_model

Nota: la primera ejecución instala automáticamente el paquete openvino si no
está presente (lo gestiona ultralytics).
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

PROYECTO = Path(__file__).resolve().parents[1]
PESOS_DEFECTO = PROYECTO / "models" / "yolov8n_seg_best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Exporta los pesos .pt a OpenVINO.")
    parser.add_argument("--model", default=str(PESOS_DEFECTO), help="Ruta a los pesos .pt.")
    parser.add_argument("--imgsz", type=int, default=640, help="Resolución de inferencia.")
    args = parser.parse_args()

    modelo = YOLO(args.model)
    ruta_exportada = modelo.export(format="openvino", imgsz=args.imgsz, half=False)
    print(f"Modelo OpenVINO generado en: {ruta_exportada}")


if __name__ == "__main__":
    main()
