"""Demostración de inferencia en tiempo real — TPF Computer Vision.

Ejecuta el modelo de detección y segmentación de instancias (YOLOv8n-seg
ajustado en el Notebook 02) sobre la señal de una cámara web o un archivo de
video, superponiendo cajas, máscaras y etiquetas de clase, junto con la tasa
de fotogramas por segundo (FPS) efectiva.

Uso típico (desde la raíz del proyecto):

    python src/demo_webcam.py                      # webcam por defecto, pesos de models/
    python src/demo_webcam.py --source video.mp4   # archivo de video
    python src/demo_webcam.py --model models/yolov8n_seg_best_openvino_model
                                                   # variante optimizada para CPU Intel

La ventana se cierra con la tecla «q».
"""

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

PROYECTO = Path(__file__).resolve().parents[1]
#PESOS_DEFECTO = PROYECTO / "models" / "yolov8n_seg_best.pt"
PESOS_DEFECTO = PROYECTO / "models" / "yolov8n_seg_best_openvino_model"


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inferencia en tiempo real con YOLOv8-seg (TPF Computer Vision)."
    )
    parser.add_argument(
        "--model",
        default=str(PESOS_DEFECTO),
        help="Ruta a los pesos (.pt) o a un modelo exportado (p. ej. directorio OpenVINO).",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Índice de la cámara (0, 1, ...) o ruta a un archivo de video.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Umbral de confianza para las predicciones (default: 0.4).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Resolución de inferencia; reducir a 480 o 320 si los FPS son bajos.",
    )
    return parser


def main() -> None:
    args = construir_parser().parse_args()

    #modelo = YOLO(args.model) # para el modelo completo (no optimizado) .pt
    modelo = YOLO(args.model, task="segment") #para el modelo optimizado OpenVINO (directorio)
    fuente = int(args.source) if args.source.isdigit() else args.source

    captura = cv2.VideoCapture(fuente)
    if not captura.isOpened():
        raise SystemExit(
            f"No se pudo abrir la fuente de video: {args.source!r}. "
            "Verificar el índice de la cámara o la ruta del archivo."
        )

    ventana = "TPF - Deteccion y Segmentacion en Tiempo Real (q: salir)"
    fps_suavizado = 0.0

    while True:
        ok, fotograma = captura.read()
        if not ok:
            break  # fin del video o error de captura

        t0 = time.perf_counter()
        resultados = modelo.predict(
            fotograma, conf=args.conf, imgsz=args.imgsz, verbose=False
        )
        anotado = resultados[0].plot()  # cajas + máscaras + etiquetas
        dt = time.perf_counter() - t0

        # Media móvil exponencial para una lectura estable de los FPS
        fps_instantaneo = 1.0 / dt if dt > 0 else 0.0
        fps_suavizado = (
            fps_instantaneo
            if fps_suavizado == 0.0
            else 0.9 * fps_suavizado + 0.1 * fps_instantaneo
        )

        n_objetos = len(resultados[0].boxes) if resultados[0].boxes is not None else 0
        texto = f"FPS: {fps_suavizado:5.1f} | objetos: {n_objetos}"
        cv2.rectangle(anotado, (8, 8), (330, 44), (30, 30, 30), thickness=-1)
        cv2.putText(
            anotado, texto, (16, 34),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
        )

        cv2.imshow(ventana, anotado)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    captura.release()
    cv2.destroyAllWindows()
    print(f"FPS promedio (últimos fotogramas): {fps_suavizado:.1f}")


if __name__ == "__main__":
    main()
