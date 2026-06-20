# ============================================================
# train.py
# Project: Safety Helmet Compliance Detection
# Function:
# - Training model YOLO dengan konfigurasi utama yang sama
# - Model: YOLOv5s, YOLOv8s, YOLOv11s
# - imgsz: 640
# - epochs: 100
# - batch: 16
# - patience: 30
# ============================================================

from ultralytics import YOLO
import argparse


def train_model(args):
    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        workers=args.workers,
        patience=args.patience,
        save=True,
        plots=args.plots,
        cache=False,
        cos_lr=True,
        close_mosaic=10,
        exist_ok=True
    )

    return results


def main():
    parser = argparse.ArgumentParser(description="Training YOLO for Helmet Detection")

    parser.add_argument("--model", required=True, help="Contoh: yolov5s.pt, yolov8s.pt, yolo11s.pt")
    parser.add_argument("--data", required=True, help="Path ke data.yaml")
    parser.add_argument("--project", default="/content/helmet_detection_training", help="Folder output training")
    parser.add_argument("--name", required=True, help="Nama run training")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--plots", action="store_true", help="Aktifkan plot saat training")

    args = parser.parse_args()

    print("=== Training Configuration ===")
    print("Model   :", args.model)
    print("Data    :", args.data)
    print("Epochs  :", args.epochs)
    print("Image   :", args.imgsz)
    print("Batch   :", args.batch)
    print("Device  :", args.device)
    print("Project :", args.project)
    print("Name    :", args.name)

    train_model(args)


if __name__ == "__main__":
    main()