# ============================================================
# predict.py
# Project: Safety Helmet Compliance Detection
# Function:
# - Prediksi gambar test/sample menggunakan model terbaik
# - Menyimpan hasil bounding box
# - Opsional menyimpan label prediksi .txt dan confidence
# ============================================================

from ultralytics import YOLO
import argparse


def predict_model(args):
    model = YOLO(args.weights)

    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        save=True,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        project=args.project,
        name=args.name,
        exist_ok=True
    )

    print("Hasil prediksi disimpan di:")
    print(f"{args.project}/{args.name}")


def main():
    parser = argparse.ArgumentParser(description="Predict using YOLO model for Helmet Detection")

    parser.add_argument("--weights", required=True, help="Path ke best.pt")
    parser.add_argument("--source", required=True, help="Path gambar/folder/video")
    parser.add_argument("--project", default="/content/helmet_detection_prediction", help="Folder output prediksi")
    parser.add_argument("--name", required=True, help="Nama folder hasil prediksi")

    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--save_txt", action="store_true", help="Simpan label prediksi dalam format .txt")
    parser.add_argument("--save_conf", action="store_true", help="Simpan confidence score pada label prediksi")

    args = parser.parse_args()

    predict_model(args)


if __name__ == "__main__":
    main()