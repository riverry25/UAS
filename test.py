# ============================================================
# test.py
# Project Safety Helmet Compliance Detection
# Function
# - Evaluasi model pada data test
# - Mengambil precision, recall, F1-score, mAP50, mAP50-95
# - Menghitung inference time dan FPS
# - Menyimpan hasil evaluasi ke CSV
# ============================================================

from ultralytics import YOLO
from pathlib import Path
import argparse
import yaml
import glob
import os
import time
import torch
import pandas as pd


IMG_EXTS = [.jpg, .jpeg, .png, .JPG, .JPEG, .PNG]


def load_yaml(yaml_path)
    with open(yaml_path, r) as file
        return yaml.safe_load(file)


def calculate_f1(precision, recall)
    return 2  (precision  recall)  (precision + recall + 1e-9)


def get_test_images(data_yaml)
    test_dir = data_yaml.get(test)

    if test_dir is None
        raise ValueError(Path test tidak ditemukan di data.yaml)

    test_images = []

    for ext in IMG_EXTS
        test_images.extend(glob.glob(os.path.join(test_dir, ext)))

    return test_dir, test_images


def calculate_inference_time(model, test_images, imgsz, conf)
    print(n=== Menghitung inference time dan FPS ===)
    print(Jumlah gambar test, len(test_images))

    if len(test_images) == 0
        return 0, 0, 0

    # Warm up
    for img_path in test_images[10]
        _ = model.predict(img_path, imgsz=imgsz, conf=conf, verbose=False)

    if torch.cuda.is_available()
        torch.cuda.synchronize()

    start_time = time.time()

    for img_path in test_images
        _ = model.predict(img_path, imgsz=imgsz, conf=conf, verbose=False)

    if torch.cuda.is_available()
        torch.cuda.synchronize()

    end_time = time.time()

    total_inference_time = end_time - start_time
    avg_inference_time = total_inference_time  len(test_images)
    fps = 1  avg_inference_time if avg_inference_time  0 else 0

    return total_inference_time, avg_inference_time, fps


def evaluate_model(args)
    model = YOLO(args.weights)
    data_yaml = load_yaml(args.data)

    print(=== Evaluasi model pada data test ===)

    metrics = model.val(
        data=args.data,
        split=test,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        plots=True,
        project=args.project,
        name=args.name,
        exist_ok=True
    )

    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    f1_score = calculate_f1(precision, recall)
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)

    test_dir, test_images = get_test_images(data_yaml)

    total_time, avg_time, fps = calculate_inference_time(
        model=model,
        test_images=test_images,
        imgsz=args.imgsz,
        conf=args.conf
    )

    output_dir = Path(args.project)  args.name
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame([{
        Model args.model_name,
        Dataset Helmet Detection,
        Jumlah Kelas data_yaml.get(nc, ),
        Kelas , .join(data_yaml.get(names, [])),
        Epochs args.epochs,
        Image Size args.imgsz,
        Batch args.batch,
        Precision precision,
        Recall recall,
        F1-Score f1_score,
        mAP@50 map50,
        mAP@5095 map50_95,
        Total Inference Time (s) total_time,
        Inference Time per Image (s) avg_time,
        FPS fps
    }])

    summary_path = output_dir  f{args.model_name}_test_metrics.csv
    summary.to_csv(summary_path, index=False)

    print(n=== Hasil Evaluasi ===)
    print(summary)
    print(nCSV disimpan di, summary_path)


def main()
    parser = argparse.ArgumentParser(description=Test YOLO model for Helmet Detection)

    parser.add_argument(--weights, required=True, help=Path ke best.pt)
    parser.add_argument(--data, required=True, help=Path ke data.yaml)
    parser.add_argument(--project, default=contenthelmet_detection_test, help=Folder output testing)
    parser.add_argument(--name, required=True, help=Nama folder hasil test)
    parser.add_argument(--model_name, required=True, help=Nama model, contoh YOLOv8s)

    parser.add_argument(--epochs, type=int, default=100)
    parser.add_argument(--imgsz, type=int, default=640)
    parser.add_argument(--batch, type=int, default=8)
    parser.add_argument(--device, default=0)
    parser.add_argument(--conf, type=float, default=0.25)

    args = parser.parse_args()

    evaluate_model(args)


if __name__ == __main__
    main()