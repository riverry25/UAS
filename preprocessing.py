# ============================================================
# preprocessing.py
# Project: Safety Helmet Compliance Detection
# Function:
# - Cari dan cek data.yaml
# - Perbaiki path train/val/test pada data.yaml
# - Konversi label polygon ke bbox YOLO jika ada
# - Bersihkan label kosong
# - Hapus gambar tanpa label
# - Hapus label tanpa gambar
# - Validasi format label YOLO
# - Hitung jumlah data train/valid/test
# ============================================================

from pathlib import Path
import argparse
import yaml
import os

IMG_EXTS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
EXPECTED_CLASSES = ["person", "helmet", "no_helmet"]


def find_data_yaml(dataset_dir):
    dataset_dir = Path(dataset_dir)

    for root, dirs, files in os.walk(dataset_dir):
        if "data.yaml" in files:
            return Path(root) / "data.yaml"

    raise FileNotFoundError("data.yaml tidak ditemukan. Cek kembali folder dataset.")


def load_yaml(yaml_path):
    with open(yaml_path, "r") as file:
        return yaml.safe_load(file)


def save_yaml(yaml_path, data_yaml):
    with open(yaml_path, "w") as file:
        yaml.dump(data_yaml, file, sort_keys=False)


def fix_data_yaml(dataset_path, yaml_path):
    data_yaml = load_yaml(yaml_path)

    print("=== Data YAML awal ===")
    print(data_yaml)

    data_yaml["train"] = str(Path(dataset_path) / "train/images")
    data_yaml["val"] = str(Path(dataset_path) / "valid/images")
    data_yaml["test"] = str(Path(dataset_path) / "test/images")

    if "names" in data_yaml:
        names = data_yaml["names"]

        if isinstance(names, dict):
            names = [names[i] for i in range(len(names))]

        data_yaml["names"] = names
        data_yaml["nc"] = len(names)

    save_yaml(yaml_path, data_yaml)

    print("\n=== Data YAML setelah diperbaiki ===")
    print(data_yaml)

    return data_yaml


def convert_label_to_bbox(label_path):
    new_lines = []

    with open(label_path, "r") as file:
        lines = file.readlines()

    for line in lines:
        parts = line.strip().split()

        if len(parts) < 5:
            continue

        cls = parts[0]
        coords = list(map(float, parts[1:]))

        # Jika format sudah bbox YOLO: class x_center y_center width height
        if len(coords) == 4:
            x_center, y_center, width, height = coords

        # Jika format polygon: class x1 y1 x2 y2 x3 y3 ...
        else:
            if len(coords) % 2 != 0:
                continue

            xs = coords[0::2]
            ys = coords[1::2]

            x_min = min(xs)
            x_max = max(xs)
            y_min = min(ys)
            y_max = max(ys)

            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min

        # Batasi nilai agar tetap 0 sampai 1
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))

        if width <= 0 or height <= 0:
            continue

        new_lines.append(f"{cls} {x_center} {y_center} {width} {height}")

    with open(label_path, "w") as file:
        file.write("\n".join(new_lines))


def convert_all_labels(dataset_path):
    print("\n=== Convert label polygon ke bbox YOLO jika ada ===")

    for split in ["train", "valid", "test"]:
        label_dir = Path(dataset_path) / split / "labels"
        count = 0

        for label_path in label_dir.glob("*.txt"):
            convert_label_to_bbox(label_path)
            count += 1

        print(f"{split}: {count} file label diproses")


def clean_dataset(dataset_path, split):
    img_dir = Path(dataset_path) / split / "images"
    label_dir = Path(dataset_path) / split / "labels"

    print(f"\n=== Membersihkan split: {split} ===")

    empty_count = 0
    for label_path in label_dir.glob("*.txt"):
        if label_path.stat().st_size == 0:
            label_path.unlink()
            empty_count += 1

    removed_images = 0
    for img_path in img_dir.iterdir():
        if img_path.suffix in IMG_EXTS:
            label_path = label_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                img_path.unlink()
                removed_images += 1

    image_stems = {
        img_path.stem
        for img_path in img_dir.iterdir()
        if img_path.suffix in IMG_EXTS
    }

    removed_labels = 0
    for label_path in label_dir.glob("*.txt"):
        if label_path.stem not in image_stems:
            label_path.unlink()
            removed_labels += 1

    print("Label kosong dihapus:", empty_count)
    print("Gambar tanpa label dihapus:", removed_images)
    print("Label tanpa gambar dihapus:", removed_labels)


def count_dataset(dataset_path, split):
    img_dir = Path(dataset_path) / split / "images"
    label_dir = Path(dataset_path) / split / "labels"

    images = []
    for ext in IMG_EXTS:
        images.extend(list(img_dir.glob(f"*{ext}")))

    labels = list(label_dir.glob("*.txt"))

    print(f"\n{split}")
    print("Images :", len(images))
    print("Labels :", len(labels))
    print("-" * 30)


def validate_labels(dataset_path, split, nc):
    label_dir = Path(dataset_path) / split / "labels"
    bad_files = []

    for label_path in label_dir.glob("*.txt"):
        with open(label_path, "r") as file:
            lines = file.readlines()

        for line_number, line in enumerate(lines, start=1):
            parts = line.strip().split()

            if len(parts) != 5:
                bad_files.append(
                    (label_path.name, line_number, "jumlah kolom bukan 5", line.strip())
                )
                break

            try:
                cls = int(float(parts[0]))
                values = list(map(float, parts[1:]))
            except ValueError:
                bad_files.append(
                    (label_path.name, line_number, "format angka salah", line.strip())
                )
                break

            if cls < 0 or cls >= nc:
                bad_files.append(
                    (label_path.name, line_number, f"class id salah: {cls}", line.strip())
                )
                break

            if any(value < 0 or value > 1 for value in values):
                bad_files.append(
                    (label_path.name, line_number, "koordinat di luar 0-1", line.strip())
                )
                break

            width = values[2]
            height = values[3]

            if width <= 0 or height <= 0:
                bad_files.append(
                    (label_path.name, line_number, "width/height tidak valid", line.strip())
                )
                break

    print(f"{split} label bermasalah:", len(bad_files))

    for item in bad_files[:10]:
        print(item)

    return bad_files


def main():
    parser = argparse.ArgumentParser(description="Preprocessing dataset Helmet Detection")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path folder dataset yang berisi train/valid/test dan data.yaml"
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    yaml_path = find_data_yaml(dataset_path)

    print("Dataset path:", dataset_path)
    print("YAML path:", yaml_path)

    data_yaml = fix_data_yaml(dataset_path, yaml_path)

    print("\n=== Cek class ===")
    print("Jumlah kelas:", data_yaml["nc"])
    print("Nama kelas:", data_yaml["names"])

    if data_yaml["names"] == EXPECTED_CLASSES:
        print("[OK] Class sesuai: person, helmet, no_helmet")
    else:
        print("[WARNING] Class berbeda dari expected.")
        print("Expected:", EXPECTED_CLASSES)

    convert_all_labels(dataset_path)

    for split in ["train", "valid", "test"]:
        clean_dataset(dataset_path, split)

    print("\n=== Hitung jumlah dataset ===")
    for split in ["train", "valid", "test"]:
        count_dataset(dataset_path, split)

    print("\n=== Validasi label ===")
    all_bad = []

    for split in ["train", "valid", "test"]:
        all_bad.extend(validate_labels(dataset_path, split, data_yaml["nc"]))

    print("\nTotal semua label bermasalah:", len(all_bad))

    if len(all_bad) == 0:
        print("[OK] Dataset siap digunakan untuk training.")
    else:
        print("[WARNING] Masih ada label bermasalah. Cek kembali hasil validasi.")


if __name__ == "__main__":
    main()