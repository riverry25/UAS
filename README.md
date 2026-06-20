# UAS AI - Safety Helmet Compliance Detection

## Project Title

**Comparative Analysis of YOLOv5s, YOLOv8s, and YOLOv11s for Safety Helmet Compliance Detection on Construction Sites Using a Standardized Experimental Pipeline**

## Project Description

This project compares three YOLO object detection models, namely YOLOv5s, YOLOv8s, and YOLOv11s, for safety helmet compliance detection on construction sites.

The purpose of this project is to evaluate the performance of each model using the same dataset, training configuration, and evaluation pipeline. The comparison focuses on detection quality, training efficiency, and inference speed.

The detected classes are:

* person
* helmet
* no_helmet

## Dataset

The dataset used in this project is the **Helmet Detection Computer Vision Dataset** from Roboflow Universe.

Dataset information:

| Item         | Description                              |
| ------------ | ---------------------------------------- |
| Dataset      | Helmet Detection Computer Vision Dataset |
| Source       | Roboflow Universe                        |
| Provider     | Northeastern University - China          |
| Workspace    | 039s Workspace                           |
| License      | Public Domain                            |
| Total images | 9,022                                    |
| Image size   | 640×640                                  |
| Classes      | person, helmet, no_helmet                |
| Train        | 6,482 images                             |
| Validation   | 1,671 images                             |
| Test         | 869 images                               |

The dataset is not included in this repository. The dataset should be downloaded manually from Roboflow and prepared in YOLO format before running the training process.

## Models

The models compared in this project are:

| Model    | Description                                 |
| -------- | ------------------------------------------- |
| YOLOv5s  | Baseline YOLO model                         |
| YOLOv8s  | Modern YOLO model                           |
| YOLOv11s | Latest YOLO model evaluated in this project |

## Experimental Setup

All models were trained using the same configuration:

| Parameter               | Value           |
| ----------------------- | --------------- |
| Image size              | 640×640         |
| Batch size              | 16              |
| Maximum epochs          | 100             |
| Early stopping patience | 30              |
| Platform                | Google Colab    |
| GPU                     | NVIDIA Tesla T4 |
| Dataset format          | YOLO format     |

## Evaluation Metrics

The models were evaluated using the following metrics:

* Precision
* Recall
* F1-score
* mAP50
* mAP50-95
* Test accuracy
* Training time
* Inference time per image
* FPS
* Confusion matrix analysis

## Final Result Summary

| Metric                   | YOLOv5s | YOLOv8s | YOLOv11s | Best Model |
| ------------------------ | ------: | ------: | -------: | ---------- |
| Precision                |   0.824 |   0.828 |    0.843 | YOLOv11s   |
| Recall                   |   0.803 |   0.802 |    0.809 | YOLOv11s   |
| F1-Score                 |   0.814 |   0.815 |    0.826 | YOLOv11s   |
| mAP50                    |   0.834 |   0.839 |    0.850 | YOLOv11s   |
| mAP50-95                 |   0.551 |   0.558 |    0.563 | YOLOv11s   |
| Test Accuracy (%)        |   90.39 |   90.85 |    91.10 | YOLOv11s   |
| Training Time (h)        |    5.49 |    3.82 |     4.35 | YOLOv8s    |
| Inference Time/Image (s) | 0.01203 | 0.01388 |  0.01449 | YOLOv5s    |
| FPS                      |   83.13 |   72.03 |    69.01 | YOLOv5s    |

## Main Findings

Based on the experiment:

* YOLOv11s achieved the best detection quality.
* YOLOv8s achieved the shortest training time.
* YOLOv5s achieved the highest inference speed.
* The person class was the most difficult class to detect.
* The helmet and no_helmet classes achieved more stable detection performance.

## Repository Structure

```text
UAS_AI_Safety_Helmet_Detection/
├── Notebook/
│   ├── helmet_detection_yolov5.ipynb
│   ├── helmet_detection_yolov8.ipynb
│   └── helmet_detection_yolov11.ipynb
│
├── Script/
│   ├── preprocessing.py
│   ├── train.py
│   ├── test.py
│   └── predict.py
│
└── readme.md
```

## File Description

### Notebook

The `Notebook` folder contains the Google Colab notebooks used for training and evaluating each YOLO model.

| File                             | Description                                   |
| -------------------------------- | --------------------------------------------- |
| `helmet_detection_yolov5.ipynb`  | Notebook for YOLOv5s training and evaluation  |
| `helmet_detection_yolov8.ipynb`  | Notebook for YOLOv8s training and evaluation  |
| `helmet_detection_yolov11.ipynb` | Notebook for YOLOv11s training and evaluation |

### Script

The `Script` folder contains Python scripts used in the experiment.

| File               | Description                                   |
| ------------------ | --------------------------------------------- |
| `preprocessing.py` | Script for dataset checking and preprocessing |
| `train.py`         | Script for model training                     |
| `test.py`          | Script for model evaluation on the test set   |
| `predict.py`       | Script for running prediction on test images  |

## How to Run

### 1. Install Required Libraries

```bash
pip install ultralytics opencv-python pandas numpy matplotlib torch torchvision
```

### 2. Prepare Dataset

Download the dataset from Roboflow and make sure the dataset is in YOLO format.

The dataset structure should contain:

```text
train/
valid/
test/
data.yaml
```

Update the dataset path in the notebook or script before running the training process.

### 3. Run Training

Training can be performed using the available notebooks:

```text
Notebook/helmet_detection_yolov5.ipynb
Notebook/helmet_detection_yolov8.ipynb
Notebook/helmet_detection_yolov11.ipynb
```

Alternatively, training can be executed using:

```bash
python Script/train.py
```

### 4. Run Testing

```bash
python Script/test.py
```

### 5. Run Prediction

```bash
python Script/predict.py
```

## Conclusion

This project shows that YOLOv11s achieved the best detection quality for safety helmet compliance detection, while YOLOv8s achieved the shortest training time and YOLOv5s achieved the highest inference speed. The results indicate that each model has different strengths depending on the deployment objective.

## Author

Ferdi Ahyana Yusri
Department of Informatics
Universitas Islam Negeri Sultan Maulana Hasanuddin Banten
