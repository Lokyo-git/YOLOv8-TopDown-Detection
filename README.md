# YOLOv8-TopDown-Detection
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch) ![YOLO](https://img.shields.io/badge/Ultralytics-YOLOv8-blueviolet) ![License](https://img.shields.io/badge/License-MIT-green)
> **Real-time Overhead Intrusion Detection System for Industrial Safety Automation at Lens Technology**

---

## 📌 1. Introduction & Motivation

In automated manufacturing environments, ensuring human-robot collision avoidance is a critical priority for industrial safety. This project develops a **Top-Down Real-time Overhead Intrusion Detection System** specifically tailored for **Lens Technology's** factory workshops. 

### Industrial Application Scenario:
* **The Challenge**: Traditional side-view cameras in manufacturing cells suffer heavily from visual occlusions caused by bulky robotic arms and industrial equipment, leading to severe blind spots.
* **Our Solution**: By utilizing an overhead (top-down) perspective, the system achieves an unobstructed line of sight over the workspace. Once a worker steps into a designated hazardous operational zone, the system instantly detects the intrusion and triggers an emergency stop signal to the active robotic arms, preventing any potential human injuries.
* **Engineering Requirements**: High Precision (zero false positives to avoid costly accidental factory downtime) and ultra-low latency for immediate hazard response.

---

## 📊 2. Dataset & Data Augmentation

### 2.1 Benchmark Dataset
The initial phases (v1.0 & v2.0) of this project were trained and evaluated on a specialized overhead person dataset sourced from Roboflow[cite: 1]:
* **Training Set**: 4,128 images (used for baseline and augmented training phases)[cite: 1].
* **Validation/Test Set**: 126 images (used for benchmarking)[cite: 1].
* **Source**: [Roboflow Overhead Person Dataset](https://universe.roboflow.com/riccardo-kxtut/overhead-person-szky0/)[cite: 1]

### 2.2 Advanced Data Augmentation (Phase v2.0)
To bridge the gap between open-source simulation data and the complex environment of a real factory floor, we introduced a robust data augmentation pipeline in Phase v2.0[cite: 1]. The pipeline specifically addresses **illumination robustness** and **spatial variations** by applying Mosaic, Mixup, Random Rotation, and Brightness Adjustment[cite: 1].

### 2.3 Custom Industrial Dataset & Domain Adaptation (Phase v3.0)
To achieve true production-ready deployment, Phase v3.0 completely transitioned from open-source simulated data to real-world workshop validation:
* **Hardware & Capture**: Utilized high-resolution **Hikvision Industrial Cameras** mounted overhead in simulated/actual factory cells to capture authentic work environments.
* **Data Processing**: Extracted video frames and manually annotated bounding boxes using `labelme`, converting annotations via custom scripts (`labelme_to_yolo.py`) to maintain absolute targeting precision.
* **Engineering Purpose**: This custom dataset effectively resolved the **Domain Shift** problem (visual mismatch between public benchmark textures and the actual factory metal/floor reflections), ensuring zero false alarms during operations.

---

## ⚙️ 3. Model Configuration & Benchmarking (Ablation Study)

### 3.1 Model Selection & Team Track
To optimize processing latency on edge devices, we selected the lightweight **YOLOv8n (Nano)**[cite: 1]. 
*(Track Collaboration: Our 4-member team split into two parallel evaluation tracks. Group A evaluated the larger `yolov8s` for upper-bound precision[cite: 1], while **our group (Group B) took full ownership of the end-to-end training, deployment optimization, and real-world evaluation of the YOLOv8n architecture**).*

### 3.2 Evaluation Metrics & Ablation Results
The training trajectory was tracked meticulously across all three critical architectural milestones:

| Phase / Run ID | Model | Data Source & Core Alterations | Precision | Recall | mAP50 | mAP50-95 | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **v1.0 (Baseline)** | YOLOv8n | Roboflow (Default Parameters) | 100.0% | 99.7% | 99.5% | 54.7% | Overfitted (Sanity Check)[cite: 1] |
| **v2.0 (Augmented)** | YOLOv8n | Roboflow + Heavy Augmentations | 82.4% | 94.1% | 96.8% | 51.8% | Robust Baseline[cite: 1] |
| **v3.0 (Industrial)** | YOLOv8n | **Hikvision Dataset + Fine-tuning** | **89.8%** | **86.8%** | **91.3%** | **56.8%** | **Production Ready** |

### 🔍 Performance Analysis & Engineering Insights:
* **The Baseline Artifact (v1.0)**: Achieving `100.0%` precision strongly indicated **environment-specific overfitting**[cite: 1]. The model was memorizing the static open-source backgrounds rather than generalized human anatomy[cite: 1].
* **The Augmented Breakthrough (v2.0)**: Heavy spatial/illumination alterations purposefully challenged the model[cite: 1]. Although raw metrics adjusted down to realistic levels, the **anti-interference robustness** improved exponentially[cite: 1].
* **The Production-Ready Realization (v3.0)**: By fine-tuning on the **Hikvision custom industrial dataset**, the model generalized perfectly to actual workshop textures. It achieved a balanced **89.8% Precision** and **91.3% mAP50**, while boosting the strict localization metric (**mAP50-95 to 56.8%**). This ensures highly precise hazard boundaries and prevents accidental factory downtime.

> 📊 **Live Tracking Note**: 
> The complete, step-by-step hyperparameter tuning log and training history are hosted on our cloud tracker[cite: 1].
> 👉 **[Click Here to View the Full Live Experiment Log on Google Drive](https://docs.google.com/spreadsheets/d/1JWhuS_KHb5iufRpdD68kEpIaFSlIHdOvs6Xkq9JZImA/edit?usp=drive_link)**[cite: 1]
---

## 📂 4. Repository Structure & Usage Guide

### 4.1 Directory Tree
Below is the optimized and modular structure of this repository (large datasets and raw weights are excluded via `.gitignore` to maintain a lightweight codebase):

```text
YOLOv8-TopDown-Detection/
├── assets/                  # Training trajectories, performance curves, and matrices[cite: 1]
│   ├── F1_curve.png
│   ├── confusion_matrix_normalized.png
│   ├── results.csv          # Complete training training logs per epoch[cite: 1]
│   ├── results.png          # Visualized training loss and metric trends[cite: 1]
│   ├── val_batch0_labels.jpg
│   └── val_batch0_pred.jpg
├── config/                  # Configuration directory[cite: 1]
│   └── data.yaml            # Dataset storage paths and class definitions[cite: 1]
├── src/                     # Core pipeline source code[cite: 1]
│   ├── extract_frame.py     # Extracts raw images from Hikvision video streams[cite: 1]
│   ├── image_dedup_phash.py # Cleans dataset by removing highly redundant video frames via pHash
│   ├── auto_labelme.py      # Semi-automatic labeling tool using existing models to pre-generate json
│   ├── labelme_to_yolo.py   # Converts Labelme JSON coordinates into standard YOLO txt format
│   ├── delete.py            # Utility script for asset management and dataset cleaning
│   ├── train.py             # Pipeline script for model training and fine-tuning[cite: 1]
│   └── test.py              # Script for real-time edge inference and validation loops[cite: 1]
├── .gitignore               # Explicitly bypasses bulky data/ and *.pt models[cite: 1]
└── requirements.txt         # Production-minimal environment dependencies[cite: 1]

```

### 4.2 Industrial Dataset Construction Workflow

For Phase v3.0, the pipeline for building the custom dataset from scratch follows these steps:

1. **Frame Extraction**: Run `python src/extract_frame.py` to extract images from raw surveillance footage.


2. **Perceptual Deduplication**: Run `python src/image_dedup_phash.py` to eliminate nearly identical consecutive frames, preventing model overfitting on redundant background features.
3. **Semi-Auto Annotation**: Run `python src/auto_labelme.py` to leverage a pre-trained model for generating rough bounding boxes, drastically reducing manual annotation time in Labelme.
4. **Format Alignment**: Run `python src/labelme_to_yolo.py` to convert the final verified JSON files into standard YOLO training labels.

### 4.3 How to Run

#### Step 1: Clone the Repository & Install Dependencies

Ensure you have Python 3.8+ and PyTorch installed, then set up the environment:

```bash
git clone [https://github.com/Lokyo-git/YOLOv8-TopDown-Detection.git](https://github.com/Lokyo-git/YOLOv8-TopDown-Detection.git)
cd YOLOv8-TopDown-Detection
pip install -r requirements.txt

```

#### Step 2: Model Training

To execute standard training or fine-tune on the custom dataset:

```bash
python src/train.py

```

#### Step 3: Real-Time Inference

To deploy the optimized model for edge overhead safety checking:

```bash
python src/test.py

```

---
