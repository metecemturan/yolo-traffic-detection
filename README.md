# Comparative Evaluation of YOLOv5s, YOLOv7-tiny, and YOLOv11s for Real-time Traffic Object Detection

This repository contains the implementation of a **real-time object detection system** that compares **YOLOv5**, **YOLOv7**, and **YOLOv11** models on **vehicle detection** tasks.  
The project evaluates **accuracy, inference speed, and computational efficiency** under unified training and validation conditions.

---

## 🚗 Project Overview

The goal of this study is to identify the most efficient YOLO model for detecting **cars, buses and motorcycles** in real-time video streams.

The models are trained on a **harmonized multi-source dataset** derived from:
- **COCO 2017**  
- **Vehicle Dataset for YOLO**  
- **PASCAL VOC 2007**  
- **Open Images V7**

Filtered annotations were created for only the required classes, and datasets were merged after converting all labels from COCO JSON format to YOLO TXT format.

---

## 🧠 Selected YOLO Models

| YOLO Version | Variant | Parameters (M) | Year | Key Features |
|---------------|----------|----------------|------|---------------|
| YOLOv5 | YOLOv5s | 7.2 | 2020 | Lightweight, PyTorch-based baseline |
| YOLOv7 | YOLOv7-tiny | 6.2 | 2022 | E-ELAN, model re-parameterization |
| YOLOv11 | YOLOv11s | 11.2 | 2024 | Sparse attention, dynamic label assignment |

> The “small” (s) and “tiny” variants were selected for their low parameter counts, which make them suitable for local GPU training while preserving the core architectural characteristics of their respective versions.  
> *Table 1 shows the parameter comparison of these models.*

---

## 🧾 Dataset Summary

| Split | Number of Images | Description |
|-------|------------------|-------------|
| Train | 17,000 | Merged COCO + Vehicle Dataset |
| Val   | 1,200  | 7% validation ratio |
| Classes | Car (0), Bus (1), Motorcycle (2), Person (3) |

---

## ⚙️ Training Configuration

| Setting | Value |
|----------|-------|
| Epochs | 100 |
| Batch Size | 16 |
| Image Size | 640x640 |
| Optimizer | SGD (momentum=0.937) |
| Augmentations | Mosaic, RandomFlip, HSV Shift |
| Hardware | NVIDIA RTX 2060 / 1660 Ti / RX 7800 XT |

Models are trained using:
```bash
python train.py --img 640 --batch 16 --epochs 100 --data data.yaml --weights None --device 0
