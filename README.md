# Comparative Evaluation of YOLOv5s, YOLOv8s, and YOLOv11s for Traffic Object Detection

This repository presents a comparative study of **YOLOv5s**, **YOLOv8s**, and **YOLOv11s** models for **real-time traffic object detection**.  
All models are trained **from scratch** under identical conditions to analyze architectural evolution, detection accuracy, and computational efficiency.

The study focuses on three traffic-related classes:
- **Car**
- **Bus**
- **Motorcycle**

---

## 📌 Project Overview

Real-time object detection is a key component of intelligent transportation systems and autonomous driving.  
This project evaluates three generations of YOLO models using a **unified and balanced dataset** constructed from multiple sources, ensuring a fair comparison.

Key goals:
- Analyze architectural improvements across YOLO versions
- Compare detection accuracy and generalization performance
- Measure training cost and inference-related behavior
- Identify the most suitable model for real-time traffic scenarios

---

## 🗂 Dataset

### Dataset Sources
- **COCO 2017** (filtered for traffic-related classes)
- **Vehicle Dataset for YOLO** (domain-specific traffic images)

### Classes
| Class | Label |
|------|------|
| Car | 0 |
| Bus | 1 |
| Motorcycle | 2 |

### Dataset Statistics
- ~17,000 training images  
- ~1,200 validation images  
- Approximately 7% validation split  

Instance distribution:
- Car: 43,911  
- Bus: 6,403  
- Motorcycle: 9,023  

All annotations are converted to **YOLO format** and normalized.

---

## 🧠 Model Selection

| Model | Year | Parameters |
|------|------|------------|
| YOLOv5s | 2020 | ~7.2M |
| YOLOv8s | 2023 | ~11.2M |
| YOLOv11s | 2024 | ~10.2M |

- **YOLOv5s**: CSP backbone, anchor-based detection  
- **YOLOv8s**: Anchor-free detection head, improved feature aggregation  
- **YOLOv11s**: Sparse attention, dynamic label assignment, refined feature scaling  

All models use the **small (s)** variants to ensure comparable computational cost.

---
## ⚙️ Training Configuration

All experiments were conducted under **identical conditions**:

- Training: **from scratch (no pretrained weights)**
- Input size: **640 × 640**
- Batch size: **16**
- Optimizer: **SGD (Ultralytics default)**
  - Initial learning rate: **0.01**
  - Momentum: **0.937**
- Class weighting enabled to mitigate class imbalance
- GPU: **NVIDIA RTX 2060**

| Model | Epochs |
|------|--------|
| YOLOv5s | 100 |
| YOLOv8s | 100 |
| YOLOv11s | 150 |

### Training Commands (Ultralytics CLI)

bash
#### YOLOv5s
yolo train model=yolov5s.yaml data=data.yaml imgsz=640 batch=16 epochs=100 device=0
#### YOLOv8s
yolo train model=yolov8s.yaml data=data.yaml imgsz=640 batch=16 epochs=100 device=0
#### YOLOv11s
yolo train model=yolov11s.yaml data=data.yaml imgsz=640 batch=16 epochs=150 device=0


## 📊 Evaluation Metrics

The following metrics are used:
- **mAP@0.5**
- **mAP@0.5:0.95**
- **Precision**
- **Recall**
- Training & validation loss curves
- Confusion matrices

---

## 📈 Results

### Quantitative Performance

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|------|--------|--------------|----------|--------|
| YOLOv5s | 0.7189 | 0.5498 | 0.7402 | 0.6806 |
| YOLOv8s | 0.7319 | 0.5669 | 0.7351 | **0.7033** |
| YOLOv11s | **0.7476** | **0.5823** | **0.7899** | 0.6817 |

### Training Time (RTX 2060)

| Model | Epochs | Total Time (hours) | Avg / Epoch (min) |
|------|--------|--------------------|------------------|
| YOLOv5s | 100 | 10.03 | 6.02 |
| YOLOv8s | 100 | 10.23 | 6.14 |
| YOLOv11s | 150 | 15.99 | 6.40 |

---

## 🔍 Key Observations

- **YOLOv5s**
  - Lowest parameter count
  - Higher background confusion
  - Baseline performance

- **YOLOv8s**
  - Best recall
  - Anchor-free detection improves object coverage
  - Slightly higher false positives

- **YOLOv11s**
  - Best overall performance
  - Highest precision and mAP
  - Most balanced confusion matrix
  - No overfitting despite longer training

---

## 🏆 Recommendation Summary

| Criterion | Best Model |
|--------|-----------|
| Highest Precision | YOLOv11s |
| Best Recall | YOLOv8s |
| Lowest Complexity | YOLOv5s |
| Best Overall Performance | **YOLOv11s** |

---

## 🚀 Future Work

- Train larger YOLO variants (m / l / x)
- Expand dataset size and class diversity
- Evaluate inference FPS and latency on edge devices
- Extend to pedestrian and traffic sign detection

---

## 📚 References

- COCO 2017 Dataset  
- Vehicle Dataset for YOLO  
- YOLOv5, YOLOv8, YOLOv11 official implementations  

---

## 👤 Authors

**Mete Cem Turan**  
**Kerem Elma**  
**Kayrahan Toprak Tosun**

---

## 📄 License

This project is released for **academic and research purposes**.
