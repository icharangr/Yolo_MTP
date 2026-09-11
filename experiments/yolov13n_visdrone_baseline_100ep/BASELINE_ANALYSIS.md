# YOLOv13-N VisDrone Baseline Analysis

## 1. Experiment Overview

This experiment establishes the YOLOv13-N baseline on the VisDrone2019-DET dataset.

- Model: YOLOv13-N
- Pretrained weights: `yolov13n.pt`
- Dataset: VisDrone2019-DET
- Training epochs: 100
- Input resolution: 640 × 640
- Batch size: 4
- Optimizer: AdamW (`optimizer=auto`)
- Device: NVIDIA GeForce RTX 5060 Laptop GPU
- Validation images: 548
- Validation instances: 38,759
- Model parameters: 2,449,845
- GFLOPs: 6.2

## 2. Overall Validation Performance

| Metric | Result |
|---|---:|
| Precision | 42.5% |
| Recall | 33.7% |
| mAP@50 | 33.2% |
| mAP@75 | 19.4% |
| mAP@50:95 | 19.4% |

## 3. Per-Class Performance

| Class | Precision | Recall | mAP@50 | mAP@75 | mAP@50:95 |
|---|---:|---:|---:|---:|---:|
| Pedestrian | 44.9% | 32.8% | 34.1% | 9.77% | 14.5% |
| People | 46.9% | 23.7% | 26.9% | 4.75% | 9.83% |
| Bicycle | 24.9% | 11.4% | 8.71% | 2.00% | 3.36% |
| Car | 63.2% | 74.9% | 75.3% | 57.1% | 51.5% |
| Van | 43.6% | 39.0% | 36.6% | 29.1% | 25.4% |
| Truck | 45.9% | 31.7% | 32.4% | 22.9% | 20.8% |
| Tricycle | 37.1% | 24.0% | 22.2% | 11.0% | 11.9% |
| Awning-tricycle | 23.7% | 16.2% | 12.7% | 8.88% | 8.13% |
| Bus | 51.1% | 44.2% | 47.1% | 38.5% | 33.1% |
| Motor | 43.9% | 38.8% | 36.3% | 9.76% | 15.1% |

## 4. Object-Size Distribution

The converted VisDrone annotations were analyzed using the following area ranges:

- Small: area < 32² pixels
- Medium: 32² ≤ area < 96² pixels
- Large: area ≥ 96² pixels

### Training Set

- Total objects: 343,204
- Small: 207,604 (60.49%)
- Medium: 116,620 (33.98%)
- Large: 18,980 (5.53%)

### Validation Set

- Total objects: 38,759
- Small: 26,586 (68.59%)
- Medium: 11,105 (28.65%)
- Large: 1,068 (2.76%)

## 5. Detection Performance by Object Size

| Object Size | AP |
|---|---:|
| Small | 4.71% |
| Medium | 19.95% |
| Large | 36.47% |

The results demonstrate a substantial decrease in detection performance as object size decreases. Small objects constitute 68.59% of the validation instances but achieve only 4.71% AP, whereas large objects achieve 36.47% AP.

## 6. Baseline Observations

The baseline performs substantially better on relatively larger and visually distinctive objects. The car category achieves the highest mAP@50:95 of 51.5%, while bicycle, awning-tricycle and people have considerably lower performance.

The normalized confusion matrix and validation predictions also indicate substantial missed detections, particularly for small and densely distributed objects. Vehicle categories such as car, van, truck and bus also show inter-class confusion.

## 7. Research Motivation

The baseline results indicate a significant small-object detection challenge in aerial imagery. Since small objects represent the majority of the VisDrone validation instances while exhibiting substantially lower AP than medium and large objects, improving small-object representation and detection is identified as the primary research direction for subsequent experiments.

The baseline model and its results will be retained as the reference point for evaluating proposed improvements and ablation experiments.
