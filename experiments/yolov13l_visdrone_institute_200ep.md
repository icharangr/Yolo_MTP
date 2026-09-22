# YOLOv13-L VisDrone Institute-GPU Baseline

## Experiment Configuration

- Model: YOLOv13-L
- Dataset: VisDrone2019-DET
- Training images: 6471
- Validation images: 548
- Number of classes: 10
- Input size: 640 × 640
- Batch size: 4
- Epochs: 200
- Seed: 0
- AMP: True
- Pretrained: True
- Optimizer: Auto (resolved to SGD)
- Learning rate (lr0): 0.01
- Final learning rate factor (lrf): 0.01
- Momentum: 0.937
- Weight decay: 0.0005
- Warmup epochs: 3.0
- Box loss gain: 7.5
- Classification loss gain: 0.5
- DFL loss gain: 1.5
- Nominal batch size (nbs): 64
- Image augmentation: Ultralytics default configuration
- Device: NVIDIA RTX PRO 4000 Blackwell
- PyTorch: 2.12.1+cu130
- Ultralytics: 8.3.63

## Model Complexity

- Parameters: 27,520,501
- GFLOPs: 88.1

## Training

- Training time: 25.895 hours
- Run directory: `/home/charan/thesis/runs/yolov13l_visdrone_institute_200ep`
- Checkpoint: `weights/best.pt`
- Final checkpoint: `weights/last.pt`

## Best Validation Results

The best epoch and validation metrics are extracted directly from `results.csv`.

- Best epoch: [UPDATE]
- Precision: [UPDATE]
- Recall: [UPDATE]
- mAP@50: [UPDATE]
- mAP@50–95: [UPDATE]

## Final Epoch Results

- Epoch: 200
- Precision: 0.569
- Recall: 0.438
- mAP@50: 0.456
- mAP@50–95: 0.280

## Final Best-Checkpoint Validation

Validation of `best.pt` produced:

- Precision: 0.573
- Recall: 0.449
- mAP@50: 0.467
- mAP@50–95: 0.290

## Per-Class Observations

The model showed substantially stronger detection performance for larger and more visually distinct objects such as cars and buses. Small and densely packed categories remained more challenging, particularly bicycle and awning-tricycle. This reflects the difficulty of object detection in aerial imagery, where object scale, occlusion, and dense spatial distributions affect detection quality.

## Qualitative Outputs

The experiment generated:

- `results.png`
- `PR_curve.png`
- `P_curve.png`
- `R_curve.png`
- `F1_curve.png`
- `confusion_matrix.png`
- `confusion_matrix_normalized.png`
- Validation prediction/label visualizations

## Conclusion

YOLOv13-L completed the planned 200-epoch VisDrone training experiment successfully on the institute GPU. The best checkpoint achieved a mAP@50 of 0.467 and mAP@50–95 of 0.290, with precision and recall of 0.573 and 0.449, respectively. The larger L model provides substantially greater representational capacity than the N and S variants, reflected by its 27.5 million parameters and 88.1 GFLOPs. Detection performance remains class-dependent, with larger objects generally being easier to detect than small and densely distributed objects. The resulting checkpoint and evaluation plots provide a controlled baseline for comparison with YOLOv13-N, YOLOv13-S, and the forthcoming YOLOv13-X experiment.
