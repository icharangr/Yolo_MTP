# YOLOv13-S VisDrone Institute-GPU Baseline

- Dataset: VisDrone2019-DET
- Train images: 6471
- Validation images: 548
- Input size: 640x640
- Batch size: 4
- Epochs: 200
- Optimizer: Auto (resolved to SGD)
- Learning rate (lr0): 0.01
- Final LR factor (lrf): 0.01
- Momentum: 0.937
- Weight decay: 0.0005
- Warmup epochs: 3
- Box loss gain: 7.5
- Classification loss gain: 0.5
- DFL loss gain: 1.5
- Seed: 0
- AMP: True
- Device: NVIDIA RTX PRO 4000 Blackwell

## Best Validation Performance

- Best epoch: 93
- Precision: 0.52843
- Recall: 0.39772
- mAP@50: 0.41057
- mAP@50-95: 0.24695

## Final Epoch Performance

- Epoch: 200
- Precision: 0.53873
- Recall: 0.39278
- mAP@50: 0.40754
- mAP@50-95: 0.24338

## Conclusion

YOLOv13-S demonstrates improved detection capability compared with the smaller
YOLOv13-N configuration on the challenging VisDrone2019-DET dataset. The best
validation performance was obtained at epoch 93, achieving an mAP@50-95 of
0.24695. Increasing model capacity provides a measurable improvement in
detection performance, although recall remains relatively limited due to the
large number of small and densely packed aerial objects. The results establish
YOLOv13-S as a baseline for comparison with larger YOLOv13 variants and future
proposed improvements.

## Run Directory

/home/charan/thesis/runs/yolov13s_visdrone_institute_200ep2
