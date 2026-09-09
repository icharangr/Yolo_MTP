# Experiments

This directory documents experiments performed for the thesis using YOLOv13 on the VisDrone2019-DET dataset.

## Experiment naming

Use the following format:

`<model>_<dataset>_<purpose>`

Examples:
- `yolov13n_visdrone_baseline`
- `yolov13n_visdrone_hypergraph`
- `yolov13n_visdrone_ablation1`

## Reproducibility requirements

For every experiment, preserve:

- model variant and pretrained checkpoint
- dataset configuration
- image size
- batch size
- number of epochs
- optimizer and learning-rate settings
- augmentation settings
- random seed
- hardware and software environment
- training and validation results
- best checkpoint path
- plots and evaluation outputs

Training outputs should remain outside Git version control unless specifically required for a release.

## Current project sequence

1. Validate the VisDrone dataset pipeline.
2. Establish a YOLOv13 baseline on VisDrone.
3. Analyze baseline performance, especially small and crowded objects.
4. Evaluate proposed architectural improvements.
5. Perform ablation studies.
6. Compare the final method against the baseline.
