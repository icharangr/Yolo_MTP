from pathlib import Path
import json
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path("/home/charans/thesis/datasets/VisDrone")
PRED = Path("/home/charans/thesis/runs/yolov13n_visdrone_baseline_size_analysis/labels")
OUT = Path("/home/charans/thesis/experiments/yolov13n_visdrone_baseline_100ep")
OUT.mkdir(parents=True, exist_ok=True)

classes = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor"
]

images = []
annotations = []
categories = [{"id": i + 1, "name": name} for i, name in enumerate(classes)]

ann_id = 1

for image_id, image_path in enumerate(sorted((ROOT / "images" / "val").glob("*.jpg")), start=1):
    W, H = Image.open(image_path).size

    images.append({
        "id": image_id,
        "file_name": image_path.name,
        "width": W,
        "height": H
    })

    label_path = ROOT / "labels" / "val" / f"{image_path.stem}.txt"

    if label_path.exists():
        for line in label_path.read_text().splitlines():
            p = line.split()
            if len(p) != 5:
                continue

            cls, xc, yc, nw, nh = map(float, p)

            bw = nw * W
            bh = nh * H
            x = (xc * W) - bw / 2
            y = (yc * H) - bh / 2

            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": int(cls) + 1,
                "bbox": [x, y, bw, bh],
                "area": bw * bh,
                "iscrowd": 0
            })
            ann_id += 1

gt_json = {
    "images": images,
    "annotations": annotations,
    "categories": categories
}

gt_file = OUT / "visdrone_val_coco_gt.json"
gt_file.write_text(json.dumps(gt_json))

# Load ground truth
coco_gt = COCO(str(gt_file))

# Load predictions
predictions = []

for image_id, image_path in enumerate(sorted((ROOT / "images" / "val").glob("*.jpg")), start=1):
    pred_file = PRED / f"{image_path.stem}.txt"

    if not pred_file.exists():
        continue

    W, H = Image.open(image_path).size

    for line in pred_file.read_text().splitlines():
        p = line.split()

        if len(p) < 6:
            continue

        cls, xc, yc, nw, nh, conf = map(float, p[:6])

        bw = nw * W
        bh = nh * H
        x = (xc * W) - bw / 2
        y = (yc * H) - bh / 2

        predictions.append({
            "image_id": image_id,
            "category_id": int(cls) + 1,
            "bbox": [x, y, bw, bh],
            "score": conf
        })

pred_file = OUT / "visdrone_val_predictions.json"
pred_file.write_text(json.dumps(predictions))

coco_dt = coco_gt.loadRes(str(pred_file))

# COCO area ranges:
# small  : area < 32^2
# medium : 32^2 <= area < 96^2
# large  : area >= 96^2
ranges = {
    "small": [0, 32**2],
    "medium": [32**2, 96**2],
    "large": [96**2, 1e10],
}

results = {}

for name, area_range in ranges.items():
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.imgIds = coco_gt.getImgIds()
    evaluator.params.catIds = coco_gt.getCatIds()
    evaluator.params.areaRng = [area_range]
    evaluator.params.areaRngLbl = [name]

    evaluator.evaluate()
    evaluator.accumulate()

    # Precision array:
    # [IoU, Recall, Class, Area, MaxDets]
    precision = evaluator.eval["precision"]

    valid = precision[precision > -1]

    ap = float(valid.mean()) if valid.size else float("nan")
    results[name] = ap

    print(f"{name.capitalize():8}: AP = {ap:.4f} ({ap*100:.2f}%)")

print("\nSize-specific AP:")
for name, ap in results.items():
    print(f"{name.capitalize():8}: {ap*100:.2f}%")

(OUT / "size_ap_results.json").write_text(json.dumps(results, indent=2))
