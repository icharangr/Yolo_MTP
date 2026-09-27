from pathlib import Path
import csv
import numpy as np
from ultralytics import YOLO

# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_DIR = Path("/home/charan/thesis/datasets/VisDrone/images/val")
LABEL_DIR = Path("/home/charan/thesis/datasets/VisDrone/labels/val")

MODELS = {
    "N": Path(
        "/home/charan/thesis/runs/yolov13n_visdrone_institute_300ep/weights/best.pt"
    ),
    "S": Path(
        "/home/charan/thesis/runs/yolov13s_visdrone_institute_200ep2/weights/best.pt"
    ),
    "L": Path(
        "/home/charan/thesis/runs/yolov13l_visdrone_institute_200ep/weights/best.pt"
    ),
    "X": Path(
        "/home/charan/thesis/runs/yolov13x_visdrone_institute_200ep/weights/best.pt"
    ),
}

OUTPUT_DIR = Path(
    "/home/charan/thesis/yolov13/experiments/oracle_routing"
)

IMG_SIZE = 640
CONF = 0.25
IOU_THRESHOLD = 0.50
DEVICE = "cuda:0"

# Smaller model wins when F1 is exactly equal
MODEL_ORDER = ["N", "S", "L", "X"]


# ============================================================
# HELPERS
# ============================================================

def load_labels(label_file):
    """
    Load YOLO-format labels.

    Format:
        class x_center y_center width height

    Coordinates are normalized to [0, 1].
    """

    if not label_file.exists():
        return []

    labels = []

    with open(label_file, "r") as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            cls, xc, yc, w, h = map(float, parts)

            labels.append({
                "cls": int(cls),
                "box": np.array(
                    [xc, yc, w, h],
                    dtype=np.float32
                )
            })

    return labels


def xywh_to_xyxy(box):
    """
    Convert:
        x_center, y_center, width, height

    to:
        x1, y1, x2, y2
    """

    xc, yc, w, h = box

    return np.array([
        xc - w / 2,
        yc - h / 2,
        xc + w / 2,
        yc + h / 2
    ], dtype=np.float32)


def iou_xyxy(a, b):
    """
    Calculate IoU between two boxes in xyxy format.
    """

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    intersection = iw * ih

    area_a = (
        max(0.0, ax2 - ax1)
        * max(0.0, ay2 - ay1)
    )

    area_b = (
        max(0.0, bx2 - bx1)
        * max(0.0, by2 - by1)
    )

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# METRIC CALCULATION
# ============================================================

def calculate_metrics(
    gt_labels,
    predictions,
    image_shape
):
    """
    Class-aware greedy matching.

    IMPORTANT:
    Ground-truth boxes are normalized YOLO coordinates.
    Model predictions are pixel coordinates.

    Therefore, GT boxes are converted to pixel coordinates
    before IoU calculation.
    """

    img_h, img_w = image_shape

    # --------------------------------------------------------
    # Convert GT boxes to pixel xyxy
    # --------------------------------------------------------

    gt = []

    for item in gt_labels:

        box = xywh_to_xyxy(item["box"])

        # Normalized -> pixel coordinates
        box[0] *= img_w
        box[2] *= img_w
        box[1] *= img_h
        box[3] *= img_h

        gt.append(
            (
                item["cls"],
                box
            )
        )

    # --------------------------------------------------------
    # Prepare predictions
    # --------------------------------------------------------

    pred = [
        (
            int(item["cls"]),
            item["box"],
            item["conf"]
        )
        for item in predictions
    ]

    # Highest-confidence predictions first
    pred.sort(
        key=lambda x: x[2],
        reverse=True
    )

    matched_gt = set()

    tp = 0
    fp = 0

    # --------------------------------------------------------
    # Greedy matching
    # --------------------------------------------------------

    for pred_cls, pred_box, _ in pred:

        best_iou = 0.0
        best_gt = None

        for idx, (gt_cls, gt_box) in enumerate(gt):

            if idx in matched_gt:
                continue

            # Class-aware matching
            if pred_cls != gt_cls:
                continue

            iou = iou_xyxy(
                pred_box,
                gt_box
            )

            if (
                iou >= IOU_THRESHOLD
                and iou > best_iou
            ):
                best_iou = iou
                best_gt = idx

        if best_gt is not None:

            tp += 1
            matched_gt.add(best_gt)

        else:

            fp += 1

    # --------------------------------------------------------
    # False negatives
    # --------------------------------------------------------

    fn = len(gt) - tp

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    # --------------------------------------------------------
    # F1
    # --------------------------------------------------------

    if precision + recall > 0:

        f1 = (
            2
            * precision
            * recall
            / (precision + recall)
        )

    else:

        f1 = 0.0

    return (
        tp,
        fp,
        fn,
        precision,
        recall,
        f1
    )


# ============================================================
# DATASET
# ============================================================

image_files = sorted(
    list(IMAGE_DIR.glob("*.jpg"))
    + list(IMAGE_DIR.glob("*.jpeg"))
    + list(IMAGE_DIR.glob("*.png"))
)

if len(image_files) == 0:

    raise RuntimeError(
        f"No validation images found in:\n{IMAGE_DIR}"
    )


print("=" * 70)
print("YOLOv13 ORACLE ROUTING EXPERIMENT")
print("=" * 70)

print(
    f"Validation images : {len(image_files)}"
)

print(
    f"Image size        : {IMG_SIZE}"
)

print(
    f"Confidence        : {CONF}"
)

print(
    f"IoU threshold     : {IOU_THRESHOLD}"
)

print(
    f"Device            : {DEVICE}"
)

print()


# ============================================================
# LOAD MODELS
# ============================================================

loaded_models = {}

for name, path in MODELS.items():

    if not path.exists():

        raise FileNotFoundError(
            f"Missing {name} model:\n{path}"
        )

    print(
        f"Loading YOLOv13-{name}: {path}"
    )

    loaded_models[name] = YOLO(
        str(path)
    )

print()
print("All models loaded.")
print()


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CSV_PATH = (
    OUTPUT_DIR
    / "oracle_routing_per_image.csv"
)


# ============================================================
# RUN EXPERIMENT
# ============================================================

rows = []

for image_idx, image_path in enumerate(
    image_files,
    start=1
):

    label_path = (
        LABEL_DIR
        / f"{image_path.stem}.txt"
    )

    gt_labels = load_labels(
        label_path
    )

    model_results = {}

    # --------------------------------------------------------
    # Run every model
    # --------------------------------------------------------

    for model_name in MODEL_ORDER:

        model = loaded_models[
            model_name
        ]

        result = model.predict(
            source=str(image_path),
            imgsz=IMG_SIZE,
            conf=CONF,
            device=DEVICE,
            verbose=False
        )[0]

        predictions = []

        if (
            result.boxes is not None
            and len(result.boxes) > 0
        ):

            boxes = (
                result.boxes.xyxy
                .cpu()
                .numpy()
            )

            classes = (
                result.boxes.cls
                .cpu()
                .numpy()
            )

            confs = (
                result.boxes.conf
                .cpu()
                .numpy()
            )

            for box, cls, conf in zip(
                boxes,
                classes,
                confs
            ):

                predictions.append({
                    "cls": int(cls),
                    "box": box,
                    "conf": float(conf)
                })

        # ----------------------------------------------------
        # IMPORTANT:
        # result.orig_shape = (height, width)
        # ----------------------------------------------------

        (
            tp,
            fp,
            fn,
            precision,
            recall,
            f1
        ) = calculate_metrics(
            gt_labels,
            predictions,
            result.orig_shape
        )

        model_results[
            model_name
        ] = {

            "tp": tp,

            "fp": fp,

            "fn": fn,

            "precision": precision,

            "recall": recall,

            "f1": f1,

            "predictions": len(
                predictions
            )
        }

    # ========================================================
    # ORACLE SELECTION
    # ========================================================

    # Select model with highest F1.
    # MODEL_ORDER provides deterministic tie-breaking.
    best_model = max(
        MODEL_ORDER,
        key=lambda name: (
            model_results[name]["f1"],
            -MODEL_ORDER.index(name)
        )
    )

    selected = model_results[
        best_model
    ]

    # ========================================================
    # STORE IMAGE RESULT
    # ========================================================

    row = {

        "image": image_path.name,

        "gt_instances": len(
            gt_labels
        ),

        "selected_model": (
            f"YOLOv13-{best_model}"
        ),

        "oracle_precision": (
            selected["precision"]
        ),

        "oracle_recall": (
            selected["recall"]
        ),

        "oracle_f1": (
            selected["f1"]
        ),

    }

    # Store metrics for every model
    for model_name in MODEL_ORDER:

        result_data = model_results[
            model_name
        ]

        prefix = (
            f"YOLOv13-{model_name}"
        )

        row[
            f"{prefix}_tp"
        ] = result_data["tp"]

        row[
            f"{prefix}_fp"
        ] = result_data["fp"]

        row[
            f"{prefix}_fn"
        ] = result_data["fn"]

        row[
            f"{prefix}_precision"
        ] = result_data["precision"]

        row[
            f"{prefix}_recall"
        ] = result_data["recall"]

        row[
            f"{prefix}_f1"
        ] = result_data["f1"]

        row[
            f"{prefix}_predictions"
        ] = result_data[
            "predictions"
        ]

    rows.append(row)

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if (
        image_idx % 25 == 0
        or image_idx == len(image_files)
    ):

        percentage = (
            image_idx
            / len(image_files)
            * 100
        )

        print(
            f"Processed "
            f"{image_idx}/{len(image_files)} "
            f"({percentage:.1f}%)"
        )


# ============================================================
# SAVE CSV
# ============================================================

fieldnames = list(
    rows[0].keys()
)

with open(
    CSV_PATH,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(rows)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("ORACLE ROUTING SUMMARY")
print("=" * 70)

for model_name in MODEL_ORDER:

    prefix = (
        f"YOLOv13-{model_name}"
    )

    precision = np.mean([
        row[
            f"{prefix}_precision"
        ]
        for row in rows
    ])

    recall = np.mean([
        row[
            f"{prefix}_recall"
        ]
        for row in rows
    ])

    f1 = np.mean([
        row[
            f"{prefix}_f1"
        ]
        for row in rows
    ])

    print()
    print(prefix)

    print(
        f"Mean Precision : "
        f"{precision:.4f}"
    )

    print(
        f"Mean Recall    : "
        f"{recall:.4f}"
    )

    print(
        f"Mean F1        : "
        f"{f1:.4f}"
    )


# ============================================================
# ORACLE ROUTING DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("ORACLE ROUTING DISTRIBUTION")
print("=" * 70)

selection_counts = {}

for model_name in MODEL_ORDER:

    count = sum(
        row["selected_model"]
        == f"YOLOv13-{model_name}"
        for row in rows
    )

    selection_counts[
        model_name
    ] = count

    percentage = (
        count
        / len(rows)
        * 100
    )

    print(
        f"YOLOv13-{model_name}: "
        f"{count:4d} images "
        f"({percentage:6.2f}%)"
    )


# ============================================================
# ORACLE PERFORMANCE
# ============================================================

oracle_precision = np.mean([
    row[
        "oracle_precision"
    ]
    for row in rows
])

oracle_recall = np.mean([
    row[
        "oracle_recall"
    ]
    for row in rows
])

oracle_f1 = np.mean([
    row[
        "oracle_f1"
    ]
    for row in rows
])


print()
print("=" * 70)
print("ORACLE PERFORMANCE")
print("=" * 70)

print(
    f"Mean Precision : "
    f"{oracle_precision:.4f}"
)

print(
    f"Mean Recall    : "
    f"{oracle_recall:.4f}"
)

print(
    f"Mean F1        : "
    f"{oracle_f1:.4f}"
)

print()
print("CSV saved to:")
print(CSV_PATH)

print()
print("Experiment completed.")
print("=" * 70)
from pathlib import Path
import csv
import numpy as np
from ultralytics import YOLO

# ============================================================
