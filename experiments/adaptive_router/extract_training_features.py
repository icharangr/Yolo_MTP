from pathlib import Path
import csv
import numpy as np

from PIL import Image
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_DIR = Path(
    "/home/charan/thesis/datasets/VisDrone/images/train"
)

LABEL_DIR = Path(
    "/home/charan/thesis/datasets/VisDrone/labels/train"
)

MODELS = {
    "N": Path(
        "/home/charan/thesis/runs/"
        "yolov13n_visdrone_institute_300ep/"
        "weights/best.pt"
    ),

    "S": Path(
        "/home/charan/thesis/runs/"
        "yolov13s_visdrone_institute_200ep2/"
        "weights/best.pt"
    ),

    "L": Path(
        "/home/charan/thesis/runs/"
        "yolov13l_visdrone_institute_200ep/"
        "weights/best.pt"
    ),

    "X": Path(
        "/home/charan/thesis/runs/"
        "yolov13x_visdrone_institute_200ep/"
        "weights/best.pt"
    ),
}


OUTPUT_DIR = Path(
    "/home/charan/thesis/yolov13/"
    "experiments/adaptive_router"
)

OUTPUT_CSV = (
    OUTPUT_DIR /
    "router_training_features.csv"
)


# ------------------------------------------------------------
# IMPORTANT
# ------------------------------------------------------------
#
# True  = completely rebuild the CSV
# False = resume from an existing checkpoint
#
# Because the previous CSV was generated with the wrong
# coordinate handling, KEEP THIS TRUE FOR THE FIRST CORRECT RUN.
#
# After a valid CSV has been produced, change to False.
# ------------------------------------------------------------

FORCE_REBUILD = True


IMG_SIZE = 640
CONF = 0.25
IOU_THRESHOLD = 0.50
DEVICE = "cuda:0"

MODEL_ORDER = [
    "N",
    "S",
    "L",
    "X",
]

CHECKPOINT_EVERY = 25


# ============================================================
# LABEL LOADING
# ============================================================

def load_labels(label_file):
    """
    Read YOLO-format labels.

    Format:
        class x_center y_center width height

    Coordinates are normalized to [0,1].
    """

    if not label_file.exists():
        return []

    labels = []

    with open(label_file, "r") as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            try:

                cls, xc, yc, w, h = map(
                    float,
                    parts
                )

            except ValueError:
                continue

            labels.append(
                {
                    "cls": int(cls),
                    "box": np.array(
                        [xc, yc, w, h],
                        dtype=np.float32
                    ),
                }
            )

    return labels


# ============================================================
# GT BOX CONVERSION
# ============================================================

def gt_xywh_to_xyxy_pixels(
    box,
    image_width,
    image_height
):
    """
    Convert normalized YOLO xywh coordinates
    into pixel xyxy coordinates.
    """

    xc, yc, w, h = box

    xc *= image_width
    yc *= image_height

    w *= image_width
    h *= image_height

    x1 = xc - w / 2.0
    y1 = yc - h / 2.0

    x2 = xc + w / 2.0
    y2 = yc + h / 2.0

    return np.array(
        [x1, y1, x2, y2],
        dtype=np.float32
    )


# ============================================================
# IOU
# ============================================================

def iou_xyxy(a, b):

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)

    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(
        0.0,
        ix2 - ix1
    )

    ih = max(
        0.0,
        iy2 - iy1
    )

    intersection = iw * ih

    area_a = (
        max(0.0, ax2 - ax1)
        *
        max(0.0, ay2 - ay1)
    )

    area_b = (
        max(0.0, bx2 - bx1)
        *
        max(0.0, by2 - by1)
    )

    union = (
        area_a
        +
        area_b
        -
        intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# METRIC CALCULATION
# ============================================================

def calculate_metrics(
    gt_labels,
    predictions,
    image_width,
    image_height
):
    """
    Class-aware greedy matching.

    IMPORTANT:
    GT coordinates are normalized.
    Predictions from Ultralytics are pixel xyxy.

    Therefore GT boxes are converted to pixel xyxy
    before IoU calculation.
    """

    gt = []

    for item in gt_labels:

        gt_box = gt_xywh_to_xyxy_pixels(
            item["box"],
            image_width,
            image_height
        )

        gt.append(
            (
                item["cls"],
                gt_box
            )
        )

    pred = []

    for item in predictions:

        pred.append(
            (
                int(item["cls"]),
                np.asarray(
                    item["box"],
                    dtype=np.float32
                ),
                float(item["conf"])
            )
        )

    # Highest confidence predictions first
    pred.sort(
        key=lambda x: x[2],
        reverse=True
    )

    matched_gt = set()

    tp = 0
    fp = 0

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

    fn = len(gt) - tp

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    if precision + recall > 0:

        f1 = (
            2.0
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
# MODEL INFERENCE
# ============================================================

def run_model(
    model,
    image_path
):
    """
    Run one YOLO model and return predictions.

    result.boxes.xyxy is in pixel coordinates.
    """

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
            .detach()
            .cpu()
            .numpy()
        )

        classes = (
            result.boxes.cls
            .detach()
            .cpu()
            .numpy()
        )

        confs = (
            result.boxes.conf
            .detach()
            .cpu()
            .numpy()
        )

        for box, cls, conf in zip(
            boxes,
            classes,
            confs
        ):

            predictions.append(
                {
                    "cls": int(cls),
                    "box": box.astype(
                        np.float32
                    ),
                    "conf": float(conf),
                }
            )

    return predictions


# ============================================================
# ROUTER FEATURE EXTRACTION
# ============================================================

def extract_router_features(
    predictions,
    image_width,
    image_height
):
    """
    Extract inference-time features from YOLOv13-N.

    These features are available to the adaptive router
    without using ground-truth information.
    """

    if len(predictions) == 0:

        return {
            "probe_num_predictions": 0,
            "probe_mean_conf": 0.0,
            "probe_max_conf": 0.0,
            "probe_median_conf": 0.0,

            "probe_mean_area_ratio": 0.0,
            "probe_median_area_ratio": 0.0,

            "probe_small_ratio": 0.0,
            "probe_medium_ratio": 0.0,
            "probe_large_ratio": 0.0,

            "probe_mean_width_ratio": 0.0,
            "probe_mean_height_ratio": 0.0,

            "probe_spatial_density": 0.0,
        }

    image_area = (
        float(image_width)
        *
        float(image_height)
    )

    image_diagonal = np.sqrt(
        image_width ** 2
        +
        image_height ** 2
    )

    confidences = []

    area_ratios = []

    widths = []
    heights = []

    centers_x = []
    centers_y = []

    small_count = 0
    medium_count = 0
    large_count = 0

    for item in predictions:

        x1, y1, x2, y2 = item["box"]

        width = max(
            0.0,
            x2 - x1
        )

        height = max(
            0.0,
            y2 - y1
        )

        area = width * height

        area_ratio = (
            area / image_area
            if image_area > 0
            else 0.0
        )

        center_x = (
            (x1 + x2) / 2.0
        )

        center_y = (
            (y1 + y2) / 2.0
        )

        confidences.append(
            item["conf"]
        )

        area_ratios.append(
            area_ratio
        )

        widths.append(
            width / image_width
        )

        heights.append(
            height / image_height
        )

        centers_x.append(
            center_x / image_width
        )

        centers_y.append(
            center_y / image_height
        )

        # ----------------------------------------------------
        # Object scale categories
        #
        # Based on normalized object area.
        # ----------------------------------------------------

        if area_ratio < 0.01:

            small_count += 1

        elif area_ratio < 0.05:

            medium_count += 1

        else:

            large_count += 1

    num_predictions = len(
        predictions
    )

    # --------------------------------------------------------
    # Spatial density
    #
    # Number of predictions per normalized image area.
    # --------------------------------------------------------

    spatial_density = (
        num_predictions / image_area
        * 10000.0
        if image_area > 0
        else 0.0
    )

    return {
        "probe_num_predictions":
            num_predictions,

        "probe_mean_conf":
            float(
                np.mean(confidences)
            ),

        "probe_max_conf":
            float(
                np.max(confidences)
            ),

        "probe_median_conf":
            float(
                np.median(confidences)
            ),

        "probe_mean_area_ratio":
            float(
                np.mean(area_ratios)
            ),

        "probe_median_area_ratio":
            float(
                np.median(area_ratios)
            ),

        "probe_small_ratio":
            float(
                small_count
                / num_predictions
            ),

        "probe_medium_ratio":
            float(
                medium_count
                / num_predictions
            ),

        "probe_large_ratio":
            float(
                large_count
                / num_predictions
            ),

        "probe_mean_width_ratio":
            float(
                np.mean(widths)
            ),

        "probe_mean_height_ratio":
            float(
                np.mean(heights)
            ),

        "probe_spatial_density":
            float(
                spatial_density
            ),
    }


# ============================================================
# DATASET
# ============================================================

image_files = []

for extension in [
    "*.jpg",
    "*.jpeg",
    "*.png"
]:

    image_files.extend(
        IMAGE_DIR.glob(extension)
    )

image_files = sorted(
    image_files
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# START
# ============================================================

print("=" * 70)
print(
    "EXPERIMENT C1 — "
    "ADAPTIVE ROUTER TRAINING FEATURE EXTRACTION"
)
print("=" * 70)

print(
    f"Training images : "
    f"{len(image_files)}"
)

print(
    f"Image size      : "
    f"{IMG_SIZE}"
)

print(
    f"Confidence      : "
    f"{CONF}"
)

print(
    f"IoU threshold   : "
    f"{IOU_THRESHOLD}"
)

print(
    f"Device          : "
    f"{DEVICE}"
)

print()


# ============================================================
# CHECK DATASET
# ============================================================

if len(image_files) == 0:

    raise RuntimeError(
        f"No images found in:\n"
        f"{IMAGE_DIR}"
    )


# ============================================================
# CHECK MODEL FILES
# ============================================================

for model_name, model_path in MODELS.items():

    if not model_path.exists():

        raise FileNotFoundError(
            f"\nMissing "
            f"YOLOv13-{model_name} model:\n"
            f"{model_path}"
        )


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading models...")
print()

loaded_models = {}

for model_name in MODEL_ORDER:

    model_path = MODELS[
        model_name
    ]

    print(
        f"Loading YOLOv13-{model_name}: "
        f"{model_path}"
    )

    loaded_models[model_name] = YOLO(
        str(model_path)
    )

print()
print("All models loaded.")
print()


# ============================================================
# CSV FIELD NAMES
# ============================================================

fieldnames = [

    "image",

    # --------------------------------------------------------
    # Router target
    # --------------------------------------------------------

    "oracle_model",

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    "gt_instances",

    # --------------------------------------------------------
    # YOLOv13-N inference features
    # --------------------------------------------------------

    "probe_num_predictions",
    "probe_mean_conf",
    "probe_max_conf",
    "probe_median_conf",

    "probe_mean_area_ratio",
    "probe_median_area_ratio",

    "probe_small_ratio",
    "probe_medium_ratio",
    "probe_large_ratio",

    "probe_mean_width_ratio",
    "probe_mean_height_ratio",

    "probe_spatial_density",

    # --------------------------------------------------------
    # YOLOv13-N metrics
    # --------------------------------------------------------

    "N_precision",
    "N_recall",
    "N_f1",

    # --------------------------------------------------------
    # YOLOv13-S metrics
    # --------------------------------------------------------

    "S_precision",
    "S_recall",
    "S_f1",

    # --------------------------------------------------------
    # YOLOv13-L metrics
    # --------------------------------------------------------

    "L_precision",
    "L_recall",
    "L_f1",

    # --------------------------------------------------------
    # YOLOv13-X metrics
    # --------------------------------------------------------

    "X_precision",
    "X_recall",
    "X_f1",
]


# ============================================================
# RESUME SUPPORT
# ============================================================

existing_rows = {}


if (
    OUTPUT_CSV.exists()
    and not FORCE_REBUILD
):

    print(
        "Existing checkpoint found:"
    )

    print(
        OUTPUT_CSV
    )

    print()

    with open(
        OUTPUT_CSV,
        "r",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            existing_rows[
                row["image"]
            ] = row

    print(
        f"Recovered "
        f"{len(existing_rows)} "
        f"completed images."
    )

    print()


elif (
    OUTPUT_CSV.exists()
    and FORCE_REBUILD
):

    print(
        "FORCE_REBUILD=True"
    )

    print(
        "Existing CSV will be "
        "replaced with corrected results."
    )

    print()


else:

    print(
        "No existing checkpoint found."
    )

    print()


rows = dict(
    existing_rows
)


# ============================================================
# PROCESS DATASET
# ============================================================

total = len(
    image_files
)


for image_idx, image_path in enumerate(
    image_files,
    start=1
):

    image_name = image_path.name

    # --------------------------------------------------------
    # RESUME
    # --------------------------------------------------------

    if image_name in rows:

        if (
            image_idx % CHECKPOINT_EVERY == 0
            or image_idx == total
        ):

            print(
                f"Skipping completed "
                f"{image_idx}/{total}"
            )

        continue


    # ========================================================
    # IMAGE DIMENSIONS
    # ========================================================

    try:

        with Image.open(
            image_path
        ) as img:

            image_width, image_height = (
                img.size
            )

    except Exception as e:

        print(
            f"WARNING: Could not read "
            f"image dimensions for "
            f"{image_name}: {e}"
        )

        image_width = IMG_SIZE
        image_height = IMG_SIZE


    # ========================================================
    # LOAD GT
    # ========================================================

    label_path = (
        LABEL_DIR
        /
        f"{image_path.stem}.txt"
    )

    gt_labels = load_labels(
        label_path
    )


    # ========================================================
    # RUN ALL MODELS
    # ========================================================

    model_results = {}

    n_predictions = []


    for model_name in MODEL_ORDER:

        predictions = run_model(
            loaded_models[model_name],
            image_path
        )


        # ----------------------------------------------------
        # CORRECTED METRIC CALCULATION
        #
        # GT normalized xywh
        #        ↓
        # pixel xyxy
        #
        # Prediction pixel xyxy
        #        ↓
        # direct IoU
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
            image_width,
            image_height
        )


        model_results[
            model_name
        ] = {

            "tp": tp,
            "fp": fp,
            "fn": fn,

            "precision":
                precision,

            "recall":
                recall,

            "f1":
                f1,
        }


        # ----------------------------------------------------
        # Save YOLOv13-N predictions
        # ----------------------------------------------------

        if model_name == "N":

            n_predictions = (
                predictions
            )


    # ========================================================
    # ORACLE TARGET SELECTION
    # ========================================================

    # Start with smallest model.
    # This provides deterministic tie breaking.

    best_model = MODEL_ORDER[0]

    best_f1 = model_results[
        best_model
    ]["f1"]


    for model_name in MODEL_ORDER[1:]:

        current_f1 = model_results[
            model_name
        ]["f1"]


        if current_f1 > best_f1:

            best_f1 = current_f1

            best_model = model_name


    # ========================================================
    # EXTRACT N-PROBE FEATURES
    # ========================================================

    features = extract_router_features(
        n_predictions,
        image_width,
        image_height
    )


    # ========================================================
    # CREATE ROW
    # ========================================================

    row = {

        "image":
            image_name,

        # Oracle target
        "oracle_model":
            f"YOLOv13-{best_model}",

        # GT complexity
        "gt_instances":
            len(gt_labels),

        # Router features
        **features,

        # N
        "N_precision":
            model_results["N"][
                "precision"
            ],

        "N_recall":
            model_results["N"][
                "recall"
            ],

        "N_f1":
            model_results["N"][
                "f1"
            ],

        # S
        "S_precision":
            model_results["S"][
                "precision"
            ],

        "S_recall":
            model_results["S"][
                "recall"
            ],

        "S_f1":
            model_results["S"][
                "f1"
            ],

        # L
        "L_precision":
            model_results["L"][
                "precision"
            ],

        "L_recall":
            model_results["L"][
                "recall"
            ],

        "L_f1":
            model_results["L"][
                "f1"
            ],

        # X
        "X_precision":
            model_results["X"][
                "precision"
            ],

        "X_recall":
            model_results["X"][
                "recall"
            ],

        "X_f1":
            model_results["X"][
                "f1"
            ],
    }


    rows[
        image_name
    ] = row


    # ========================================================
    # CHECKPOINT
    # ========================================================

    if (
        image_idx % CHECKPOINT_EVERY == 0
        or image_idx == total
    ):

        with open(
            OUTPUT_CSV,
            "w",
            newline=""
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames
            )

            writer.writeheader()

            for sorted_name in sorted(
                rows.keys()
            ):

                writer.writerow(
                    rows[
                        sorted_name
                    ]
                )


        print(
            f"Processed "
            f"{image_idx}/{total} "
            f"("
            f"{100.0 * image_idx / total:.1f}"
            f"%)"
            f" | checkpoint saved"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()

print("=" * 70)
print(
    "EXPERIMENT C1 EXTRACTION COMPLETE"
)
print("=" * 70)

print(
    f"Images processed : "
    f"{len(rows)}"
)

print(
    f"CSV saved to     : "
    f"{OUTPUT_CSV}"
)

print()


# ============================================================
# ROUTING DISTRIBUTION
# ============================================================

distribution = {}


for row in rows.values():

    model = row[
        "oracle_model"
    ]

    distribution[
        model
    ] = (
        distribution.get(
            model,
            0
        )
        + 1
    )


print(
    "ORACLE ROUTING TARGET DISTRIBUTION"
)

print(
    "-" * 40
)


for model in [
    "YOLOv13-N",
    "YOLOv13-S",
    "YOLOv13-L",
    "YOLOv13-X",
]:

    count = distribution.get(
        model,
        0
    )

    percentage = (
        100.0
        * count
        / len(rows)
        if len(rows) > 0
        else 0.0
    )

    print(
        f"{model}: "
        f"{count:5d} "
        f"({percentage:6.2f}%)"
    )


# ============================================================
# FINAL MODEL METRICS
# ============================================================

print()
print(
    "MEAN MODEL METRICS"
)

print(
    "-" * 40
)


for model_name in [
    "N",
    "S",
    "L",
    "X"
]:

    precisions = [
        float(
            row[
                f"{model_name}_precision"
            ]
        )
        for row in rows.values()
    ]

    recalls = [
        float(
            row[
                f"{model_name}_recall"
            ]
        )
        for row in rows.values()
    ]

    f1s = [
        float(
            row[
                f"{model_name}_f1"
            ]
        )
        for row in rows.values()
    ]


    print(
        f"\nYOLOv13-{model_name}"
    )

    print(
        f"Mean Precision : "
        f"{np.mean(precisions):.4f}"
    )

    print(
        f"Mean Recall    : "
        f"{np.mean(recalls):.4f}"
    )

    print(
        f"Mean F1        : "
        f"{np.mean(f1s):.4f}"
    )


# ============================================================
# ORACLE PERFORMANCE
# ============================================================

oracle_f1s = []

oracle_precisions = []

oracle_recalls = []


for row in rows.values():

    selected = row[
        "oracle_model"
    ]

    prefix = selected.replace(
        "YOLOv13-",
        ""
    )

    oracle_precisions.append(
        float(
            row[
                f"{prefix}_precision"
            ]
        )
    )

    oracle_recalls.append(
        float(
            row[
                f"{prefix}_recall"
            ]
        )
    )

    oracle_f1s.append(
        float(
            row[
                f"{prefix}_f1"
            ]
        )
    )


print()
print("=" * 70)
print(
    "ORACLE PERFORMANCE"
)
print("=" * 70)

print(
    f"Mean Precision : "
    f"{np.mean(oracle_precisions):.4f}"
)

print(
    f"Mean Recall    : "
    f"{np.mean(oracle_recalls):.4f}"
)

print(
    f"Mean F1        : "
    f"{np.mean(oracle_f1s):.4f}"
)

print()
print(
    "Experiment completed."
)
print("=" * 70)
