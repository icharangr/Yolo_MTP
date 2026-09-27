#!/usr/bin/env python3

import csv
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    "/home/charan/thesis/yolov13"
)

# ------------------------------------------------------------
# Validation dataset
# ------------------------------------------------------------
#
# IMPORTANT:
# Keep these two paths the same as your previous C3 script.
#
# If your current evaluate_adaptive_router.py already has
# correct paths here, copy those exact paths.
# ------------------------------------------------------------

IMAGE_DIR = Path(
    "/home/charan/thesis/datasets/VisDrone/VisDrone2019-DET-val/images"
)

LABEL_DIR = Path(
    "/home/charan/thesis/datasets/VisDrone/VisDrone2019-DET-val/annotations"
)


# ------------------------------------------------------------
# Router
# ------------------------------------------------------------

ROUTER_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "adaptive_router"
    / "router_model"
    / "random_forest_router.joblib"
)

FEATURE_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "adaptive_router"
    / "router_model"
    / "router_features.json"
)


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "adaptive_router"
    / "c3_results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "adaptive_router_results.csv"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "adaptive_router_metrics.json"
)


# ------------------------------------------------------------
# Models
# ------------------------------------------------------------

MODEL_PATHS = {

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


MODEL_ORDER = [
    "N",
    "S",
    "L",
    "X",
]


# ------------------------------------------------------------
# Inference
# ------------------------------------------------------------

IMG_SIZE = 640

CONF = 0.25

IOU_THRESHOLD = 0.50

DEVICE = "cuda:0"


# ------------------------------------------------------------
# Debug
# ------------------------------------------------------------

DEBUG_IMAGES = 3

DEBUG_ENABLED = True


# ============================================================
# DATASET DISCOVERY
# ============================================================

IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".JPG",
    ".JPEG",
    ".PNG",
]


def find_image_files():

    if not IMAGE_DIR.exists():

        raise FileNotFoundError(
            "\nValidation image directory does not exist:\n"
            f"{IMAGE_DIR}\n\n"
            "If your previous C3 script used a different "
            "IMAGE_DIR, keep that path."
        )

    files = []

    for extension in IMAGE_EXTENSIONS:

        files.extend(
            IMAGE_DIR.glob(
                f"*{extension}"
            )
        )

    files = sorted(
        files
    )

    if len(files) == 0:

        raise RuntimeError(
            "\nNo validation images found in:\n"
            f"{IMAGE_DIR}"
        )

    return files


image_files = find_image_files()


# ============================================================
# LABEL LOADING
# ============================================================
def load_labels(
    label_file
):
    """
    Load native VisDrone DET annotations.

    Native VisDrone format:

        bbox_left,
        bbox_top,
        bbox_width,
        bbox_height,
        score,
        category,
        truncation,
        occlusion

    Returned boxes are pixel-coordinate xyxy boxes.
    Class IDs are converted from VisDrone 1-10
    to YOLO 0-9.
    """

    if not label_file.exists():
        return []

    labels = []

    with open(
        label_file,
        "r"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            parts = line.split(",")

            if len(parts) != 8:
                continue

            try:

                x = float(parts[0])
                y = float(parts[1])
                w = float(parts[2])
                h = float(parts[3])

                score = float(parts[4])
                category = int(parts[5])
                truncation = int(parts[6])
                occlusion = int(parts[7])

            except ValueError:

                continue

            # ------------------------------------------------
            # Ignore category 0
            # ------------------------------------------------

            if category == 0:
                continue

            # ------------------------------------------------
            # Keep valid VisDrone detection categories
            # 1 ... 10
            # ------------------------------------------------

            if category < 1 or category > 10:
                continue

            # ------------------------------------------------
            # VisDrone -> YOLO class numbering
            #
            # 1 -> 0
            # 2 -> 1
            # ...
            # 10 -> 9
            # ------------------------------------------------

            cls = category - 1

            # ------------------------------------------------
            # Convert xywh -> xyxy
            #
            # IMPORTANT:
            # These are ALREADY pixel coordinates.
            # ------------------------------------------------

            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h

            labels.append(
                {
                    "cls": cls,
                    "box": np.array(
                        [
                            x1,
                            y1,
                            x2,
                            y2
                        ],
                        dtype=np.float32
                    )
                }
            )

    return labels
# ============================================================
# BOX CONVERSION
# ============================================================

def xywh_to_xyxy(
    box
):

    xc, yc, w, h = box

    return np.array(
        [
            xc - w / 2.0,
            yc - h / 2.0,
            xc + w / 2.0,
            yc + h / 2.0,
        ],
        dtype=np.float32
    )


def gt_xywh_to_xyxy_pixels(
    box,
    image_width,
    image_height
):

    xc, yc, w, h = box

    return np.array(
        [
            (xc - w / 2.0)
            * image_width,

            (yc - h / 2.0)
            * image_height,

            (xc + w / 2.0)
            * image_width,

            (yc + h / 2.0)
            * image_height,
        ],
        dtype=np.float32
    )


# ============================================================
# IOU
# ============================================================

def iou_xyxy(
    a,
    b
):

    ax1, ay1, ax2, ay2 = a

    bx1, by1, bx2, by2 = b

    ix1 = max(
        ax1,
        bx1
    )

    iy1 = max(
        ay1,
        by1
    )

    ix2 = min(
        ax2,
        bx2
    )

    iy2 = min(
        ay2,
        by2
    )

    iw = max(
        0.0,
        ix2 - ix1
    )

    ih = max(
        0.0,
        iy2 - iy1
    )

    intersection = (
        iw * ih
    )

    area_a = (
        max(
            0.0,
            ax2 - ax1
        )
        *
        max(
            0.0,
            ay2 - ay1
        )
    )

    area_b = (
        max(
            0.0,
            bx2 - bx1
        )
        *
        max(
            0.0,
            by2 - by1
        )
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

    return (
        intersection
        / union
    )


# ============================================================
# CORRECTED METRIC CALCULATION
# ============================================================

# ============================================================
# METRIC CALCULATION
# ============================================================

def calculate_metrics(
    gt_labels,
    predictions,
    image_width=None,
    image_height=None
):
    """
    Calculate class-aware detection metrics.

    GT:
        Native VisDrone annotations converted to pixel xyxy.

    Predictions:
        Ultralytics predictions already in pixel xyxy.

    image_width and image_height are accepted for compatibility
    with the existing evaluation loop but are not required because
    both GT and predictions are already in pixel coordinates.
    """

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    gt = []

    for item in gt_labels:

        gt.append(
            (
                int(item["cls"]),
                np.asarray(
                    item["box"],
                    dtype=np.float32
                )
            )
        )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

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

    # Highest-confidence predictions first
    pred.sort(
        key=lambda x: x[2],
        reverse=True
    )

    matched_gt = set()

    tp = 0
    fp = 0

    # --------------------------------------------------------
    # Class-aware greedy matching
    # --------------------------------------------------------

    for pred_cls, pred_box, pred_conf in pred:

        best_iou = 0.0
        best_gt = None

        for gt_index, (gt_cls, gt_box) in enumerate(gt):

            if gt_index in matched_gt:
                continue

            # Classes must match
            if pred_cls != gt_cls:
                continue

            current_iou = iou_xyxy(
                pred_box,
                gt_box
            )

            if (
                current_iou >= IOU_THRESHOLD
                and current_iou > best_iou
            ):

                best_iou = current_iou
                best_gt = gt_index

        if best_gt is not None:

            tp += 1

            matched_gt.add(
                best_gt
            )

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

    f1 = (
        2.0 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return (
        tp,
        fp,
        fn,
        precision,
        recall,
        f1
    )

# ============================================================
# ROUTER FEATURE EXTRACTION
# ============================================================

def extract_router_features(
    predictions,
    image_width,
    image_height
):
    """
    Extract the 12 features expected by the trained
    Random Forest adaptive router.

    All prediction boxes are pixel-coordinate xyxy.
    """

    num_predictions = len(predictions)

    # --------------------------------------------------------
    # Empty prediction case
    # --------------------------------------------------------

    if num_predictions == 0:

        return {
            "probe_num_predictions": 0.0,
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

    # --------------------------------------------------------
    # Image area
    # --------------------------------------------------------

    image_area = (
        float(image_width)
        * float(image_height)
    )

    if image_area <= 0:
        image_area = 1.0

    # --------------------------------------------------------
    # Feature containers
    # --------------------------------------------------------

    confidences = []

    area_ratios = []

    width_ratios = []

    height_ratios = []

    small_count = 0

    medium_count = 0

    large_count = 0

    # --------------------------------------------------------
    # Process predictions
    # --------------------------------------------------------

    for prediction in predictions:

        box = np.asarray(
            prediction["box"],
            dtype=np.float32
        )

        conf = float(
            prediction["conf"]
        )

        x1, y1, x2, y2 = box

        width = max(
            0.0,
            float(x2 - x1)
        )

        height = max(
            0.0,
            float(y2 - y1)
        )

        area = (
            width
            * height
        )

        area_ratio = (
            area / image_area
        )

        width_ratio = (
            width / float(image_width)
            if image_width > 0
            else 0.0
        )

        height_ratio = (
            height / float(image_height)
            if image_height > 0
            else 0.0
        )

        confidences.append(conf)

        area_ratios.append(
            area_ratio
        )

        width_ratios.append(
            width_ratio
        )

        height_ratios.append(
            height_ratio
        )

        # ----------------------------------------------------
        # Object-size distribution
        #
        # Area ratio thresholds
        # ----------------------------------------------------

        if area_ratio < 0.01:

            small_count += 1

        elif area_ratio < 0.05:

            medium_count += 1

        else:

            large_count += 1

    # --------------------------------------------------------
    # Spatial density
    #
    # Number of detected objects per normalized image area.
    # --------------------------------------------------------

    density = (
        num_predictions
        / image_area
    )

    # --------------------------------------------------------
    # Return exactly the router feature schema
    # --------------------------------------------------------

    return {

        "probe_num_predictions":
            float(num_predictions),

        "probe_mean_conf":
            float(
                np.mean(
                    confidences
                )
            ),

        "probe_max_conf":
            float(
                np.max(
                    confidences
                )
            ),

        "probe_median_conf":
            float(
                np.median(
                    confidences
                )
            ),

        "probe_mean_area_ratio":
            float(
                np.mean(
                    area_ratios
                )
            ),

        "probe_median_area_ratio":
            float(
                np.median(
                    area_ratios
                )
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
                np.mean(
                    width_ratios
                )
            ),

        "probe_mean_height_ratio":
            float(
                np.mean(
                    height_ratios
                )
            ),

        "probe_spatial_density":
            float(
                density
            ),
    }


# ============================================================
# MODEL INFERENCE
# ============================================================

def run_model(
    model,
    image_path
):

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

        for (
            box,
            cls,
            conf
        ) in zip(
            boxes,
            classes,
            confs
        ):

            predictions.append({

                "cls":
                    int(cls),

                "box":
                    np.asarray(
                        box,
                        dtype=np.float32
                    ),

                "conf":
                    float(conf)
            })

    return predictions


# ============================================================
# TIMED INFERENCE
# ============================================================

def timed_run_model(
    model,
    image_path
):

    if torch.cuda.is_available():

        torch.cuda.synchronize()

    start = time.perf_counter()

    predictions = run_model(
        model,
        image_path
    )

    if torch.cuda.is_available():

        torch.cuda.synchronize()

    elapsed = (
        time.perf_counter()
        - start
    )

    return (
        predictions,
        elapsed
    )


# ============================================================
# HEADER
# ============================================================

print("=" * 70)

print(
    "C3 — LEARNED ADAPTIVE ROUTER EVALUATION"
)

print("=" * 70)

print()

print(
    f"Validation images : "
    f"{len(image_files)}"
)

print(
    f"Image size        : "
    f"{IMG_SIZE}"
)

print(
    f"Confidence        : "
    f"{CONF}"
)

print(
    f"IoU threshold     : "
    f"{IOU_THRESHOLD}"
)

print(
    f"Device            : "
    f"{DEVICE}"
)

print()


# ============================================================
# LOAD ROUTER
# ============================================================

if not ROUTER_PATH.exists():

    raise FileNotFoundError(
        f"Router not found:\n"
        f"{ROUTER_PATH}"
    )


router = joblib.load(
    ROUTER_PATH
)

print(
    "Router loaded:"
)

print(
    ROUTER_PATH
)

print()


# ============================================================
# LOAD FEATURES
# ============================================================

if FEATURE_PATH.exists():

    with open(
        FEATURE_PATH,
        "r"
    ) as f:

        FEATURE_COLUMNS = json.load(
            f
        )

else:

    FEATURE_COLUMNS = [

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
    ]


print(
    "Router features:"
)

for feature in FEATURE_COLUMNS:

    print(
        f"  {feature}"
    )

print()


# ============================================================
# LOAD DETECTORS
# ============================================================

print("=" * 70)

print(
    "LOADING DETECTORS"
)

print("=" * 70)


loaded_models = {}


for model_name in MODEL_ORDER:

    model_path = (
        MODEL_PATHS[
            model_name
        ]
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"Missing YOLOv13-{model_name}:\n"
            f"{model_path}"
        )

    print(
        f"Loading YOLOv13-{model_name}: "
        f"{model_path}"
    )

    loaded_models[
        model_name
    ] = YOLO(
        str(model_path)
    )


print()

print(
    "All detectors loaded."
)

print()


# ============================================================
# METRIC STORAGE
# ============================================================

fixed_metrics = {

    model_name: {

        "precision": [],
        "recall": [],
        "f1": [],
        "latency": [],

    }

    for model_name
    in MODEL_ORDER
}


adaptive_metrics = {

    "precision": [],
    "recall": [],
    "f1": [],
    "latency": [],
}


oracle_metrics = {

    "precision": [],
    "recall": [],
    "f1": [],
}


routing_distribution = {

    model_name: 0

    for model_name
    in MODEL_ORDER
}


# ============================================================
# RESULT ROWS
# ============================================================

result_rows = []


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

    # --------------------------------------------------------
    # IMAGE DIMENSIONS
    # --------------------------------------------------------

    from PIL import Image

    try:

        with Image.open(
            image_path
        ) as img:

            image_width, image_height = (
                img.size
            )

    except Exception:

        image_width = IMG_SIZE

        image_height = IMG_SIZE


    # --------------------------------------------------------
    # LOAD GT
    # --------------------------------------------------------

    label_path = (
        LABEL_DIR
        /
        f"{image_path.stem}.txt"
    )

    gt_labels = load_labels(
        label_path
    )

	    # --------------------------------------------------------
    # TEMPORARY GT DIAGNOSTIC
    # --------------------------------------------------------

    if image_path == image_files[0]:

        print()
        print("=" * 70)
        print("GT DIAGNOSTIC")
        print("=" * 70)

        print(
            "Image:",
            image_path.name
        )

        print(
            "Image size:",
            image_width,
            "x",
            image_height
        )

        print(
            "GT instances:",
            len(gt_labels)
        )

        for i, item in enumerate(
            gt_labels[:10]
        ):

            print(
                f"GT[{i}] "
                f"class={item['cls']} "
                f"box={item['box']}"
            )

        print("=" * 70)


    # --------------------------------------------------------
    # N-PROBE
    # --------------------------------------------------------

    n_predictions, n_latency = (
        timed_run_model(
            loaded_models["N"],
            image_path
        )
    )

    # --------------------------------------------------------
    # TEMPORARY PREDICTION DIAGNOSTIC
    # --------------------------------------------------------

    if image_path == image_files[0]:

        print()
        print("=" * 70)
        print("PREDICTION DIAGNOSTIC")
        print("=" * 70)

        print(
            "Predictions:",
            len(n_predictions)
        )

        for i, item in enumerate(
            n_predictions[:10]
        ):

            print(
                f"PRED[{i}] "
                f"class={item['cls']} "
                f"conf={item['conf']:.4f} "
                f"box={item['box']}"
            )

        print("=" * 70)

    if image_path == image_files[0]:

        print()
        print("=" * 70)
        print("BEST MATCH DIAGNOSTIC")
        print("=" * 70)

        for p_idx, prediction in enumerate(
            n_predictions[:10]
        ):

            best_iou = 0.0
            best_class_iou = 0.0
            best_class = None

            for gt_item in gt_labels:

                current_iou = iou_xyxy(
                    prediction["box"],
                    gt_item["box"]
                )

                if current_iou > best_iou:

                    best_iou = current_iou

                if (
                    prediction["cls"]
                    == gt_item["cls"]
                    and current_iou > best_class_iou
                ):

                    best_class_iou = current_iou
                    best_class = gt_item["cls"]

            print(
                f"PRED[{p_idx}] "
                f"class={prediction['cls']} "
                f"best_any_class_iou={best_iou:.4f} "
                f"best_same_class_iou={best_class_iou:.4f}"
            )

        print("=" * 70)

    # --------------------------------------------------------
    # N METRICS
    # --------------------------------------------------------

    (
        n_tp,
        n_fp,
        n_fn,
        n_precision,
        n_recall,
        n_f1
    ) = calculate_metrics(
        gt_labels,
        n_predictions
    )


    # --------------------------------------------------------
    # ROUTER FEATURES
    # --------------------------------------------------------

    features = extract_router_features(
        n_predictions,
        image_width,
        image_height
    )


    feature_vector = pd.DataFrame(
        [[
            features[
                col
            ]

            for col
            in FEATURE_COLUMNS
        ]],
        columns=FEATURE_COLUMNS
    )


    # --------------------------------------------------------
    # ROUTER PREDICTION
    # --------------------------------------------------------

    predicted_model_full = (
        router.predict(
            feature_vector
        )[0]
    )


    predicted_model = (
        str(
            predicted_model_full
        )
        .replace(
            "YOLOv13-",
            ""
        )
    )


    if predicted_model not in MODEL_ORDER:

        raise ValueError(
            "Invalid router prediction: "
            f"{predicted_model_full}"
        )


    routing_distribution[
        predicted_model
    ] += 1


    # --------------------------------------------------------
    # SELECTED MODEL
    # --------------------------------------------------------

    if predicted_model == "N":

        adaptive_predictions = (
            n_predictions
        )

        adaptive_latency = (
            n_latency
        )

    else:

        (
            selected_predictions,
            selected_latency
        ) = timed_run_model(
            loaded_models[
                predicted_model
            ],
            image_path
        )

        adaptive_predictions = (
            selected_predictions
        )

        # Actual deployed pipeline:
        #
        # N probe
        # +
        # selected detector

        adaptive_latency = (
            n_latency
            +
            selected_latency
        )


    # --------------------------------------------------------
    # ADAPTIVE METRICS
    # --------------------------------------------------------

    (
        adaptive_tp,
        adaptive_fp,
        adaptive_fn,
        adaptive_precision,
        adaptive_recall,
        adaptive_f1
    ) = calculate_metrics(
        gt_labels,
        adaptive_predictions
    )


    adaptive_metrics[
        "precision"
    ].append(
        adaptive_precision
    )

    adaptive_metrics[
        "recall"
    ].append(
        adaptive_recall
    )

    adaptive_metrics[
        "f1"
    ].append(
        adaptive_f1
    )

    adaptive_metrics[
        "latency"
    ].append(
        adaptive_latency
    )


    # --------------------------------------------------------
    # FIXED MODEL EVALUATION
    # --------------------------------------------------------

    model_results = {}


    for model_name in MODEL_ORDER:

        if model_name == "N":

            predictions = (
                n_predictions
            )

            latency = (
                n_latency
            )

        else:

            (
                predictions,
                latency
            ) = timed_run_model(
                loaded_models[
                    model_name
                ],
                image_path
            )


        (
            tp,
            fp,
            fn,
            precision,
            recall,
            f1
        ) = calculate_metrics(
            gt_labels,
            predictions
        )


        model_results[
            model_name
        ] = {

            "tp":
                tp,

            "fp":
                fp,

            "fn":
                fn,

            "precision":
                precision,

            "recall":
                recall,

            "f1":
                f1,

            "latency":
                latency,
        }


        fixed_metrics[
            model_name
        ]["precision"].append(
            precision
        )

        fixed_metrics[
            model_name
        ]["recall"].append(
            recall
        )

        fixed_metrics[
            model_name
        ]["f1"].append(
            f1
        )

        fixed_metrics[
            model_name
        ]["latency"].append(
            latency
        )


    # --------------------------------------------------------
    # ORACLE
    # --------------------------------------------------------

    oracle_model = (
        MODEL_ORDER[0]
    )

    oracle_f1 = (
        model_results[
            oracle_model
        ]["f1"]
    )


    for model_name in MODEL_ORDER[1:]:

        current_f1 = (
            model_results[
                model_name
            ]["f1"]
        )

        if current_f1 > oracle_f1:

            oracle_f1 = (
                current_f1
            )

            oracle_model = (
                model_name
            )


    oracle_precision = (
        model_results[
            oracle_model
        ]["precision"]
    )

    oracle_recall = (
        model_results[
            oracle_model
        ]["recall"]
    )


    oracle_metrics[
        "precision"
    ].append(
        oracle_precision
    )

    oracle_metrics[
        "recall"
    ].append(
        oracle_recall
    )

    oracle_metrics[
        "f1"
    ].append(
        oracle_f1
    )


    # --------------------------------------------------------
    # DEBUG FIRST THREE IMAGES
    # --------------------------------------------------------

    if (
        DEBUG_ENABLED
        and image_idx <= DEBUG_IMAGES
    ):

        for model_name in MODEL_ORDER:

            r = (
                model_results[
                    model_name
                ]
            )

            print()

            print(
                "-" * 50
            )

            print(
                f"DEBUG IMAGE "
                f"{image_idx}/"
                f"{DEBUG_IMAGES}: "
                f"{image_path.name}"
            )

            print(
                f"MODEL          : "
                f"YOLOv13-{model_name}"
            )

            print(
                f"GT instances   : "
                f"{len(gt_labels)}"
            )

            print(
                f"Predictions    : "
                f"{r['tp'] + r['fp']}"
            )

            print(
                f"TP             : "
                f"{r['tp']}"
            )

            print(
                f"FP             : "
                f"{r['fp']}"
            )

            print(
                f"FN             : "
                f"{r['fn']}"
            )

            print(
                f"Precision      : "
                f"{r['precision']:.6f}"
            )

            print(
                f"Recall         : "
                f"{r['recall']:.6f}"
            )

            print(
                f"F1             : "
                f"{r['f1']:.6f}"
            )

            print(
                f"Latency        : "
                f"{r['latency'] * 1000.0:.3f} ms"
            )

            print(
                "-" * 50
            )


    # --------------------------------------------------------
    # SAVE RESULT ROW
    # --------------------------------------------------------

    row = {

        "image":
            image_path.name,

        "gt_instances":
            len(gt_labels),

        "router_model":
            f"YOLOv13-{predicted_model}",

        "oracle_model":
            f"YOLOv13-{oracle_model}",

        "oracle_match":
            int(
                predicted_model
                == oracle_model
            ),

        "adaptive_precision":
            adaptive_precision,

        "adaptive_recall":
            adaptive_recall,

        "adaptive_f1":
            adaptive_f1,

        "adaptive_latency_ms":
            adaptive_latency
            * 1000.0,

        "oracle_precision":
            oracle_precision,

        "oracle_recall":
            oracle_recall,

        "oracle_f1":
            oracle_f1,

    }


    for model_name in MODEL_ORDER:

        prefix = (
            model_name
        )

        row[
            f"{prefix}_precision"
        ] = model_results[
            model_name
        ]["precision"]

        row[
            f"{prefix}_recall"
        ] = model_results[
            model_name
        ]["recall"]

        row[
            f"{prefix}_f1"
        ] = model_results[
            model_name
        ]["f1"]

        row[
            f"{prefix}_latency_ms"
        ] = (
            model_results[
                model_name
            ]["latency"]
            * 1000.0
        )


    result_rows.append(
        row
    )


    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if (
        image_idx % 25 == 0
        or image_idx == total
    ):

        print(
            f"Processed "
            f"{image_idx}/"
            f"{total} "
            f"("
            f"{100.0 * image_idx / total:.1f}%"
            f") "
            f"| router=YOLOv13-"
            f"{predicted_model}"
        )


# ============================================================
# SAVE CSV
# ============================================================

if result_rows:

    fieldnames = list(
        result_rows[0].keys()
    )

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

        writer.writerows(
            result_rows
        )


# ============================================================
# SUMMARY
# ============================================================

print()

print("=" * 70)

print(
    "C3 — ADAPTIVE ROUTING SUMMARY"
)

print("=" * 70)


# ============================================================
# FIXED MODEL PERFORMANCE
# ============================================================

print()

print(
    "FIXED MODEL PERFORMANCE"
)

print(
    "-" * 70
)


fixed_summary = {}


for model_name in MODEL_ORDER:

    precision = np.mean(
        fixed_metrics[
            model_name
        ]["precision"]
    )

    recall = np.mean(
        fixed_metrics[
            model_name
        ]["recall"]
    )

    f1 = np.mean(
        fixed_metrics[
            model_name
        ]["f1"]
    )

    latency = np.mean(
        fixed_metrics[
            model_name
        ]["latency"]
    )

    latency_ms = (
        latency
        * 1000.0
    )

    fps = (

        1000.0
        / latency_ms

        if latency_ms > 0

        else 0.0
    )


    fixed_summary[
        model_name
    ] = {

        "precision":
            float(precision),

        "recall":
            float(recall),

        "f1":
            float(f1),

        "latency_ms":
            float(latency_ms),

        "fps":
            float(fps),
    }


    print()

    print(
        f"YOLOv13-{model_name}"
    )

    print(
        f"  Precision : "
        f"{precision:.4f}"
    )

    print(
        f"  Recall    : "
        f"{recall:.4f}"
    )

    print(
        f"  F1        : "
        f"{f1:.4f}"
    )

    print(
        f"  Latency   : "
        f"{latency_ms:.2f} ms"
    )

    print(
        f"  FPS       : "
        f"{fps:.2f}"
    )


# ============================================================
# ADAPTIVE
# ============================================================

adaptive_precision = np.mean(
    adaptive_metrics[
        "precision"
    ]
)

adaptive_recall = np.mean(
    adaptive_metrics[
        "recall"
    ]
)

adaptive_f1 = np.mean(
    adaptive_metrics[
        "f1"
    ]
)

adaptive_latency = np.mean(
    adaptive_metrics[
        "latency"
    ]
)

adaptive_latency_ms = (
    adaptive_latency
    * 1000.0
)

adaptive_fps = (

    1000.0
    / adaptive_latency_ms

    if adaptive_latency_ms > 0

    else 0.0
)


print()

print(
    "LEARNED ADAPTIVE ROUTER"
)

print(
    f"Mean Precision : "
    f"{adaptive_precision:.4f}"
)

print(
    f"Mean Recall    : "
    f"{adaptive_recall:.4f}"
)

print(
    f"Mean F1        : "
    f"{adaptive_f1:.4f}"
)

print(
    f"Mean Latency   : "
    f"{adaptive_latency_ms:.2f} ms"
)

print(
    f"Mean FPS       : "
    f"{adaptive_fps:.2f}"
)


# ============================================================
# ORACLE
# ============================================================

oracle_precision = np.mean(
    oracle_metrics[
        "precision"
    ]
)

oracle_recall = np.mean(
    oracle_metrics[
        "recall"
    ]
)

oracle_f1 = np.mean(
    oracle_metrics[
        "f1"
    ]
)


print()

print(
    "ORACLE UPPER BOUND"
)

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


# ============================================================
# ROUTING DISTRIBUTION
# ============================================================

print()

print(
    "ADAPTIVE ROUTING DISTRIBUTION"
)

print(
    "-" * 40
)


for model_name in MODEL_ORDER:

    count = (
        routing_distribution[
            model_name
        ]
    )

    percentage = (

        100.0
        * count
        / total

        if total > 0

        else 0.0
    )

    print(
        f"YOLOv13-{model_name}: "
        f"{count:4d} "
        f"({percentage:5.2f}%)"
    )


# ============================================================
# ROUTER ACCURACY
# ============================================================

oracle_routing_accuracy = (

    np.mean(
        [
            int(
                row[
                    "oracle_match"
                ]
            )

            for row
            in result_rows
        ]
    )

    if result_rows

    else 0.0
)


# ============================================================
# ORACLE GAP
# ============================================================

oracle_gap = (
    oracle_f1
    -
    adaptive_f1
)


# ============================================================
# BEST FIXED MODEL
# ============================================================

fixed_f1s = {

    model_name:
        fixed_summary[
            model_name
        ]["f1"]

    for model_name
    in MODEL_ORDER
}


best_fixed_model = max(
    fixed_f1s,
    key=fixed_f1s.get
)


best_fixed_f1 = (
    fixed_f1s[
        best_fixed_model
    ]
)


best_fixed_latency = (
    fixed_summary[
        best_fixed_model
    ]["latency_ms"]
)


adaptive_vs_fixed = (
    adaptive_f1
    -
    best_fixed_f1
)


latency_change = (

    (
        adaptive_latency_ms
        -
        best_fixed_latency
    )
    /
    best_fixed_latency
    * 100.0

    if best_fixed_latency > 0

    else 0.0
)


# ============================================================
# ROUTER DIAGNOSTICS
# ============================================================

print()

print(
    "ROUTER DIAGNOSTICS"
)

print(
    f"Oracle-model routing accuracy : "
    f"{oracle_routing_accuracy:.4f}"
)

print(
    f"Oracle-model routing accuracy : "
    f"{oracle_routing_accuracy * 100.0:.2f}%"
)

print(
    f"Oracle → Adaptive F1 gap       : "
    f"{oracle_gap:.4f}"
)

print()

print(
    f"Best fixed model               : "
    f"YOLOv13-{best_fixed_model}"
)

print(
    f"Best fixed F1                  : "
    f"{best_fixed_f1:.4f}"
)

print(
    f"Adaptive vs best-fixed F1      : "
    f"{adaptive_vs_fixed:+.4f}"
)

print()

print(
    f"Best-fixed latency             : "
    f"{best_fixed_latency:.2f} ms"
)

print(
    f"Adaptive latency change        : "
    f"{latency_change:+.2f}%"
)


# ============================================================
# SAVE JSON
# ============================================================

metrics_output = {

    "configuration": {

        "image_size":
            IMG_SIZE,

        "confidence":
            CONF,

        "iou_threshold":
            IOU_THRESHOLD,

        "device":
            DEVICE,

        "validation_images":
            total,
    },

    "fixed_models":
        fixed_summary,

    "adaptive_router": {

        "precision":
            float(
                adaptive_precision
            ),

        "recall":
            float(
                adaptive_recall
            ),

        "f1":
            float(
                adaptive_f1
            ),

        "latency_ms":
            float(
                adaptive_latency_ms
            ),

        "fps":
            float(
                adaptive_fps
            ),
    },

    "oracle": {

        "precision":
            float(
                oracle_precision
            ),

        "recall":
            float(
                oracle_recall
            ),

        "f1":
            float(
                oracle_f1
            ),
    },

    "routing_distribution":
        routing_distribution,

    "router_diagnostics": {

        "oracle_model_routing_accuracy":
            float(
                oracle_routing_accuracy
            ),

        "oracle_model_routing_accuracy_percent":
            float(
                oracle_routing_accuracy
                * 100.0
            ),

        "oracle_adaptive_f1_gap":
            float(
                oracle_gap
            ),

        "best_fixed_model":
            f"YOLOv13-{best_fixed_model}",

        "best_fixed_f1":
            float(
                best_fixed_f1
            ),

        "adaptive_vs_best_fixed_f1":
            float(
                adaptive_vs_fixed
            ),

        "best_fixed_latency_ms":
            float(
                best_fixed_latency
            ),

        "adaptive_latency_change_percent":
            float(
                latency_change
            ),
    }
}


with open(
    OUTPUT_JSON,
    "w"
) as f:

    json.dump(
        metrics_output,
        f,
        indent=4
    )


# ============================================================
# FINAL
# ============================================================

print()

print("=" * 70)

print(
    "C3 RESULTS SAVED"
)

print("=" * 70)

print()

print(
    "CSV:"
)

print(
    OUTPUT_CSV
)

print()

print(
    "Metrics:"
)

print(
    OUTPUT_JSON
)

print()

print(
    "C3 COMPLETE"
)

print("=" * 70)
