
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

IMAGE_DIR = Path(
    "/home/charan/thesis/datasets/VisDrone/"
    "VisDrone2019-DET-val/images"
)

LABEL_DIR = Path(
    "/home/charan/thesis/datasets/VisDrone/"
    "VisDrone2019-DET-val/annotations"
)


# ============================================================
# C3 ROUTER
# ============================================================

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


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "adaptive_router"
    / "c4_cost_aware"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "c4_cost_aware_results.csv"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "c4_cost_aware_metrics.json"
)


# ============================================================
# MODELS
# ============================================================

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


# ============================================================
# INFERENCE
# ============================================================

IMG_SIZE = 640

CONF = 0.25

IOU_THRESHOLD = 0.50

DEVICE = "cuda:0"


# ============================================================
# C4 COST PENALTIES
# ============================================================
#
# lambda = 0
#     pure learned router
#
# larger lambda
#     stronger latency penalty
#
# The sweep is reported so that C4 does not depend on
# one arbitrary manually selected value.
#
# ============================================================

COST_LAMBDAS = [
    0.00,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    1.00,
]


# ============================================================
# DATASET
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
            f"Image directory not found:\n{IMAGE_DIR}"
        )

    files = []

    for ext in IMAGE_EXTENSIONS:

        files.extend(
            IMAGE_DIR.glob(
                f"*{ext}"
            )
        )

    files = sorted(
        set(files)
    )

    if len(files) == 0:

        raise RuntimeError(
            f"No validation images found in:\n{IMAGE_DIR}"
        )

    return files


image_files = find_image_files()


# ============================================================
# LABEL LOADING
# ============================================================

def load_labels(label_file):

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

                category = int(parts[5])

            except ValueError:

                continue

            # VisDrone category 0 is ignored.
            if category == 0:
                continue

            # Valid detection categories are 1-10.
            if category < 1 or category > 10:
                continue

            cls = category - 1

            labels.append(
                {
                    "cls": cls,
                    "box": np.array(
                        [
                            x,
                            y,
                            x + w,
                            y + h,
                        ],
                        dtype=np.float32
                    )
                }
            )

    return labels


# ============================================================
# IoU
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
# METRICS
# ============================================================

def calculate_metrics(
    gt_labels,
    predictions
):

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

        for idx, (
            gt_cls,
            gt_box
        ) in enumerate(gt):

            if idx in matched_gt:
                continue

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
                best_gt = idx

        if best_gt is not None:

            tp += 1

            matched_gt.add(
                best_gt
            )

        else:

            fp += 1

    fn = (
        len(gt)
        - tp
    )

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
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
# ROUTER FEATURES
# ============================================================

def extract_router_features(
    predictions,
    image_width,
    image_height
):

    num_predictions = len(
        predictions
    )

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

    image_area = (
        float(image_width)
        * float(image_height)
    )

    if image_area <= 0:

        image_area = 1.0

    confidences = []
    area_ratios = []
    width_ratios = []
    height_ratios = []

    small_count = 0
    medium_count = 0
    large_count = 0

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
            area
            / image_area
        )

        width_ratio = (
            width
            / float(image_width)
            if image_width > 0
            else 0.0
        )

        height_ratio = (
            height
            / float(image_height)
            if image_height > 0
            else 0.0
        )

        confidences.append(conf)
        area_ratios.append(area_ratio)
        width_ratios.append(width_ratio)
        height_ratios.append(height_ratio)

        if area_ratio < 0.01:

            small_count += 1

        elif area_ratio < 0.05:

            medium_count += 1

        else:

            large_count += 1

    density = (
        num_predictions
        / image_area
    )

    return {

        "probe_num_predictions":
            float(num_predictions),

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
                np.mean(width_ratios)
            ),

        "probe_mean_height_ratio":
            float(
                np.mean(height_ratios)
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

            predictions.append(
                {
                    "cls": int(cls),
                    "box": np.asarray(
                        box,
                        dtype=np.float32
                    ),
                    "conf": float(conf),
                }
            )

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
# LOAD ROUTER
# ============================================================

print("=" * 70)
print("C4 — COST-AWARE ADAPTIVE ROUTER")
print("=" * 70)
print()

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


if not ROUTER_PATH.exists():

    raise FileNotFoundError(
        f"Router not found:\n{ROUTER_PATH}"
    )


router = joblib.load(
    ROUTER_PATH
)

print(
    f"Router loaded: {ROUTER_PATH}"
)


if FEATURE_PATH.exists():

    with open(
        FEATURE_PATH,
        "r"
    ) as f:

        FEATURE_COLUMNS = json.load(f)

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


# ============================================================
# LOAD MODELS
# ============================================================

print()
print("Loading models...")

loaded_models = {}

for model_name in MODEL_ORDER:

    print(
        f"  Loading YOLOv13-{model_name}"
    )

    loaded_models[model_name] = YOLO(
        str(
            MODEL_PATHS[
                model_name
            ]
        )
    )


print()
print("All models loaded.")
print()


# ============================================================
# CALIBRATION / COST MEASUREMENT
# ============================================================
#
# We measure the actual fixed-model latency using the first
# validation image. C3 already established the approximate
# latencies, but C4 uses measurements from this process.
#
# ============================================================

print(
    "Measuring model inference cost..."
)

warmup_image = image_files[0]

model_latency = {}

for model_name in MODEL_ORDER:

    # Warm-up
    _ = run_model(
        loaded_models[model_name],
        warmup_image
    )

    measurements = []

    for _ in range(3):

        _, elapsed = timed_run_model(
            loaded_models[model_name],
            warmup_image
        )

        measurements.append(
            elapsed
        )

    model_latency[model_name] = float(
        np.median(
            measurements
        )
    )


print()

for model_name in MODEL_ORDER:

    print(
        f"YOLOv13-{model_name}: "
        f"{model_latency[model_name] * 1000:.2f} ms"
    )


# ============================================================
# NORMALIZED COST
# ============================================================

latencies = np.array(
    [
        model_latency[m]
        for m in MODEL_ORDER
    ],
    dtype=np.float64
)

min_latency = float(
    np.min(latencies)
)

max_latency = float(
    np.max(latencies)
)

if max_latency > min_latency:

    normalized_cost = {

        model_name:
            (
                model_latency[model_name]
                - min_latency
            )
            /
            (
                max_latency
                - min_latency
            )

        for model_name
        in MODEL_ORDER
    }

else:

    normalized_cost = {
        model_name: 0.0
        for model_name
        in MODEL_ORDER
    }


print()
print("Normalized model costs:")

for model_name in MODEL_ORDER:

    print(
        f"  {model_name}: "
        f"{normalized_cost[model_name]:.4f}"
    )


# ============================================================
# STORAGE
# ============================================================

records = []

all_model_metrics = {
    model_name: {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "precision": [],
        "recall": [],
        "f1": [],
        "latency": [],
    }
    for model_name in MODEL_ORDER
}


# ============================================================
# RUN DATASET
# ============================================================

print()
print("=" * 70)
print("RUNNING C4 EVALUATION")
print("=" * 70)
print()


for image_index, image_path in enumerate(
    image_files,
    start=1
):

    if image_index % 25 == 0:

        print(
            f"[{image_index}/{len(image_files)}]"
        )


    # --------------------------------------------------------
    # IMAGE SIZE
    # --------------------------------------------------------

    try:

        from PIL import Image

        with Image.open(
            image_path
        ) as img:

            image_width, image_height = img.size

    except Exception:

        image_width = IMG_SIZE
        image_height = IMG_SIZE


    # --------------------------------------------------------
    # GT
    # --------------------------------------------------------

    label_path = (
        LABEL_DIR
        / f"{image_path.stem}.txt"
    )

    gt_labels = load_labels(
        label_path
    )


    # --------------------------------------------------------
    # N PROBE
    # --------------------------------------------------------

    n_predictions, n_latency = (
        timed_run_model(
            loaded_models["N"],
            image_path
        )
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
            features[col]
            for col
            in FEATURE_COLUMNS
        ]],
        columns=FEATURE_COLUMNS
    )


    # --------------------------------------------------------
    # ROUTER PROBABILITIES
    # --------------------------------------------------------

    probabilities = router.predict_proba(
        feature_vector
    )[0]

    classes = list(
        router.classes_
    )

    probability_map = {}

    for cls, probability in zip(
        classes,
        probabilities
    ):

        model_name = str(
            cls
        ).replace(
            "YOLOv13-",
            ""
        )

        probability_map[
            model_name
        ] = float(
            probability
        )


    # --------------------------------------------------------
    # ENSURE ALL MODELS PRESENT
    # --------------------------------------------------------

    for model_name in MODEL_ORDER:

        if model_name not in probability_map:

            probability_map[
                model_name
            ] = 0.0


    # --------------------------------------------------------
    # RUN ALL MODELS
    #
    # This gives us the ground-truth per-image utility
    # required for an honest C4 evaluation.
    # --------------------------------------------------------

    per_model = {}

    for model_name in MODEL_ORDER:

        if model_name == "N":

            predictions = n_predictions
            latency = n_latency

        else:

            predictions, latency = (
                timed_run_model(
                    loaded_models[
                        model_name
                    ],
                    image_path
                )
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


        per_model[
            model_name
        ] = {

            "predictions": predictions,

            "latency": latency,

            "tp": tp,

            "fp": fp,

            "fn": fn,

            "precision": precision,

            "recall": recall,

            "f1": f1,
        }


    # --------------------------------------------------------
    # COST-AWARE ROUTING
    #
    # utility(model) =
    #
    #     router_probability(model)
    #     *
    #     (1 - lambda * normalized_cost(model))
    #
    # This preserves the learned router preference while
    # explicitly penalizing expensive models.
    # --------------------------------------------------------

    selected_models = {}

    for cost_lambda in COST_LAMBDAS:

        scores = {}

        for model_name in MODEL_ORDER:

            probability = (
                probability_map[
                    model_name
                ]
            )

            cost = (
                normalized_cost[
                    model_name
                ]
            )

            score = (
                probability
                *
                (
                    1.0
                    -
                    cost_lambda * cost
                )
            )

            scores[
                model_name
            ] = score


        selected_model = max(
            scores,
            key=scores.get
        )

        selected_models[
            cost_lambda
        ] = {

            "model":
                selected_model,

            "score":
                scores[
                    selected_model
                ],
        }


    # --------------------------------------------------------
    # RECORD
    # --------------------------------------------------------

    record = {

        "image":
            image_path.name,

        "image_index":
            image_index,

        "num_gt":
            len(gt_labels),

    }


    for model_name in MODEL_ORDER:

        record[
            f"prob_{model_name}"
        ] = probability_map[
            model_name
        ]

        record[
            f"f1_{model_name}"
        ] = per_model[
            model_name
        ]["f1"]

        record[
            f"precision_{model_name}"
        ] = per_model[
            model_name
        ]["precision"]

        record[
            f"recall_{model_name}"
        ] = per_model[
            model_name
        ]["recall"]

        record[
            f"latency_{model_name}"
        ] = (
            per_model[
                model_name
            ]["latency"]
            * 1000.0
        )


    for cost_lambda in COST_LAMBDAS:

        key = (
            f"{cost_lambda:.2f}"
            .replace(".", "p")
        )

        record[
            f"selected_{key}"
        ] = selected_models[
            cost_lambda
        ]["model"]


    records.append(
        record
    )


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(
    records
)


# ============================================================
# C4 SUMMARY FOR EACH COST
# ============================================================

lambda_results = {}

for cost_lambda in COST_LAMBDAS:

    key = (
        f"{cost_lambda:.2f}"
        .replace(".", "p")
    )

    selected_column = (
        f"selected_{key}"
    )

    precisions = []
    recalls = []
    f1s = []
    latencies = []

    routing_distribution = {
        model_name: 0
        for model_name in MODEL_ORDER
    }

    for image_path in image_files:

        row = df[
            df["image"]
            == image_path.name
        ]

        if len(row) == 0:
            continue

        row = row.iloc[0]

        selected_model = row[
            selected_column
        ]

        routing_distribution[
            selected_model
        ] += 1

        precisions.append(
            row[
                f"precision_{selected_model}"
            ]
        )

        recalls.append(
            row[
                f"recall_{selected_model}"
            ]
        )

        f1s.append(
            row[
                f"f1_{selected_model}"
            ]
        )

                 # ----------------------------------------------------
        # DEPLOYED C4 LATENCY
        #
        # Every routed image first runs the N probe.
        #
        # If N is selected:
        #     latency = N
        #
        # If S/L/X is selected:
        #     latency = N probe + selected model
        # ----------------------------------------------------

        n_probe_latency = row[
            "latency_N"
        ]

        if selected_model == "N":

            deployed_latency = (
                n_probe_latency
            )

        else:

            deployed_latency = (
                n_probe_latency
                +
                row[
                    f"latency_{selected_model}"
                ]
            )

        latencies.append(
            deployed_latency
        )


    mean_precision = float(
        np.mean(
            precisions
        )
    )

    mean_recall = float(
        np.mean(
            recalls
        )
    )

    mean_f1 = float(
        np.mean(
            f1s
        )
    )

    mean_latency = float(
        np.mean(
            latencies
        )
    )

    fps = (
        1000.0 / mean_latency
        if mean_latency > 0
        else 0.0
    )

    lambda_results[
        str(cost_lambda)
    ] = {

        "lambda":
            cost_lambda,

        "precision":
            mean_precision,

        "recall":
            mean_recall,

        "f1":
            mean_f1,

        "latency_ms":
            mean_latency,

        "fps":
            fps,

        "routing_distribution":
            routing_distribution,
    }


# ============================================================
# FIXED MODEL BASELINES
# ============================================================

fixed_results = {}

for model_name in MODEL_ORDER:

    precision_values = df[
        f"precision_{model_name}"
    ].to_numpy()

    recall_values = df[
        f"recall_{model_name}"
    ].to_numpy()

    f1_values = df[
        f"f1_{model_name}"
    ].to_numpy()

    latency_values = df[
        f"latency_{model_name}"
    ].to_numpy()


    mean_precision = float(
        np.mean(
            precision_values
        )
    )

    mean_recall = float(
        np.mean(
            recall_values
        )
    )

    mean_f1 = float(
        np.mean(
            f1_values
        )
    )

    mean_latency = float(
        np.mean(
            latency_values
        )
    )

    fps = (
        1000.0 / mean_latency
        if mean_latency > 0
        else 0.0
    )

    fixed_results[
        model_name
    ] = {

        "precision":
            mean_precision,

        "recall":
            mean_recall,

        "f1":
            mean_f1,

        "latency_ms":
            mean_latency,

        "fps":
            fps,
    }


# ============================================================
# SAVE CSV
# ============================================================

df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# SAVE JSON
# ============================================================

metrics = {

    "experiment":
        "C4_cost_aware_adaptive_router",

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
            len(image_files),

        "cost_lambdas":
            COST_LAMBDAS,
    },

    "model_latency_calibration": {
        model_name:
            model_latency[
                model_name
            ] * 1000.0
        for model_name
        in MODEL_ORDER
    },

    "normalized_cost": {
        model_name:
            normalized_cost[
                model_name
        ]
        for model_name
        in MODEL_ORDER
    },

    "fixed_models":
        fixed_results,

    "cost_aware_results":
        lambda_results,
}


with open(
    OUTPUT_JSON,
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=2
    )


# ============================================================
# PRINT SUMMARY
# ============================================================

print()
print("=" * 70)
print("C4 — COST-AWARE ADAPTIVE ROUTING SUMMARY")
print("=" * 70)

print()
print("FIXED MODEL BASELINES")
print("-" * 70)

for model_name in MODEL_ORDER:

    result = fixed_results[
        model_name
    ]

    print(
        f"YOLOv13-{model_name}"
    )

    print(
        f"  Precision : "
        f"{result['precision']:.4f}"
    )

    print(
        f"  Recall    : "
        f"{result['recall']:.4f}"
    )

    print(
        f"  F1        : "
        f"{result['f1']:.4f}"
    )

    print(
        f"  Latency   : "
        f"{result['latency_ms']:.2f} ms"
    )

    print(
        f"  FPS       : "
        f"{result['fps']:.2f}"
    )

    print()


print("=" * 70)
print("C4 COST SWEEP")
print("=" * 70)

print()

for cost_lambda in COST_LAMBDAS:

    result = lambda_results[
        str(cost_lambda)
    ]

    print(
        f"lambda={cost_lambda:.2f}"
    )

    print(
        f"  Precision : "
        f"{result['precision']:.4f}"
    )

    print(
        f"  Recall    : "
        f"{result['recall']:.4f}"
    )

    print(
        f"  F1        : "
        f"{result['f1']:.4f}"
    )

    print(
        f"  Latency   : "
        f"{result['latency_ms']:.2f} ms"
    )

    print(
        f"  FPS       : "
        f"{result['fps']:.2f}"
    )

    print(
        "  Routing   : "
        + ", ".join(
            [
                f"{m}={result['routing_distribution'][m]}"
                for m in MODEL_ORDER
            ]
        )
    )

    print()


# ============================================================
# REFERENCE C3
# ============================================================

C3_JSON = (
    PROJECT_ROOT
    / "experiments"
    / "adaptive_router"
    / "c4_results"
    / "adaptive_router_metrics.json"
)


if C3_JSON.exists():

    with open(
        C3_JSON,
        "r"
    ) as f:

        c3 = json.load(f)

    print("=" * 70)
    print("C3 REFERENCE")
    print("=" * 70)

    print()

    c3_adaptive = c3[
        "adaptive_router"
    ]

    print(
        f"C3 Adaptive F1      : "
        f"{c3_adaptive['f1']:.4f}"
    )

    print(
        f"C3 Adaptive Latency : "
        f"{c3_adaptive['latency_ms']:.2f} ms"
    )

    print()


# ============================================================
# FILES
# ============================================================

print("=" * 70)
print("C4 RESULTS SAVED")
print("=" * 70)

print()
print("CSV:")
print(OUTPUT_CSV)

print()
print("Metrics:")
print(OUTPUT_JSON)

print()
print("C4 COMPLETE")
print("=" * 70)

