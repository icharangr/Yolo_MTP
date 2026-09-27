from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ============================================================
# CONFIGURATION
# ============================================================

CSV_PATH = Path(
    "/home/charan/thesis/yolov13/experiments/"
    "adaptive_router/router_training_features.csv"
)

OUTPUT_DIR = Path(
    "/home/charan/thesis/yolov13/experiments/"
    "adaptive_router/router_model"
)

RANDOM_STATE = 42

TEST_SIZE = 0.20

# ============================================================
# FEATURES
# ============================================================

# IMPORTANT:
# These are the ONLY features available to the router
# during real inference.
#
# DO NOT include:
#   oracle_model
#   gt_instances
#   N/S/L/X precision
#   N/S/L/X recall
#   N/S/L/X F1

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

TARGET_COLUMN = "oracle_model"

# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("C2 — LEARNED ADAPTIVE ROUTER TRAINING")
print("=" * 70)

print()
print(f"Loading CSV:")
print(CSV_PATH)
print()

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Training CSV not found:\n{CSV_PATH}"
    )

df = pd.read_csv(CSV_PATH)

print(f"Total rows : {len(df)}")
print()

# ============================================================
# VALIDATE DATA
# ============================================================

missing_features = [
    col
    for col in FEATURE_COLUMNS
    if col not in df.columns
]

if missing_features:
    raise ValueError(
        "Missing feature columns:\n"
        + "\n".join(missing_features)
    )

if TARGET_COLUMN not in df.columns:
    raise ValueError(
        f"Missing target column: {TARGET_COLUMN}"
    )

# Remove invalid rows
df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

before = len(df)

df = df.dropna(
    subset=FEATURE_COLUMNS + [TARGET_COLUMN]
).reset_index(drop=True)

print(
    f"Rows removed because of NaN/Inf: "
    f"{before - len(df)}"
)

print(
    f"Rows available for training: "
    f"{len(df)}"
)

# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)

target_counts = df[TARGET_COLUMN].value_counts()

for model in [
    "YOLOv13-N",
    "YOLOv13-S",
    "YOLOv13-L",
    "YOLOv13-X",
]:

    count = int(
        target_counts.get(model, 0)
    )

    percentage = (
        100.0 * count / len(df)
        if len(df) > 0
        else 0.0
    )

    print(
        f"{model}: "
        f"{count:5d} "
        f"({percentage:6.2f}%)"
    )

# ============================================================
# PREPARE X / Y
# ============================================================

X = df[FEATURE_COLUMNS].astype(
    np.float32
)

y = df[TARGET_COLUMN].astype(str)

# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print()
print("=" * 70)
print("DATA SPLIT")
print("=" * 70)

print(
    f"Training samples   : {len(X_train)}"
)

print(
    f"Validation samples : {len(X_val)}"
)

print()

# ============================================================
# RANDOM FOREST ROUTER
# ============================================================

print("=" * 70)
print("TRAINING RANDOM FOREST ROUTER")
print("=" * 70)

router = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=3,
    max_features="sqrt",
    class_weight="balanced_subsample",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

router.fit(
    X_train,
    y_train
)

print()
print("Training complete.")

# ============================================================
# PREDICTION
# ============================================================

y_pred = router.predict(X_val)

# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_val,
    y_pred
)

balanced_accuracy = balanced_accuracy_score(
    y_val,
    y_pred
)

macro_f1 = f1_score(
    y_val,
    y_pred,
    average="macro"
)

weighted_f1 = f1_score(
    y_val,
    y_pred,
    average="weighted"
)

print()
print("=" * 70)
print("ROUTER PERFORMANCE")
print("=" * 70)

print(
    f"Accuracy           : {accuracy:.4f}"
)

print(
    f"Balanced Accuracy  : {balanced_accuracy:.4f}"
)

print(
    f"Macro F1           : {macro_f1:.4f}"
)

print(
    f"Weighted F1        : {weighted_f1:.4f}"
)

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_val,
        y_pred,
        labels=[
            "YOLOv13-N",
            "YOLOv13-S",
            "YOLOv13-L",
            "YOLOv13-X",
        ],
        zero_division=0,
    )
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

labels = [
    "YOLOv13-N",
    "YOLOv13-S",
    "YOLOv13-L",
    "YOLOv13-X",
]

cm = confusion_matrix(
    y_val,
    y_pred,
    labels=labels
)

print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print()

print(
    "Actual \\ Predicted"
)

print(
    "             "
    + " ".join(
        f"{x[-1]:>6}"
        for x in labels
    )
)

for label, row in zip(labels, cm):

    print(
        f"{label:12s}"
        + " ".join(
            f"{value:6d}"
            for value in row
        )
    )

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print()
print("=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

importance = router.feature_importances_

importance_df = pd.DataFrame({
    "feature": FEATURE_COLUMNS,
    "importance": importance,
})

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)

print()

for _, row in importance_df.iterrows():

    print(
        f"{row['feature']:30s} "
        f"{row['importance']:.6f}"
    )

# ============================================================
# SAVE MODEL
# ============================================================

import joblib

model_path = OUTPUT_DIR / "random_forest_router.joblib"

joblib.dump(
    router,
    model_path
)

print()
print(
    f"Router saved to:\n"
    f"{model_path}"
)

# ============================================================
# SAVE FEATURE LIST
# ============================================================

feature_path = OUTPUT_DIR / "router_features.json"

with open(
    feature_path,
    "w"
) as f:

    json.dump(
        FEATURE_COLUMNS,
        f,
        indent=4
    )

print(
    f"Feature list saved to:\n"
    f"{feature_path}"
)

# ============================================================
# SAVE VALIDATION PREDICTIONS
# ============================================================

validation_output = X_val.copy()

validation_output[
    "oracle_model"
] = y_val.values

validation_output[
    "predicted_model"
] = y_pred

validation_output[
    "correct"
] = (
    validation_output["oracle_model"]
    ==
    validation_output["predicted_model"]
)

prediction_path = (
    OUTPUT_DIR
    / "router_validation_predictions.csv"
)

validation_output.to_csv(
    prediction_path,
    index=False
)

print(
    f"Validation predictions saved to:\n"
    f"{prediction_path}"
)

# ============================================================
# SAVE METRICS
# ============================================================

metrics = {
    "accuracy": float(accuracy),
    "balanced_accuracy": float(
        balanced_accuracy
    ),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(
        weighted_f1
    ),
    "training_samples": int(
        len(X_train)
    ),
    "validation_samples": int(
        len(X_val)
    ),
    "random_state": RANDOM_STATE,
    "n_estimators": 500,
}

metrics_path = (
    OUTPUT_DIR
    / "router_metrics.json"
)

with open(
    metrics_path,
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )

print(
    f"Metrics saved to:\n"
    f"{metrics_path}"
)

# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("C2 ROUTER TRAINING COMPLETE")
print("=" * 70)
