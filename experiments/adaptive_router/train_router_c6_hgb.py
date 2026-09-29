from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
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
    "adaptive_router/router_model_c6_hgb"
)

RANDOM_STATE = 42
TEST_SIZE = 0.20

# ============================================================
# FEATURES
# ============================================================

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

LABELS = [
    "YOLOv13-N",
    "YOLOv13-S",
    "YOLOv13-L",
    "YOLOv13-X",
]

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
print("C6-A — HISTGRADIENTBOOSTING ADAPTIVE ROUTER")
print("=" * 70)

print()
print("Loading CSV:")
print(CSV_PATH)
print()

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Training CSV not found:\n{CSV_PATH}"
    )

df = pd.read_csv(CSV_PATH)

print(f"Total rows : {len(df)}")

# ============================================================
# VALIDATION
# ============================================================

missing_features = [
    c for c in FEATURE_COLUMNS
    if c not in df.columns
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

counts = df[TARGET_COLUMN].value_counts()

for label in LABELS:

    count = int(counts.get(label, 0))

    percentage = (
        100.0 * count / len(df)
        if len(df) > 0
        else 0.0
    )

    print(
        f"{label}: "
        f"{count:5d} "
        f"({percentage:6.2f}%)"
    )

# ============================================================
# PREPARE DATA
# ============================================================

X = df[FEATURE_COLUMNS].astype(np.float32)
y = df[TARGET_COLUMN].astype(str)

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

print(f"Training samples   : {len(X_train)}")
print(f"Validation samples : {len(X_val)}")

# ============================================================
# HISTGRADIENTBOOSTING ROUTER
# ============================================================

print()
print("=" * 70)
print("TRAINING HISTGRADIENTBOOSTING ROUTER")
print("=" * 70)

router = HistGradientBoostingClassifier(
    learning_rate=0.05,
    max_iter=300,
    max_leaf_nodes=31,
    min_samples_leaf=15,
    l2_regularization=1.0,
    max_features=1.0,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=25,
    tol=1e-5,
    random_state=RANDOM_STATE,
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
print("C6-A ROUTER PERFORMANCE")
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
        labels=LABELS,
        zero_division=0,
    )
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_val,
    y_pred,
    labels=LABELS
)

print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print()
print("Actual \\ Predicted")
print(
    "             "
    + " ".join(
        f"{x[-1]:>6}"
        for x in LABELS
    )
)

for label, row in zip(LABELS, cm):

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

if hasattr(router, "feature_importances_"):

    importance = router.feature_importances_

    importance_df = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": importance,
    })

    importance_df = importance_df.sort_values(
        "importance",
        ascending=False
    )

    for _, row in importance_df.iterrows():

        print(
            f"{row['feature']:30s} "
            f"{row['importance']:.6f}"
        )

else:

    print(
        "HistGradientBoostingClassifier "
        "does not expose feature_importances_."
    )

# ============================================================
# SAVE MODEL
# ============================================================

import joblib

model_path = (
    OUTPUT_DIR
    / "hist_gradient_boosting_router.joblib"
)

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

feature_path = (
    OUTPUT_DIR
    / "router_features.json"
)

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
    "classifier": "HistGradientBoostingClassifier",
    "accuracy": float(accuracy),
    "balanced_accuracy": float(
        balanced_accuracy
    ),
    "macro_f1": float(
        macro_f1
    ),
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
    "test_size": TEST_SIZE,
    "learning_rate": 0.05,
    "max_iter": 300,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 15,
    "l2_regularization": 1.0,
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
print("C6-A HISTGRADIENTBOOSTING TRAINING COMPLETE")
print("=" * 70)
