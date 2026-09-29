from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
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
    "adaptive_router/router_model_c6_pairwise"
)

RANDOM_STATE = 42
TEST_SIZE = 0.20

MODEL_ORDER = [
    "N",
    "S",
    "L",
    "X",
]

MODEL_NAMES = {
    "N": "YOLOv13-N",
    "S": "YOLOv13-S",
    "L": "YOLOv13-L",
    "X": "YOLOv13-X",
}

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

F1_COLUMNS = {
    model: f"{model}_f1"
    for model in MODEL_ORDER
}

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("C6-C — PAIRWISE F1 UTILITY-DIFFERENCE ROUTER")
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
# VALIDATE DATA
# ============================================================

required_columns = (
    FEATURE_COLUMNS
    + list(F1_COLUMNS.values())
)

missing = [
    c for c in required_columns
    if c not in df.columns
]

if missing:
    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing)
    )

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

before = len(df)

df = df.dropna(
    subset=required_columns
).reset_index(drop=True)

print(
    f"Rows removed because of NaN/Inf: "
    f"{before - len(df)}"
)

print(
    f"Rows available: {len(df)}"
)

# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

indices = np.arange(len(df))

train_idx, val_idx = train_test_split(
    indices,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df["oracle_model"],
)

train_df = df.iloc[train_idx].copy()
val_df = df.iloc[val_idx].copy()

print()
print("=" * 70)
print("DATA SPLIT")
print("=" * 70)

print(
    f"Training samples   : {len(train_df)}"
)

print(
    f"Validation samples : {len(val_df)}"
)

# ============================================================
# FEATURES
# ============================================================

X_train = train_df[
    FEATURE_COLUMNS
].astype(np.float32)

X_val = val_df[
    FEATURE_COLUMNS
].astype(np.float32)

# ============================================================
# PAIRWISE TARGETS
# ============================================================

pairs = []

for i, model_a in enumerate(MODEL_ORDER):

    for model_b in MODEL_ORDER[i + 1:]:

        pairs.append(
            (model_a, model_b)
        )

print()
print("=" * 70)
print("PAIRWISE UTILITY DIFFERENCES")
print("=" * 70)

for a, b in pairs:
    print(
        f"{MODEL_NAMES[a]} - {MODEL_NAMES[b]}"
    )

# ============================================================
# TRAIN PAIRWISE REGRESSORS
# ============================================================

models = {}
regression_metrics = {}

print()
print("=" * 70)
print("TRAINING PAIRWISE F1 REGRESSORS")
print("=" * 70)

for model_a, model_b in pairs:

    target_name = (
        f"{model_a}_minus_{model_b}"
    )

    y_train = (
        train_df[F1_COLUMNS[model_a]]
        -
        train_df[F1_COLUMNS[model_b]]
    ).astype(np.float32)

    y_val = (
        val_df[F1_COLUMNS[model_a]]
        -
        val_df[F1_COLUMNS[model_b]]
    ).astype(np.float32)

    print()
    print(
        f"Training utility-difference model: "
        f"{MODEL_NAMES[model_a]} - "
        f"{MODEL_NAMES[model_b]}"
    )

    regressor = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    regressor.fit(
        X_train,
        y_train
    )

    prediction = regressor.predict(
        X_val
    )

    mae = mean_absolute_error(
        y_val,
        prediction
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_val,
            prediction
        )
    )

    r2 = r2_score(
        y_val,
        prediction
    )

    models[target_name] = regressor

    regression_metrics[target_name] = {
        "model_a": MODEL_NAMES[model_a],
        "model_b": MODEL_NAMES[model_b],
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }

    print(
        f"MAE  : {mae:.6f}"
    )

    print(
        f"RMSE : {rmse:.6f}"
    )

    print(
        f"R2   : {r2:.6f}"
    )

# ============================================================
# CONSTRUCT PREDICTED F1 SCORES
# ============================================================

print()
print("=" * 70)
print("RECONSTRUCTING MODEL UTILITY")
print("=" * 70)

# Use N as an anchor.
#
# Predicted pairwise differences:
#
# S - N
# L - N
# X - N
#
# These allow relative utility estimates:
#
# N = 0
# S = S-N
# L = L-N
# X = X-N
#
# Only relative ordering matters for routing.

predicted_relative = pd.DataFrame(
    index=val_df.index
)

predicted_relative["N"] = 0.0

# Pairwise regressors were trained with N as the
# first operand for these three comparisons:
#
# N-S, N-L, N-X
#
# Therefore:
#
# S-N = -(N-S)
# L-N = -(N-L)
# X-N = -(N-X)

for model in ["S", "L", "X"]:

    key = f"N_minus_{model}"

    predicted_relative[model] = -(
        models[key].predict(X_val)
    )

# ============================================================
# ROUTING DECISION
# ============================================================

router_indices = (
    predicted_relative
    .values
    .argmax(axis=1)
)

router_models = [
    MODEL_ORDER[i]
    for i in router_indices
]

router_model_names = [
    MODEL_NAMES[m]
    for m in router_models
]

# ============================================================
# TRUE F1 MATRIX
# ============================================================

true_f1 = val_df[
    [
        F1_COLUMNS[m]
        for m in MODEL_ORDER
    ]
].values

oracle_indices = (
    true_f1.argmax(axis=1)
)

oracle_models = [
    MODEL_ORDER[i]
    for i in oracle_indices
]

oracle_model_names = [
    MODEL_NAMES[m]
    for m in oracle_models
]

# ============================================================
# ADAPTIVE / ORACLE F1
# ============================================================

adaptive_f1 = np.array([
    true_f1[i, router_indices[i]]
    for i in range(len(val_df))
])

oracle_f1 = true_f1.max(
    axis=1
)

f1_regret = (
    oracle_f1
    -
    adaptive_f1
)

routing_accuracy = np.mean(
    np.array(router_models)
    ==
    np.array(oracle_models)
)

# ============================================================
# METRICS
# ============================================================

adaptive_mean_f1 = float(
    np.mean(adaptive_f1)
)

oracle_mean_f1 = float(
    np.mean(oracle_f1)
)

mean_regret = float(
    np.mean(f1_regret)
)

median_regret = float(
    np.median(f1_regret)
)

maximum_regret = float(
    np.max(f1_regret)
)

regret_thresholds = {
    "regret_le_0.01": float(
        np.mean(f1_regret <= 0.01)
    ),
    "regret_le_0.02": float(
        np.mean(f1_regret <= 0.02)
    ),
    "regret_le_0.05": float(
        np.mean(f1_regret <= 0.05)
    ),
    "regret_le_0.10": float(
        np.mean(f1_regret <= 0.10)
    ),
}

print()
print("=" * 70)
print("C6-C PAIRWISE ROUTER PERFORMANCE")
print("=" * 70)

print(
    f"Oracle F1             : "
    f"{oracle_mean_f1:.4f}"
)

print(
    f"Adaptive F1           : "
    f"{adaptive_mean_f1:.4f}"
)

print(
    f"Mean F1 regret        : "
    f"{mean_regret:.4f}"
)

print(
    f"Median F1 regret      : "
    f"{median_regret:.4f}"
)

print(
    f"Maximum F1 regret     : "
    f"{maximum_regret:.4f}"
)

print(
    f"Routing accuracy      : "
    f"{routing_accuracy:.4f}"
)

print()
print("Regret thresholds:")

for threshold, value in regret_thresholds.items():

    print(
        f"{threshold:20s}: "
        f"{value * 100:.2f}%"
    )

# ============================================================
# ROUTING DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("ROUTING DISTRIBUTION")
print("=" * 70)

for model in MODEL_ORDER:

    router_count = sum(
        m == model
        for m in router_models
    )

    oracle_count = sum(
        m == model
        for m in oracle_models
    )

    print(
        f"{MODEL_NAMES[model]}:"
        f" router={router_count:4d}"
        f" | oracle={oracle_count:4d}"
    )

# ============================================================
# CONFUSION MATRIX
# ============================================================

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(
    oracle_model_names,
    router_model_names,
    labels=[
        MODEL_NAMES[m]
        for m in MODEL_ORDER
    ]
)

print()
print("=" * 70)
print("PAIRWISE ROUTER CONFUSION MATRIX")
print("=" * 70)

print()
print(
    "Router   N    S    L    X"
)

for i, model in enumerate(MODEL_ORDER):

    print(
        f"{model:7s}"
        + "".join(
            f"{value:5d}"
            for value in cm[i]
        )
    )

# ============================================================
# PER-IMAGE RESULTS
# ============================================================

results = val_df[
    [
        "image",
        "gt_instances",
    ]
].copy()

results["router_model"] = router_model_names
results["oracle_model"] = oracle_model_names

results["adaptive_f1"] = adaptive_f1
results["oracle_f1"] = oracle_f1
results["f1_regret"] = f1_regret

for model in MODEL_ORDER:

    results[
        f"predicted_relative_{model}"
    ] = predicted_relative[model].values

results["oracle_match"] = (
    results["router_model"]
    ==
    results["oracle_model"]
)

results_path = (
    OUTPUT_DIR
    / "c6_pairwise_router_results.csv"
)

results.to_csv(
    results_path,
    index=False
)

print()
print(
    f"Per-image results saved to:\n"
    f"{results_path}"
)

# ============================================================
# SUMMARY
# ============================================================

summary = {
    "experiment": "C6-C",
    "method": (
        "Random Forest pairwise "
        "F1 utility-difference regression"
    ),
    "training_samples": int(
        len(train_df)
    ),
    "validation_samples": int(
        len(val_df)
    ),
    "oracle_f1": oracle_mean_f1,
    "adaptive_f1": adaptive_mean_f1,
    "mean_f1_regret": mean_regret,
    "median_f1_regret": median_regret,
    "maximum_f1_regret": maximum_regret,
    "routing_accuracy": float(
        routing_accuracy
    ),
    **regret_thresholds,
    "regression_metrics": regression_metrics,
}

summary_path = (
    OUTPUT_DIR
    / "c6_pairwise_router_summary.json"
)

with open(
    summary_path,
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )

metrics_path = (
    OUTPUT_DIR
    / "pairwise_regression_metrics.json"
)

with open(
    metrics_path,
    "w"
) as f:

    json.dump(
        regression_metrics,
        f,
        indent=4
    )

# ============================================================
# SAVE MODELS
# ============================================================

import joblib

for name, model in models.items():

    model_path = (
        OUTPUT_DIR
        / f"{name}_regressor.joblib"
    )

    joblib.dump(
        model,
        model_path
    )

# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("C6-C PAIRWISE ROUTER COMPLETE")
print("=" * 70)
