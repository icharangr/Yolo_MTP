from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# CONFIGURATION
# ============================================================

CSV_PATH = Path(
    "/home/charan/thesis/yolov13/experiments/"
    "adaptive_router/router_training_features.csv"
)

OUTPUT_DIR = Path(
    "/home/charan/thesis/yolov13/experiments/"
    "adaptive_router/router_model_c6_utility"
)

RANDOM_STATE = 42
TEST_SIZE = 0.20

MODELS = [
    "N",
    "S",
    "L",
    "X",
]

MODEL_LABELS = {
    "N": "YOLOv13-N",
    "S": "YOLOv13-S",
    "L": "YOLOv13-L",
    "X": "YOLOv13-X",
}

# ============================================================
# INFERENCE-TIME FEATURES
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

# ============================================================
# TARGETS
# ============================================================

TARGET_COLUMNS = {
    "N": "N_f1",
    "S": "S_f1",
    "L": "L_f1",
    "X": "X_f1",
}


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
print("C6-B — UTILITY-AWARE F1 REGRESSION ROUTER")
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

print(
    f"Total rows : {len(df)}"
)


# ============================================================
# VALIDATE COLUMNS
# ============================================================

required_columns = (
    FEATURE_COLUMNS
    + list(TARGET_COLUMNS.values())
)

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing_columns)
    )


# ============================================================
# CLEAN DATA
# ============================================================

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
    f"Rows available: "
    f"{len(df)}"
)


# ============================================================
# PREPARE FEATURES
# ============================================================

X = df[
    FEATURE_COLUMNS
].astype(np.float32)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

indices = np.arange(
    len(df)
)

train_idx, val_idx = train_test_split(
    indices,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

X_train = X.iloc[train_idx]
X_val = X.iloc[val_idx]

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


# ============================================================
# TRAIN FOUR F1 REGRESSORS
# ============================================================

regressors = {}

regression_metrics = {}

print()
print("=" * 70)
print("TRAINING F1 UTILITY REGRESSORS")
print("=" * 70)

for model in MODELS:

    target_column = TARGET_COLUMNS[
        model
    ]

    y = df[
        target_column
    ].astype(np.float32)

    y_train = y.iloc[
        train_idx
    ]

    y_val = y.iloc[
        val_idx
    ]

    print()
    print(
        f"Training utility model: "
        f"{MODEL_LABELS[model]}"
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

    regression_metrics[
        model
    ] = {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }

    regressors[
        model
    ] = regressor

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
# SAVE TRAINED F1 REGRESSORS
# ============================================================

print()
print("=" * 70)
print("SAVING TRAINED F1 REGRESSORS")
print("=" * 70)

for model in MODELS:

    model_path = (
        OUTPUT_DIR
        / f"{model}_f1_regressor.joblib"
    )

    joblib.dump(
        regressors[model],
        model_path
    )

    print(
        f"{MODEL_LABELS[model]} saved to:"
    )

    print(
        model_path
    )



# ============================================================
# PREDICT F1 UTILITY FOR ALL FOUR MODELS
# ============================================================

predicted_f1 = pd.DataFrame(
    index=X_val.index
)

actual_f1 = pd.DataFrame(
    index=X_val.index
)

for model in MODELS:

    target_column = TARGET_COLUMNS[
        model
    ]

    predicted_f1[
        f"{model}_predicted_f1"
    ] = regressors[
        model
    ].predict(X_val)

    actual_f1[
        f"{model}_actual_f1"
    ] = df.loc[
        X_val.index,
        target_column
    ].values


# ============================================================
# ROUTING DECISION
# ============================================================

predicted_columns = [
    f"{model}_predicted_f1"
    for model in MODELS
]

actual_columns = [
    f"{model}_actual_f1"
    for model in MODELS
]

predicted_matrix = predicted_f1[
    predicted_columns
].values

actual_matrix = actual_f1[
    actual_columns
].values

predicted_best_idx = np.argmax(
    predicted_matrix,
    axis=1
)

oracle_best_idx = np.argmax(
    actual_matrix,
    axis=1
)

adaptive_f1 = actual_matrix[
    np.arange(len(actual_matrix)),
    predicted_best_idx
]

oracle_f1 = actual_matrix[
    np.arange(len(actual_matrix)),
    oracle_best_idx
]

f1_regret = (
    oracle_f1
    -
    adaptive_f1
)

routing_correct = (
    predicted_best_idx
    ==
    oracle_best_idx
)

routed_models = [
    MODELS[index]
    for index in predicted_best_idx
]

oracle_models = [
    MODELS[index]
    for index in oracle_best_idx
]


# ============================================================
# UTILITY ROUTER METRICS
# ============================================================

mean_adaptive_f1 = np.mean(
    adaptive_f1
)

mean_oracle_f1 = np.mean(
    oracle_f1
)

mean_regret = np.mean(
    f1_regret
)

median_regret = np.median(
    f1_regret
)

max_regret = np.max(
    f1_regret
)

routing_accuracy = np.mean(
    routing_correct
)

regret_001 = np.mean(
    f1_regret <= 0.01
)

regret_002 = np.mean(
    f1_regret <= 0.02
)

regret_005 = np.mean(
    f1_regret <= 0.05
)

regret_010 = np.mean(
    f1_regret <= 0.10
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 70)
print("C6-B UTILITY ROUTER PERFORMANCE")
print("=" * 70)

print()
print(
    f"Oracle F1             : "
    f"{mean_oracle_f1:.4f}"
)

print(
    f"Adaptive F1           : "
    f"{mean_adaptive_f1:.4f}"
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
    f"{max_regret:.4f}"
)

print(
    f"Routing accuracy      : "
    f"{routing_accuracy:.4f}"
)

print()
print(
    "Regret thresholds:"
)

print(
    f"<= 0.01              : "
    f"{100 * regret_001:.2f}%"
)

print(
    f"<= 0.02              : "
    f"{100 * regret_002:.2f}%"
)

print(
    f"<= 0.05              : "
    f"{100 * regret_005:.2f}%"
)

print(
    f"<= 0.10              : "
    f"{100 * regret_010:.2f}%"
)


# ============================================================
# ROUTING DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("ROUTING DISTRIBUTION")
print("=" * 70)

for model in MODELS:

    router_count = sum(
        m == model
        for m in routed_models
    )

    oracle_count = sum(
        m == model
        for m in oracle_models
    )

    print(
        f"{MODEL_LABELS[model]}: "
        f"router={router_count:4d} | "
        f"oracle={oracle_count:4d}"
    )


# ============================================================
# ROUTER CONFUSION MATRIX
# ============================================================

print()
print("=" * 70)
print("UTILITY ROUTER CONFUSION MATRIX")
print("=" * 70)

cm = pd.crosstab(
    pd.Series(
        oracle_models,
        name="Oracle"
    ),
    pd.Series(
        routed_models,
        name="Router"
    ),
    dropna=False
)

cm = cm.reindex(
    index=MODELS,
    columns=MODELS,
    fill_value=0
)

print()
print(cm)


# ============================================================
# PER-IMAGE RESULTS
# ============================================================

results = pd.DataFrame({
    "image": df.loc[
        X_val.index,
        "image"
    ].values,

    "gt_instances": df.loc[
        X_val.index,
        "gt_instances"
    ].values
    if "gt_instances" in df.columns
    else np.nan,

    "router_model": [
        MODEL_LABELS[m]
        for m in routed_models
    ],

    "oracle_model": [
        MODEL_LABELS[m]
        for m in oracle_models
    ],

    "router_correct": routing_correct,

    "adaptive_f1": adaptive_f1,

    "oracle_f1": oracle_f1,

    "f1_regret": f1_regret,
})

for model in MODELS:

    results[
        f"{model}_predicted_f1"
    ] = predicted_f1[
        f"{model}_predicted_f1"
    ].values

    results[
        f"{model}_actual_f1"
    ] = actual_f1[
        f"{model}_actual_f1"
    ].values


results = results.sort_values(
    "f1_regret",
    ascending=False
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_path = (
    OUTPUT_DIR
    / "c6_utility_router_results.csv"
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
# SAVE REGRESSION METRICS
# ============================================================

regression_metrics_path = (
    OUTPUT_DIR
    / "utility_regression_metrics.json"
)

with open(
    regression_metrics_path,
    "w"
) as f:

    json.dump(
        regression_metrics,
        f,
        indent=4
    )

print(
    f"Regression metrics saved to:\n"
    f"{regression_metrics_path}"
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = {
    "experiment": "C6-B",
    "method": "Random Forest F1 utility regression",
    "training_samples": int(len(X_train)),
    "validation_samples": int(len(X_val)),

    "oracle_f1": float(
        mean_oracle_f1
    ),

    "adaptive_f1": float(
        mean_adaptive_f1
    ),

    "mean_f1_regret": float(
        mean_regret
    ),

    "median_f1_regret": float(
        median_regret
    ),

    "maximum_f1_regret": float(
        max_regret
    ),

    "routing_accuracy": float(
        routing_accuracy
    ),

    "regret_le_0.01": float(
        regret_001
    ),

    "regret_le_0.02": float(
        regret_002
    ),

    "regret_le_0.05": float(
        regret_005
    ),

    "regret_le_0.10": float(
        regret_010
    ),

    "regression_metrics": regression_metrics,
}

summary_path = (
    OUTPUT_DIR
    / "c6_utility_router_summary.json"
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

print(
    f"Summary saved to:\n"
    f"{summary_path}"
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("C6-B UTILITY ROUTER COMPLETE")
print("=" * 70)
