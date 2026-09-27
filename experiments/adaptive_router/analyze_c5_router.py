#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(
    "/home/charan/thesis/yolov13"
)

C3_CSV = (
    ROOT
    / "experiments"
    / "adaptive_router"
    / "c3_results"
    / "adaptive_router_results.csv"
)

C4_JSON = (
    ROOT
    / "experiments"
    / "adaptive_router"
    / "c4_cost_aware"
    / "c4_cost_aware_metrics.json"
)

OUTPUT_DIR = (
    ROOT
    / "experiments"
    / "adaptive_router"
    / "c5_router_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


MODELS = [
    "N",
    "S",
    "L",
    "X",
]


# ============================================================
# LOAD C3
# ============================================================

print("=" * 70)
print("C5 — ROUTER DIAGNOSTIC ANALYSIS")
print("=" * 70)

print()
print("Loading C3 results:")
print(C3_CSV)

df = pd.read_csv(C3_CSV)

print(
    f"Validation images : {len(df)}"
)

print()


# ============================================================
# BASIC VALIDATION
# ============================================================

required_columns = [
    "image",
    "gt_instances",
    "router_model",
    "oracle_model",
    "oracle_match",
    "adaptive_f1",
    "oracle_f1",
]

for model in MODELS:

    required_columns.extend([
        f"{model}_f1",
        f"{model}_precision",
        f"{model}_recall",
        f"{model}_latency_ms",
    ])


missing = [
    c for c in required_columns
    if c not in df.columns
]

if missing:

    raise RuntimeError(
        "Missing columns:\n"
        + "\n".join(missing)
    )


# ============================================================
# NORMALIZE MODEL NAMES
# ============================================================

def short_model(name):

    name = str(name)

    return (
        name
        .replace("YOLOv13-", "")
        .replace("YOLOv13", "")
        .strip("- ")
    )


df["router_short"] = (
    df["router_model"]
    .apply(short_model)
)

df["oracle_short"] = (
    df["oracle_model"]
    .apply(short_model)
)


# ============================================================
# 1. ROUTING ACCURACY
# ============================================================

routing_accuracy = (
    df["oracle_match"]
    .astype(float)
    .mean()
)

print("=" * 70)
print("1. ROUTING ACCURACY")
print("=" * 70)

print(
    f"Overall router accuracy : "
    f"{routing_accuracy:.4f}"
)

print(
    f"Percentage              : "
    f"{routing_accuracy * 100:.2f}%"
)

print()


# ============================================================
# 2. CONFUSION MATRIX
# ============================================================

confusion = pd.crosstab(
    df["oracle_short"],
    df["router_short"],
    rownames=["Oracle"],
    colnames=["Router"],
    dropna=False,
)

confusion = confusion.reindex(
    index=MODELS,
    columns=MODELS,
    fill_value=0,
)

print("=" * 70)
print("2. ROUTER CONFUSION MATRIX")
print("=" * 70)

print()
print(confusion)
print()

confusion.to_csv(
    OUTPUT_DIR
    / "router_confusion_matrix.csv"
)


# ============================================================
# 3. PER-ORACLE-MODEL ACCURACY
# ============================================================

print("=" * 70)
print("3. PER-MODEL ROUTING ACCURACY")
print("=" * 70)

per_model_rows = []

for model in MODELS:

    subset = df[
        df["oracle_short"] == model
    ]

    if len(subset) == 0:
        continue

    accuracy = (
        subset["oracle_match"]
        .astype(float)
        .mean()
    )

    per_model_rows.append({
        "oracle_model": model,
        "images": len(subset),
        "routing_accuracy": accuracy,
        "routing_accuracy_percent":
            accuracy * 100,
        "mean_gt_instances":
            subset["gt_instances"].mean(),
        "mean_oracle_f1":
            subset["oracle_f1"].mean(),
        "mean_adaptive_f1":
            subset["adaptive_f1"].mean(),
    })

    print(
        f"{model}: "
        f"{len(subset):3d} images | "
        f"accuracy={accuracy*100:.2f}% | "
        f"mean GT={subset['gt_instances'].mean():.2f}"
    )

print()

per_model_df = pd.DataFrame(
    per_model_rows
)

per_model_df.to_csv(
    OUTPUT_DIR
    / "per_oracle_model_analysis.csv",
    index=False,
)


# ============================================================
# 4. ROUTING DISTRIBUTION
# ============================================================

print("=" * 70)
print("4. ROUTING DISTRIBUTION")
print("=" * 70)

router_distribution = (
    df["router_short"]
    .value_counts()
    .reindex(MODELS, fill_value=0)
)

oracle_distribution = (
    df["oracle_short"]
    .value_counts()
    .reindex(MODELS, fill_value=0)
)

for model in MODELS:

    router_count = int(
        router_distribution[model]
    )

    oracle_count = int(
        oracle_distribution[model]
    )

    print(
        f"{model}: "
        f"router={router_count:3d} | "
        f"oracle={oracle_count:3d}"
    )

print()


# ============================================================
# 5. ORACLE REGRET
# ============================================================

df["f1_regret"] = (
    df["oracle_f1"]
    -
    df["adaptive_f1"]
)

print("=" * 70)
print("5. ROUTING REGRET")
print("=" * 70)

print(
    f"Mean oracle F1       : "
    f"{df['oracle_f1'].mean():.4f}"
)

print(
    f"Mean adaptive F1     : "
    f"{df['adaptive_f1'].mean():.4f}"
)

print(
    f"Mean F1 regret       : "
    f"{df['f1_regret'].mean():.4f}"
)

print(
    f"Maximum F1 regret    : "
    f"{df['f1_regret'].max():.4f}"
)

print()


# ============================================================
# 6. F1 MARGIN
#
# How decisive is the oracle choice?
#
# Large margin:
#   Wrong routing is costly.
#
# Small margin:
#   Several models perform similarly.
# ============================================================

f1_columns = [
    f"{m}_f1"
    for m in MODELS
]

f1_matrix = df[
    f1_columns
].to_numpy()

sorted_f1 = np.sort(
    f1_matrix,
    axis=1
)

best_f1 = sorted_f1[:, -1]

second_best_f1 = sorted_f1[:, -2]

df["oracle_margin"] = (
    best_f1
    -
    second_best_f1
)

print("=" * 70)
print("6. ORACLE F1 MARGIN")
print("=" * 70)

print(
    f"Mean margin       : "
    f"{df['oracle_margin'].mean():.4f}"
)

print(
    f"Median margin     : "
    f"{df['oracle_margin'].median():.4f}"
)

print(
    f"90th percentile   : "
    f"{df['oracle_margin'].quantile(0.90):.4f}"
)

print()


# ============================================================
# 7. HOW OFTEN IS A WRONG ROUTE ACTUALLY BAD?
# ============================================================

wrong = df[
    df["oracle_match"] == 0
].copy()

print("=" * 70)
print("7. WRONG ROUTES — SEVERITY")
print("=" * 70)

print(
    f"Wrongly routed images : "
    f"{len(wrong)}"
)

if len(wrong) > 0:

    for threshold in [
        0.005,
        0.010,
        0.020,
        0.050,
        0.100,
    ]:

        fraction = (
            wrong["f1_regret"] <= threshold
        ).mean()

        print(
            f"Regret <= {threshold:.3f}: "
            f"{fraction*100:.2f}%"
        )

print()


# ============================================================
# 8. WRONG ROUTING BY OBJECT COUNT
# ============================================================

print("=" * 70)
print("8. ROUTING ACCURACY BY GT INSTANCE COUNT")
print("=" * 70)

bins = [
    -1,
    5,
    10,
    20,
    50,
    100,
    np.inf,
]

labels = [
    "0-5",
    "6-10",
    "11-20",
    "21-50",
    "51-100",
    "100+",
]

df["gt_density_group"] = pd.cut(
    df["gt_instances"],
    bins=bins,
    labels=labels,
)

density_rows = []

for group in labels:

    subset = df[
        df["gt_density_group"] == group
    ]

    if len(subset) == 0:
        continue

    acc = (
        subset["oracle_match"]
        .astype(float)
        .mean()
    )

    density_rows.append({
        "gt_group": group,
        "images": len(subset),
        "routing_accuracy": acc,
        "routing_accuracy_percent":
            acc * 100,
        "mean_oracle_f1":
            subset["oracle_f1"].mean(),
        "mean_f1_regret":
            subset["f1_regret"].mean(),
    })

    print(
        f"{group:>6}: "
        f"{len(subset):3d} images | "
        f"accuracy={acc*100:.2f}% | "
        f"regret={subset['f1_regret'].mean():.4f}"
    )

print()

pd.DataFrame(
    density_rows
).to_csv(
    OUTPUT_DIR
    / "routing_accuracy_by_gt_density.csv",
    index=False,
)


# ============================================================
# 9. MODEL-SPECIFIC F1 ADVANTAGE
# ============================================================

print("=" * 70)
print("9. ORACLE MODEL ADVANTAGE")
print("=" * 70)

advantage_rows = []

for model in MODELS:

    current = df[
        f"{model}_f1"
    ].to_numpy()

    other_columns = [
        f"{m}_f1"
        for m in MODELS
        if m != model
    ]

    other_best = df[
        other_columns
    ].max(axis=1)

    advantage = (
        current
        -
        other_best
    )

    oracle_subset = df[
        df["oracle_short"] == model
    ]

    if len(oracle_subset) == 0:
        continue

    oracle_advantage = advantage[
        df["oracle_short"] == model
    ]

    advantage_rows.append({
        "model": model,
        "oracle_images": len(oracle_subset),
        "mean_advantage_when_oracle":
            oracle_advantage.mean(),
        "median_advantage_when_oracle":
            np.median(oracle_advantage),
        "positive_advantage_fraction":
            (oracle_advantage > 0).mean(),
    })

    print(
        f"{model}: "
        f"mean advantage="
        f"{oracle_advantage.mean():.4f} | "
        f"positive="
        f"{(oracle_advantage > 0).mean()*100:.2f}%"
    )

print()

pd.DataFrame(
    advantage_rows
).to_csv(
    OUTPUT_DIR
    / "oracle_model_advantage.csv",
    index=False,
)


# ============================================================
# 10. HARDEST ROUTING CASES
# ============================================================

print("=" * 70)
print("10. TOP 30 HIGHEST-REGRET CASES")
print("=" * 70)

hard_cases = (
    df.sort_values(
        "f1_regret",
        ascending=False
    )
    [
        [
            "image",
            "gt_instances",
            "router_model",
            "oracle_model",
            "adaptive_f1",
            "oracle_f1",
            "f1_regret",
            "oracle_margin",
        ]
    ]
    .head(30)
)

print(
    hard_cases.to_string(
        index=False
    )
)

print()

hard_cases.to_csv(
    OUTPUT_DIR
    / "top_30_high_regret_cases.csv",
    index=False,
)


# ============================================================
# 11. C4 COST-AWARE SUMMARY
# ============================================================

print("=" * 70)
print("11. C4 COST-AWARE TRADE-OFF")
print("=" * 70)

if C4_JSON.exists():

    with open(C4_JSON) as f:
        c4 = json.load(f)

    rows = []

    for key, result in (
        c4["cost_aware_results"]
        .items()
    ):

        rows.append({
            "lambda":
                result["lambda"],

            "f1":
                result["f1"],

            "latency_ms":
                result["latency_ms"],

            "fps":
                result["fps"],

            "N":
                result["routing_distribution"]["N"],

            "S":
                result["routing_distribution"]["S"],

            "L":
                result["routing_distribution"]["L"],

            "X":
                result["routing_distribution"]["X"],
        })

    c4_df = pd.DataFrame(rows)

    print(
        c4_df.to_string(
            index=False
        )
    )

    print()

    c4_df.to_csv(
        OUTPUT_DIR
        / "c4_tradeoff_summary.csv",
        index=False,
    )


# ============================================================
# 12. SAVE ENRICHED PER-IMAGE DATA
# ============================================================

df.to_csv(
    OUTPUT_DIR
    / "c5_enriched_per_image.csv",
    index=False,
)


# ============================================================
# SUMMARY JSON
# ============================================================

summary = {

    "experiment":
        "C5_router_diagnostic_analysis",

    "validation_images":
        int(len(df)),

    "routing_accuracy":
        float(routing_accuracy),

    "mean_oracle_f1":
        float(df["oracle_f1"].mean()),

    "mean_adaptive_f1":
        float(df["adaptive_f1"].mean()),

    "mean_f1_regret":
        float(df["f1_regret"].mean()),

    "median_f1_regret":
        float(df["f1_regret"].median()),

    "mean_oracle_margin":
        float(df["oracle_margin"].mean()),

    "median_oracle_margin":
        float(df["oracle_margin"].median()),

    "router_distribution":
        {
            model:
                int(router_distribution[model])
            for model in MODELS
        },

    "oracle_distribution":
        {
            model:
                int(oracle_distribution[model])
            for model in MODELS
        },

    "wrong_routes":
        int((df["oracle_match"] == 0).sum()),

    "correct_routes":
        int((df["oracle_match"] == 1).sum()),
}


with open(
    OUTPUT_DIR
    / "c5_router_analysis_summary.json",
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


# ============================================================
# COMPLETE
# ============================================================

print("=" * 70)
print("C5 DIAGNOSTIC ANALYSIS COMPLETE")
print("=" * 70)

print()
print(
    "Results saved to:"
)

print(
    OUTPUT_DIR
)

print()
print(
    "IMPORTANT:"
)

print(
    "No model was retrained."
)

print(
    "No C3/C4 results were modified."
)

print(
    "This stage is diagnostic only."
)

print("=" * 70)
