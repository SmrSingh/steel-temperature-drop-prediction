"""
utils.py

Shared utility functions for the
Steel Temperature Prediction project.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# General Utilities
# ============================================================

def set_random_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)


def ensure_directory(path: str | Path) -> Path:
    """
    Create directory if it does not exist.
    """

    path = Path(path)

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


# ============================================================
# Model Utilities
# ============================================================

def load_model_artifact(model_path: str | Path) -> dict:
    """
    Load saved model artifact.
    """

    return joblib.load(model_path)


def save_model_artifact(
    artifact: dict,
    model_path: str | Path,
) -> None:
    """
    Save model artifact.
    """

    model_path = Path(model_path)

    ensure_directory(model_path.parent)

    joblib.dump(
        artifact,
        model_path,
    )


# ============================================================
# Metrics
# ============================================================

def save_metrics(
    metrics: dict,
    output_path: str | Path,
) -> None:

    output_path = Path(output_path)

    ensure_directory(output_path.parent)

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )


def save_feature_importance(
    feature_names: list[str],
    importance,
    output_path: str | Path,
) -> None:

    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importance,
        }
    )

    importance_df.sort_values(
        by="Importance",
        ascending=False,
        inplace=True,
    )

    output_path = Path(output_path)

    ensure_directory(output_path.parent)

    importance_df.to_csv(
        output_path,
        index=False,
    )


# ============================================================
# Feature Engineering
# ============================================================

def engineer_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create all engineered features required by the model.

    This function MUST remain identical for both
    training and inference.
    """

    df = df.copy()

    df["TOTAL_PROCESS_TIME"] = (
        df["LFTEMP_CASTSTART_DUR"]
        +
        df["CASTSTART_TO_T40_DUR"]
    )

    df["SUPERHEAT"] = (
        df["LF_LAST_TEMP"]
        -
        df["LIQ_TEMP"]
    )

    df["LF_TO_TOTAL_TIME_RATIO"] = (
        df["LFTEMP_CASTSTART_DUR"]
        /
        df["LFTEMP_T40_DUR"]
    )

    df["CAST_TO_TOTAL_TIME_RATIO"] = (
        df["CASTSTART_TO_T40_DUR"]
        /
        df["LFTEMP_T40_DUR"]
    )

    df["LADLE_LIFE_PER_HOUR"] = (
        df["LADLE_LIFE"]
        /
        (df["TAT"] / 60)
    )

    df["SUPERHEAT_PER_MIN"] = (
        df["SUPERHEAT"]
        /
        df["TOTAL_PROCESS_TIME"]
    )

    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True,
    )

    df.dropna(
        inplace=True,
    )

    df.reset_index(
        drop=True,
        inplace=True,
    )

    return df


# ============================================================
# Prediction Utilities
# ============================================================

def prepare_prediction_dataframe(
    input_data: dict,
    feature_list: list[str],
) -> pd.DataFrame:
    """
    Prepare a prediction dataframe from raw user input.
    """

    df = pd.DataFrame([input_data])

    df = engineer_features(df)

    missing = [
        feature
        for feature in feature_list
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing input features: {missing}"
        )

    return df[feature_list]