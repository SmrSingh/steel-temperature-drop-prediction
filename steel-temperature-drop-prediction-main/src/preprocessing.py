"""
preprocessing.py

Data preprocessing and feature engineering for the
Steel Temperature Prediction project.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "HEAT_ID",
    "LF_LAST_TEMP_TIME",
    "CAST_START",
    "T40_TIME",
    "LFTEMP_CASTSTART_DUR",
    "CASTSTART_TO_T40_DUR",
    "LFTEMP_T40_DUR",
    "LIQ_TEMP",
    "LF_LAST_TEMP",
    "LADLE_LIFE",
    "TAT",
    "T40_TEMP",
]

TARGET_COLUMN = "LF2T40_TEMP_DROP"


# ---------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------

def load_data(file_path: str | Path) -> pd.DataFrame:
    return pd.read_excel(file_path)


# ---------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    return df


def validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )


def convert_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    datetime_columns = [
        "LF_LAST_TEMP_TIME",
        "CAST_START",
        "T40_TIME",
    ]

    for col in datetime_columns:
        df[col] = pd.to_datetime(
            df[col],
            errors="coerce",
        )

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df
        .drop_duplicates()
        .reset_index(drop=True)
    )


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df
        .dropna(subset=REQUIRED_COLUMNS)
        .reset_index(drop=True)
    )


def filter_tat(df: pd.DataFrame) -> pd.DataFrame:

    return (
        df[
            (df["TAT"] >= 40)
            &
            (df["TAT"] <= 480)
        ]
        .reset_index(drop=True)
    )


def create_target(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    if TARGET_COLUMN not in df.columns:
        df[TARGET_COLUMN] = (
            df["LF_LAST_TEMP"]
            -
            df["T40_TEMP"]
        )

    return df


# ---------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # ---------------------------------------------------------------
    # Overall processing duration
    # ---------------------------------------------------------------

    df["TOTAL_PROCESS_TIME"] = (
        df["LFTEMP_CASTSTART_DUR"]
        +
        df["CASTSTART_TO_T40_DUR"]
    )

    # ---------------------------------------------------------------
    # Superheat available after LF
    # ---------------------------------------------------------------

    df["SUPERHEAT"] = (
        df["LF_LAST_TEMP"]
        -
        df["LIQ_TEMP"]
    )

    # ---------------------------------------------------------------
    # Ratio Features
    # ---------------------------------------------------------------

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

    # ---------------------------------------------------------------
    # Ladle utilization
    # ---------------------------------------------------------------

    df["LADLE_LIFE_PER_HOUR"] = (
        df["LADLE_LIFE"]
        /
        (df["TAT"] / 60)
    )

    # ---------------------------------------------------------------
    # Cooling rate proxy
    # (NOT target leakage)
    # ---------------------------------------------------------------

    df["SUPERHEAT_PER_MIN"] = (
        df["SUPERHEAT"]
        /
        df["TOTAL_PROCESS_TIME"]
    )

    # ---------------------------------------------------------------
    # Numerical stability
    # ---------------------------------------------------------------

    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True,
    )

    df.dropna(inplace=True)

    df.reset_index(
        drop=True,
        inplace=True,
    )

    return df


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------

def preprocess_data(file_path: str | Path) -> pd.DataFrame:

    df = load_data(file_path)

    df = standardize_columns(df)

    validate_columns(df)

    df = convert_datetime_columns(df)

    df = remove_duplicates(df)

    df = handle_missing_values(df)

    df = filter_tat(df)

    df = create_target(df)
    
    df = filter_invalid_target(df)

    df = engineer_features(df)

    return df


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

def save_cleaned_data(
    df: pd.DataFrame,
    output_path: str | Path,
):

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_excel(
        output_path,
        index=False,
    )
    

def filter_invalid_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove impossible temperature drops.
    """

    df = df.copy()

    df = df[
        (df["LF2T40_TEMP_DROP"] >= 0)
        &
        (df["LF2T40_TEMP_DROP"] <= 120)
    ]

    df.reset_index(drop=True, inplace=True)

    return df


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":

    INPUT_FILE = Path("data/LF1_Data.xlsx")

    OUTPUT_FILE = Path("data/cleaned_data.xlsx")

    cleaned_df = preprocess_data(INPUT_FILE)

    save_cleaned_data(
        cleaned_df,
        OUTPUT_FILE,
    )

    print(f"Rows : {len(cleaned_df)}")
    print(f"Columns : {cleaned_df.shape[1]}")
    print(f"Saved : {OUTPUT_FILE}")