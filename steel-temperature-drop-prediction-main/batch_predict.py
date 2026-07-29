"""
batch_predict.py

Generate temperature drop and T40 predictions
for an entire Excel dataset.
"""

from __future__ import annotations

from pathlib import Path

from src.utils import load_model_artifact
import pandas as pd

# -----------------------------------------------------
# Configuration
# -----------------------------------------------------

INPUT_FILE = Path("data/cleaned_data.xlsx")
OUTPUT_DIR = Path("outputs/predictions")
OUTPUT_FILE = OUTPUT_DIR / "predictions.xlsx"

MODEL_PATH = Path("models/best_model.pkl")


# -----------------------------------------------------
# Load Model
# -----------------------------------------------------

artifact = load_model_artifact(MODEL_PATH)

model = artifact["model"]
features = artifact["features"]


# -----------------------------------------------------
# Load Data
# -----------------------------------------------------

df = pd.read_excel(INPUT_FILE)

X = df[features].copy()


# -----------------------------------------------------
# Prediction
# -----------------------------------------------------

predicted_drop = model.predict(X)

df["PREDICTED_TEMP_DROP"] = predicted_drop

df["PREDICTED_T40_TEMP"] = (
    df["LF_LAST_TEMP"] -
    df["PREDICTED_TEMP_DROP"]
)

# Optional comparison if actual values exist
if "T40_TEMP" in df.columns:

    df["TEMPERATURE_ERROR"] = (
        df["T40_TEMP"] -
        df["PREDICTED_T40_TEMP"]
    )

    df["ABS_ERROR"] = (
        df["TEMPERATURE_ERROR"]
        .abs()
    )


# -----------------------------------------------------
# Save
# -----------------------------------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

df.to_excel(
    OUTPUT_FILE,
    index=False,
)

print("=" * 60)
print("Batch Prediction Completed")
print("=" * 60)
print(f"Input File : {INPUT_FILE}")
print(f"Rows       : {len(df)}")
print(f"Output     : {OUTPUT_FILE}")