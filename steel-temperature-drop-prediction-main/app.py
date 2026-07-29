from __future__ import annotations

from src.utils import (
    load_model_artifact,
    prepare_prediction_dataframe,
)
import pandas as pd
import streamlit as st
from pathlib import Path

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------

MODEL_PATH = Path("models/best_model.pkl")

st.set_page_config(
    page_title="Steel Temperature Prediction",
    page_icon="🔥",
    layout="wide",
)

# -------------------------------------------------------
# Load Model
# -------------------------------------------------------

artifact = load_model_artifact(MODEL_PATH)

model = artifact["model"]
features = artifact["features"]

# -------------------------------------------------------
# Title
# -------------------------------------------------------

st.title("🔥 Steel Temperature Prediction")

st.write(
    """
Predict the expected temperature drop between the Ladle Furnace
and the T40 measurement.

The application also estimates the expected T40 temperature.
"""
)

# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------

st.sidebar.header("Process Inputs")

LF_LAST_TEMP = st.sidebar.number_input(
    "LF Last Temperature",
    value=1612.0,
)

LIQ_TEMP = st.sidebar.number_input(
    "Liquidus Temperature",
    value=1570.0,
)

LFTEMP_CASTSTART_DUR = st.sidebar.number_input(
    "LF Temp → Cast Start Duration",
    value=18.0,
)

CASTSTART_TO_T40_DUR = st.sidebar.number_input(
    "Cast Start → T40 Duration",
    value=22.0,
)

LFTEMP_T40_DUR = st.sidebar.number_input(
    "LF Temp → T40 Duration",
    value=40.0,
)

LADLE_LIFE = st.sidebar.number_input(
    "Ladle Life",
    value=165.0,
)

TAT = st.sidebar.number_input(
    "Turn Around Time",
    value=95.0,
)

TARGET_SUPERHEAT = st.sidebar.number_input(
    "Target Superheat",
    value=37.0,
)


# -------------------------------------------------------
# Prediction
# -------------------------------------------------------

if st.button("Predict"):

    raw_input = {
    "LF_LAST_TEMP": LF_LAST_TEMP,
    "LIQ_TEMP": LIQ_TEMP,
    "LFTEMP_CASTSTART_DUR": LFTEMP_CASTSTART_DUR,
    "CASTSTART_TO_T40_DUR": CASTSTART_TO_T40_DUR,
    "LFTEMP_T40_DUR": LFTEMP_T40_DUR,
    "LADLE_LIFE": LADLE_LIFE,
    "TAT": TAT,
    }

    input_df = prepare_prediction_dataframe(
        raw_input,
        features,
    )

    predicted_drop = model.predict(input_df)[0]

    predicted_t40 = (
        LF_LAST_TEMP -
        predicted_drop
    )

    required_cast_temp = (
        LIQ_TEMP +
        TARGET_SUPERHEAT
    )

    recommended_lf_exit = (
        required_cast_temp +
        predicted_drop
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Predicted Drop",
        f"{predicted_drop:.2f} °C",
    )

    col2.metric(
        "Predicted T40",
        f"{predicted_t40:.2f} °C",
    )

    col3.metric(
        "Recommended LF Exit",
        f"{recommended_lf_exit:.2f} °C",
    )

    st.success("Prediction completed successfully.")