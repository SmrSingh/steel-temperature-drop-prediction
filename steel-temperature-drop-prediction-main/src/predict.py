"""
predict.py

Inference script for predicting the temperature drop and
estimated T40 temperature.
"""

from __future__ import annotations

from pathlib import Path

from utils import (
    load_model_artifact,
    prepare_prediction_dataframe,
)


MODEL_PATH = Path("models/best_model.pkl")


def predict(
    input_data: dict,
):

    artifact = load_model_artifact(MODEL_PATH)

    model = artifact["model"]
    feature_list = artifact["features"]
    model_name = artifact["model_name"]

    X = prepare_prediction_dataframe(
    input_data,
    feature_list,
    )

    predicted_drop = float(
        model.predict(X)[0]
    )

    predicted_t40 = (
        input_data["LF_LAST_TEMP"]
        -
        predicted_drop
    )

    return {
        "Model": model_name,
        "Predicted_Temperature_Drop": round(
            predicted_drop,
            2,
        ),
        "Predicted_T40_Temperature": round(
            predicted_t40,
            2,
        ),
    }


if __name__ == "__main__":

    sample_input = {

    "LF_LAST_TEMP": 1612,

    "LIQ_TEMP": 1570,

    "LFTEMP_CASTSTART_DUR": 18,

    "CASTSTART_TO_T40_DUR": 22,

    "LFTEMP_T40_DUR": 40,

    "LADLE_LIFE": 165,

    "TAT": 95,
}

    result = predict(sample_input)

    print()

    print("=" * 60)

    print("Prediction")

    print("=" * 60)

    for key, value in result.items():

        print(f"{key:30}: {value}")