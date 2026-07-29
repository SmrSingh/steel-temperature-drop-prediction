"""
train.py

Train multiple regression models and automatically save
the best-performing model.
"""

from __future__ import annotations
from sklearn.model_selection import RandomizedSearchCV
import json
from pathlib import Path

from utils import (
    save_model_artifact,
    save_metrics,
    save_feature_importance,
    ensure_directory
)

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    RepeatedKFold,
    cross_val_score,
    train_test_split,
)

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DATA_PATH = Path("data/cleaned_data.xlsx")

MODEL_DIR = Path("models")

MODEL_PATH = MODEL_DIR / "best_model.pkl"
LEADERBOARD_PATH = MODEL_DIR / "leaderboard.csv"
METRICS_PATH = MODEL_DIR / "metrics.json"
FEATURE_IMPORTANCE_PATH = MODEL_DIR / "feature_importance.csv"

MODEL_PATH = MODEL_DIR / "best_model.pkl"
LEADERBOARD_PATH = MODEL_DIR / "leaderboard.csv"
METRICS_PATH = MODEL_DIR / "metrics.json"

TARGET = "LF2T40_TEMP_DROP"

FEATURES = [
    "LF_LAST_TEMP",
    "LIQ_TEMP",
    "LFTEMP_CASTSTART_DUR",
    "CASTSTART_TO_T40_DUR",
    "LFTEMP_T40_DUR",
    "LADLE_LIFE",
    "TAT",
    "TOTAL_PROCESS_TIME",
    "SUPERHEAT",
    "LF_TO_TOTAL_TIME_RATIO",
    "CAST_TO_TOTAL_TIME_RATIO",
    "LADLE_LIFE_PER_HOUR",
    "SUPERHEAT_PER_MIN",
]

TEST_SIZE = 0.20
RANDOM_STATE = 42


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------

def get_models():

    return {
        "RandomForest": RandomForestRegressor(
            n_estimators=600,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=600,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "HistGradientBoosting": HistGradientBoostingRegressor(
            random_state=RANDOM_STATE,
        ),
    }


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def evaluate_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
):

    cv = RepeatedKFold(
        n_splits=5,
        n_repeats=3,
        random_state=RANDOM_STATE,
    )

    cv_mae = (
        -cross_val_score(
            clone(model),
            X_train,
            y_train,
            scoring="neg_mean_absolute_error",
            cv=cv,
            n_jobs=-1,
        )
    ).mean()

    model.fit(
        X_train,
        y_train,
    )

    prediction = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        prediction,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            prediction,
        )
    )

    r2 = r2_score(
        y_test,
        prediction,
    )

    return {
        "cv_mae": cv_mae,
        "test_mae": mae,
        "rmse": rmse,
        "r2": r2,
        "model": model,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():

    MODEL_DIR.mkdir(exist_ok=True)

    df = pd.read_excel(DATA_PATH)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print("=" * 70)
    print("Hyperparameter Tuning : Random Forest")
    print("=" * 70)

    parameter_grid = {
        "n_estimators": [200, 400, 600, 800, 1000],
        "max_depth": [None, 10, 20, 30, 40],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
        "bootstrap": [True, False],
    }

    base_model = RandomForestRegressor(
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=parameter_grid,
        n_iter=50,
        scoring="neg_mean_absolute_error",
        cv=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=2,
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    predictions = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    r2 = r2_score(y_test, predictions)

    print("\n")
    print("=" * 70)
    print("Best Hyperparameters")
    print("=" * 70)

    for key, value in search.best_params_.items():
        print(f"{key}: {value}")

    print("\n")
    print("=" * 70)
    print("Model Performance")
    print("=" * 70)

    print(f"MAE  : {mae:.3f}")
    print(f"RMSE : {rmse:.3f}")
    print(f"R2   : {r2:.4f}")

    feature_importance = pd.DataFrame(
        {
            "Feature": FEATURES,
            "Importance": best_model.feature_importances_,
        }
    ).sort_values(
        by="Importance",
        ascending=False,
    )

    save_feature_importance(
        FEATURE_IMPORTANCE_PATH,
        index=False,
    )

    metrics = {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "Best Parameters": search.best_params_,
    }

    with open(METRICS_PATH, "w") as f:
        save_metrics(metrics, f, indent=4)

    artifact = {
        "model": best_model,
        "features": FEATURES,
        "model_name": "RandomForest",
    }

    save_model_artifact(
        artifact,
        MODEL_PATH,
    )

    print("\n")
    print("=" * 70)
    print("Training Completed Successfully")
    print("=" * 70)


if __name__ == "__main__":
    main()