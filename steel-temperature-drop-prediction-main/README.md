# 🔥 Steel Temperature Prediction using Machine Learning

> An end-to-end Machine Learning project developed during my AI/ML internship at Tata Steel to predict molten steel temperature drop between the Ladle Furnace (LF1) and the T40 measurement before casting.

---

## Project Overview

In steel manufacturing, maintaining the correct casting temperature is critical for product quality.

After molten steel leaves the **Ladle Furnace (LF1)**, it loses heat while being transported to the caster. If the temperature is too low, casting quality can be affected. If it is too high, unnecessary energy is consumed.

This project predicts the expected **temperature drop** between:

- **LF Last Temperature**
- **T40 Temperature (before casting)**

The predicted temperature drop helps estimate the expected casting temperature and assists operators in selecting an appropriate LF exit temperature.

---

## Business Problem

The objective is to predict:

```text
LF2T40_TEMP_DROP
```

where

```text
LF2T40_TEMP_DROP = LF_LAST_TEMP − T40_TEMP
```

After predicting the temperature drop,

```text
Predicted T40 Temperature =
LF_LAST_TEMP − Predicted Temperature Drop
```

This enables proactive temperature control before casting.

---

## Dataset

The project uses approximately **six months of historical LF1 process data** collected from steel production.

### Input Features

| Feature | Description |
|----------|-------------|
| LF_LAST_TEMP | Temperature after Ladle Furnace |
| LIQ_TEMP | Liquidus Temperature |
| LFTEMP_CASTSTART_DUR | Time from LF measurement to casting start |
| CASTSTART_TO_T40_DUR | Time from casting start to T40 |
| LFTEMP_T40_DUR | Total time from LF to T40 |
| LADLE_LIFE | Number of heats completed by ladle |
| TAT | Turn Around Time |

### Engineered Features

The preprocessing pipeline automatically creates:

- TOTAL_PROCESS_TIME
- SUPERHEAT
- LF_TO_TOTAL_TIME_RATIO
- CAST_TO_TOTAL_TIME_RATIO
- LADLE_LIFE_PER_HOUR
- SUPERHEAT_PER_MIN

---

## Data Preprocessing

The preprocessing pipeline performs:

- Column validation
- Datetime conversion
- Duplicate removal
- Missing value handling
- TAT filtering (40–480 minutes)
- Target generation
- Invalid target removal
- Feature engineering

---

## Machine Learning Pipeline

The project follows a complete ML workflow:

```text
Raw Excel Data
        │
        ▼
Data Cleaning
        │
        ▼
Feature Engineering
        │
        ▼
Model Training
        │
        ▼
Hyperparameter Tuning
        │
        ▼
Model Evaluation
        │
        ▼
Prediction
        │
        ▼
Streamlit Application
```

---

## Model

Algorithm:

- Random Forest Regressor

Hyperparameter tuning:

- RandomizedSearchCV
- 5-Fold Cross Validation

Evaluation Metrics:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

Current Performance

| Metric | Value |
|---------|-------|
| MAE | ~6.9 °C |
| RMSE | ~9.5 °C |
| R² | ~0.38 |

> These results represent the current baseline model and will be improved in future iterations through enhanced feature engineering and data analysis.

---

## Project Structure

```text
Steel_Temperature_Prediction/
│
├── app.py
├── batch_predict.py
├── requirements.txt
├── README.md
├── LICENSE
│
├── data/
│   ├── LF1_Data.xlsx
│   └── cleaned_data.xlsx
│
├── models/
│   ├── best_model.pkl
│   ├── metrics.json
│   ├── feature_importance.csv
│   └── leaderboard.csv
│
├── notebooks/
│   ├── 01_Data_Understanding.ipynb
│   ├── 02_EDA.ipynb
│   └── 03_Model_Development.ipynb
│
├── outputs/
│   └── predictions/
│
└── src/
    ├── preprocessing.py
    ├── train.py
    ├── predict.py
    └── utils.py
```

---

## Installation

Clone the repository

```bash
git clone <repository-url>
```

Create virtual environment

```bash
python -m venv .venv
```

Activate environment

Windows

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Preprocess Data

```bash
python src/preprocessing.py
```

---

### 2. Train Model

```bash
python src/train.py
```

---

### 3. Single Prediction

```bash
python src/predict.py
```

---

### 4. Batch Prediction

```bash
python batch_predict.py
```

Predictions are saved to:

```text
outputs/predictions/predictions.xlsx
```

---

### 5. Launch Streamlit Application

```bash
streamlit run app.py
```

---

## Streamlit Application

The application allows users to:

- Enter process parameters
- Predict temperature drop
- Predict T40 temperature
- Calculate recommended LF exit temperature

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- OpenPyXL

---

## Key Learnings

This project provided practical experience in:

- Industrial Machine Learning
- Regression Modeling
- Feature Engineering
- Data Cleaning
- Hyperparameter Tuning
- Model Evaluation
- End-to-End ML Pipeline Development
- Building an Interactive ML Application

---

## Future Improvements

Planned improvements include:

- Additional domain-specific feature engineering
- Improved target validation
- Advanced error analysis
- Gradient Boosting and XGBoost comparison
- Explainable AI using SHAP
- Improved model performance
- Time-aware validation strategy

---

## Disclaimer

This project was developed for educational purposes during an internship. It demonstrates an end-to-end machine learning workflow using historical process data and is not intended for direct production use without additional validation.

---

## Author

**Rahul**

AI/ML Engineer

AI/ML Intern – Tata Steel