# Heart Disease Risk Predictor

A small machine learning web app that estimates whether a patient is at high or low risk of heart disease from 11 clinical measurements. The model is a K-Nearest Neighbours classifier served through a Streamlit interface.

> For educational purposes only. This is not a medical diagnosis tool.

## Demo

```bash
streamlit run app.py
```

Fill in the form (age, blood pressure, cholesterol, ECG results, etc.), press **Predict**, and the app returns a high/low risk result with a risk score.

## Dataset

`data/heart.csv` contains 918 patient records with 11 features and a binary target (`HeartDisease`, 1 = disease, 0 = no disease), about 55% positive.

| Feature | Description |
|---|---|
| Age | Age in years |
| Sex | M / F |
| ChestPainType | TA (typical angina), ATA (atypical angina), NAP (non-anginal pain), ASY (asymptomatic) |
| RestingBP | Resting blood pressure (mm Hg) |
| Cholesterol | Serum cholesterol (mg/dL) |
| FastingBS | Fasting blood sugar > 120 mg/dL (1 = yes, 0 = no) |
| RestingECG | Normal, ST (ST-T wave abnormality), LVH (left ventricular hypertrophy) |
| MaxHR | Maximum heart rate achieved |
| ExerciseAngina | Exercise-induced angina (Y / N) |
| Oldpeak | ST depression induced by exercise relative to rest |
| ST_Slope | Slope of the peak exercise ST segment: Up, Flat, Down |

## Approach

1. One-hot encode the categorical features (`drop_first=True`).
2. Stratified 80/20 train/test split (`random_state=42`).
3. Standardise features with `StandardScaler`, fitted on the training set only.
4. Train a `KNeighborsClassifier` (default `k=5`).
5. Save the model, scaler, and column order to `models/` for the app.

## Results

| Metric | Value |
|---|---|
| 5-fold CV accuracy (train set) | 0.860 ± 0.034 |
| Test accuracy | 0.886 |
| Test precision / recall (disease class) | 0.89 / 0.91 |

Confusion matrix on the 184 test samples:

|  | Predicted no disease | Predicted disease |
|---|---|---|
| **Actual no disease** | 70 | 12 |
| **Actual disease** | 9 | 93 |

## Project structure

```
heart-stroke-prediction/
├── app.py              Streamlit app
├── train.py            Training and evaluation script
├── requirements.txt
├── data/
│   └── heart.csv
└── models/
    ├── knn_heart.pkl   Trained KNN model
    ├── scaler.pkl      Fitted StandardScaler
    └── columns.pkl     Feature order expected by the model
```

## Getting started

```bash
git clone https://github.com/<your-username>/heart-stroke-prediction.git
cd heart-stroke-prediction

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
```

To retrain the model and regenerate the files in `models/`:

```bash
python train.py
```

## Tech stack

Python, pandas, scikit-learn, Streamlit, joblib.

## Limitations

- About 19% of rows have `Cholesterol = 0`, which is a missing value in the original data and is not imputed here.
- KNN is simple and sensitive to feature scaling and the choice of `k`; no hyperparameter tuning has been done.
- Trained on a single small dataset, so results will not generalise to real clinical use.

## Possible improvements

- Impute the zero cholesterol values and compare models (Logistic Regression, Random Forest, XGBoost).
- Tune `k` with grid search and report ROC-AUC.
- Deploy on Streamlit Community Cloud.

## Author

Khayati
