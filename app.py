from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_DIR = Path(__file__).resolve().parent / "models"

st.set_page_config(page_title="Heart Disease Risk Predictor", page_icon="❤️")


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_DIR / "knn_heart.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    columns = joblib.load(MODEL_DIR / "columns.pkl")
    return model, scaler, columns


def build_features(columns, numeric, categorical):
    row = dict.fromkeys(columns, 0)
    row.update(numeric)
    for prefix, value in categorical.items():
        key = f"{prefix}_{value}"
        if key in row:
            row[key] = 1
    return pd.DataFrame([row])[columns]


model, scaler, expected_columns = load_artifacts()

st.title("Heart Disease Risk Predictor")
st.markdown("Enter the patient details below to estimate heart disease risk.")

age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

if st.button("Predict"):
    features = build_features(
        expected_columns,
        numeric={
            "Age": age,
            "RestingBP": resting_bp,
            "Cholesterol": cholesterol,
            "FastingBS": fasting_bs,
            "MaxHR": max_hr,
            "Oldpeak": oldpeak,
        },
        categorical={
            "Sex": sex,
            "ChestPainType": chest_pain,
            "RestingECG": resting_ecg,
            "ExerciseAngina": exercise_angina,
            "ST_Slope": st_slope,
        },
    )

    scaled = scaler.transform(features)
    prediction = model.predict(scaled)[0]
    risk = model.predict_proba(scaled)[0][1]

    if prediction == 1:
        st.error(f"High risk of heart disease (risk score {risk:.0%})")
    else:
        st.success(f"Low risk of heart disease (risk score {risk:.0%})")

    st.caption("For educational purposes only. Not a medical diagnosis.")
