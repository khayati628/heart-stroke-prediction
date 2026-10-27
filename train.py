from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "heart.csv"
MODEL_DIR = BASE_DIR / "models"

TARGET = "HeartDisease"
CATEGORICAL = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
RANDOM_STATE = 42


def load_features():
    df = pd.read_csv(DATA_PATH)
    y = df[TARGET]
    X = pd.get_dummies(df.drop(columns=[TARGET]), columns=CATEGORICAL, drop_first=True)
    return X.astype(float), y


def main():
    X, y = load_features()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    cv_scores = cross_val_score(
        make_pipeline(StandardScaler(), KNeighborsClassifier()),
        X_train,
        y_train,
        cv=5,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = KNeighborsClassifier()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    print(f"5-fold CV accuracy (train): {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
    print(f"Test accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=["No disease", "Disease"]))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "knn_heart.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(list(X.columns), MODEL_DIR / "columns.pkl")
    print(f"Saved model artifacts to {MODEL_DIR}")


if __name__ == "__main__":
    main()
