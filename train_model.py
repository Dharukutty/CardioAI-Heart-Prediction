"""Train a lightweight feed-forward ANN using scikit-learn MLPClassifier."""
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier

DATA_FILE = "heart_disease.csv"
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

df = pd.read_csv(DATA_FILE).replace("?", np.nan)
df = df.apply(pd.to_numeric, errors="coerce")
if "target" not in df.columns:
    if "num" in df.columns:
        df = df.rename(columns={"num": "target"})
    else:
        raise ValueError("CSV must contain 'target' or UCI 'num' column.")
missing = [c for c in FEATURES if c not in df.columns]
if missing:
    raise ValueError(f"Missing expected columns: {missing}")

df = df.dropna(subset=["target"]).drop_duplicates().copy()
y = (df["target"].astype(float) > 0).astype(int)
X = df[FEATURES].copy()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ANN architecture: 13 inputs -> 32 neurons -> 16 neurons -> 1 output.
model = MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu",
    solver="adam", alpha=0.001, batch_size=16, learning_rate_init=0.001,
    max_iter=500, early_stopping=True, validation_fraction=0.15,
    n_iter_no_change=20, random_state=42)
model.fit(X_train, y_train)

prob = model.predict_proba(X_test)[:, 1]
pred = (prob >= 0.5).astype(int)
print("Test accuracy:", round(accuracy_score(y_test, pred) * 100, 2), "%")
print("Confusion matrix:\n", confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, zero_division=0))
print("Training iterations:", model.n_iter_)

joblib.dump(model, "heart_disease_ann.joblib")
joblib.dump(imputer, "heart_imputer.pkl")
joblib.dump(scaler, "heart_scaler.pkl")
joblib.dump(FEATURES, "heart_features.pkl")
print("Saved model and preprocessing files.")
