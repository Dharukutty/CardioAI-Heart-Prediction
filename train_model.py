"""Train an educational ANN heart-disease classifier from heart_disease.csv."""
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

DATA_FILE = "heart_disease.csv"
FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

# Load and clean the original, unscaled CSV.
df = pd.read_csv(DATA_FILE).replace("?", np.nan)
df = df.apply(pd.to_numeric, errors="coerce")
if "target" not in df.columns:
    if "num" in df.columns:
        df = df.rename(columns={"num": "target"})
    else:
        raise ValueError("CSV must contain a 'target' column (or UCI 'num' column).")

missing_features = [c for c in FEATURES if c not in df.columns]
if missing_features:
    raise ValueError(f"Missing expected columns: {missing_features}")

df = df.dropna(subset=["target"]).drop_duplicates().copy()
# UCI Cleveland labels may be 0-4; convert to binary: 0 vs >0.
y = (df["target"].astype(float) > 0).astype(int)
X = df[FEATURES].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Fit preprocessing only on training data to avoid test-data leakage.
imputer = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled = scaler.transform(X_test_imp)

model = Sequential([
    Input(shape=(len(FEATURES),)),
    Dense(32, activation="relu"),
    Dropout(0.3),
    Dense(16, activation="relu"),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
early_stop = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
model.fit(
    X_train_scaled, y_train,
    epochs=100, batch_size=16, validation_split=0.2,
    callbacks=[early_stop], verbose=1
)

prob = model.predict(X_test_scaled, verbose=0).ravel()
pred = (prob >= 0.5).astype(int)
print("Test accuracy:", round(accuracy_score(y_test, pred) * 100, 2), "%")
print("Confusion matrix:\n", confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, zero_division=0))

model.save("heart_disease_ann.keras")
joblib.dump(imputer, "heart_imputer.pkl")
joblib.dump(scaler, "heart_scaler.pkl")
joblib.dump(FEATURES, "heart_features.pkl")
print("Saved: heart_disease_ann.keras, heart_imputer.pkl, heart_scaler.pkl, heart_features.pkl")
