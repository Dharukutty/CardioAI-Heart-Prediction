from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import json, os, sqlite3
from datetime import datetime, timezone
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:5173", "http://127.0.0.1:5173"]}})

# Lightweight scikit-learn ANN model (MLPClassifier)
model = joblib.load(os.path.join(BASE_DIR, "heart_disease_ann.joblib"))
imputer = joblib.load(os.path.join(BASE_DIR, "heart_imputer.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "heart_scaler.pkl"))
features = joblib.load(os.path.join(BASE_DIR, "heart_features.pkl"))
DATABASE_PATH = os.path.join(BASE_DIR, "cardioai_history.db")

def get_db_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection

def init_history_db():
    with get_db_connection() as connection:
        connection.execute("""CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL, values_json TEXT NOT NULL,
            prediction INTEGER NOT NULL, probability REAL, message TEXT
        )""")

init_history_db()

def make_prediction(data):
    if not isinstance(data, dict):
        raise ValueError("Please submit valid JSON data.")
    row, clean_values = [], {}
    for feature in features:
        if feature not in data:
            raise ValueError(f"Missing field: {feature}")
        value = float(data[feature])
        if not np.isfinite(value):
            raise ValueError(f"Invalid value for {feature}")
        row.append(value)
        clean_values[feature] = value
    x = np.asarray(row, dtype=np.float64).reshape(1, -1)
    x = scaler.transform(imputer.transform(x))
    prediction = int(model.predict(x)[0])
    return prediction, clean_values

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = probability = error = None
    values = {}
    if request.method == "POST":
        try:
            pred, values = make_prediction(request.form)
            prediction = "Higher predicted risk category" if pred == 1 else "Lower predicted risk category"
            probability = None
        except (ValueError, TypeError) as exc:
            error = str(exc)
    return render_template("index.html", features=features, prediction=prediction,
                           probability=probability, error=error, values=values)

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "model": "scikit-learn MLPClassifier ANN"})

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json(silent=True)
        prediction, _ = make_prediction(data)
        risk_category = "Higher predicted risk category" if prediction == 1 else "Lower predicted risk category"
        return jsonify({"prediction": prediction, "risk_category": risk_category,
                        "message": "ANN classification result. This is an educational prediction, not a medical diagnosis."})
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed. Please check backend logs."}), 500

@app.route("/api/history", methods=["GET", "POST"])
def prediction_history():
    if request.method == "GET":
        try:
            with get_db_connection() as connection:
                rows = connection.execute("""SELECT id, created_at, values_json,
                    prediction, probability, message FROM prediction_history
                    ORDER BY id DESC LIMIT 200""").fetchall()
            history = [{"id": r["id"], "created_at": r["created_at"],
                        "values": json.loads(r["values_json"]),
                        "prediction": r["prediction"], "probability": r["probability"],
                        "message": r["message"]} for r in rows]
            return jsonify({"history": history})
        except Exception:
            app.logger.exception("Could not read prediction history")
            return jsonify({"error": "Could not load prediction history."}), 500

    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not isinstance(data.get("values"), dict):
        return jsonify({"error": "Please submit valid prediction history data."}), 400
    values = data["values"]
    if any(f not in values for f in features):
        return jsonify({"error": "One or more required prediction fields are missing."}), 400
    try:
        clean = {f: float(values[f]) for f in features}
        if not all(np.isfinite(v) for v in clean.values()):
            raise ValueError("Prediction values must be finite numbers.")
        pred = int(data.get("prediction"))
        prob = None if data.get("probability") is None else float(data["probability"])
        if pred not in (0, 1) or (prob is not None and (not np.isfinite(prob) or not 0 <= prob <= 1)):
            raise ValueError("Invalid prediction or probability.")
        timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        with get_db_connection() as connection:
            cur = connection.execute("""INSERT INTO prediction_history
                (created_at, values_json, prediction, probability, message)
                VALUES (?, ?, ?, ?, ?)""",
                (timestamp, json.dumps(clean), pred, prob, str(data.get("message", ""))[:500]))
            record_id = cur.lastrowid
        return jsonify({"saved": True, "id": record_id, "created_at": timestamp}), 201
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        app.logger.exception("Could not save prediction history")
        return jsonify({"error": "Could not save prediction history."}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
