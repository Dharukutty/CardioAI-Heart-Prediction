from flask import Flask, render_template, request, jsonify
import json
import os
import sqlite3
from datetime import datetime, timezone
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(
    app,
    resources={
        r"/api/*": {
            "origins": [
                "http://localhost:5173",
                "http://127.0.0.1:5173"
            ]
        }
    }
)
model = load_model("heart_disease_ann.keras")
imputer = joblib.load("heart_imputer.pkl")
scaler = joblib.load("heart_scaler.pkl")
features = joblib.load("heart_features.pkl")

# Prediction history is stored in a small SQLite database beside this app.py file.
DATABASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cardioai_history.db")


def get_db_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_history_db():
    with get_db_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                values_json TEXT NOT NULL,
                prediction INTEGER NOT NULL,
                probability REAL,
                message TEXT
            )
            """
        )


init_history_db()


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None
    error = None
    values = {}

    if request.method == "POST":
        try:
            # Read in the same feature order used during training.
            row = []
            for feature in features:
                raw = request.form.get(feature, "").strip()
                if raw == "":
                    raise ValueError(f"Please enter a value for {feature}.")
                value = float(raw)
                if not np.isfinite(value):
                    raise ValueError(f"Please enter a valid number for {feature}.")
                row.append(value)
                values[feature] = raw

            x = np.array(row, dtype=float).reshape(1, -1)
            x = imputer.transform(x)
            x = scaler.transform(x)
            probability = float(model.predict(x, verbose=0)[0][0])
            prediction = "Higher predicted risk" if probability >= 0.5 else "Lower predicted risk"
        except (ValueError, TypeError) as exc:
            error = str(exc)

    return render_template(
        "index.html", features=features, prediction=prediction,
        probability=probability, error=error, values=values
    )
 

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "error": "Please submit valid JSON data."
            }), 400

        row = []

        for feature in features:
            if feature not in data:
                return jsonify({
                    "error": f"Missing field: {feature}"
                }), 400

            value = float(data[feature])

            if not np.isfinite(value):
                return jsonify({
                    "error": f"Invalid value for {feature}"
                }), 400

            row.append(value)

        # Convert input into the correct shape and data type.
        x = np.asarray(row, dtype=np.float32).reshape(1, -1)

        # Apply the same preprocessing used during model training.
        x = imputer.transform(x)
        x = scaler.transform(x)

        # Convert the input to a TensorFlow tensor.
        x_tensor = tf.convert_to_tensor(
            x,
            dtype=tf.float32
        )

        # Get the actual ANN prediction.
        probability = float(
            model.predict(x, verbose=0).reshape(-1)[0]
        )

        prediction = 1 if probability >= 0.5 else 0

        return jsonify({
            "prediction": prediction,
            "probability": probability,
            "message": (
                "Heart disease class predicted by the ANN model."
            )
        })

    except (ValueError, TypeError) as exc:
        return jsonify({
            "error": str(exc)
        }), 400

    except Exception:
        app.logger.exception("Prediction failed")

        return jsonify({
            "error": "Prediction failed. Please check the backend."
        }), 500

@app.route("/api/history", methods=["GET", "POST"])
def prediction_history():
    """List saved predictions or save one real response from the ANN API."""
    if request.method == "GET":
        try:
            with get_db_connection() as connection:
                rows = connection.execute(
                    """
                    SELECT id, created_at, values_json, prediction, probability, message
                    FROM prediction_history
                    ORDER BY id DESC
                    LIMIT 200
                    """
                ).fetchall()

            history = []
            for row in rows:
                history.append({
                    "id": row["id"],
                    "created_at": row["created_at"],
                    "values": json.loads(row["values_json"]),
                    "prediction": row["prediction"],
                    "probability": row["probability"],
                    "message": row["message"],
                })
            return jsonify({"history": history})
        except Exception:
            app.logger.exception("Could not read prediction history")
            return jsonify({"error": "Could not load prediction history."}), 500

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Please submit valid JSON data."}), 400

    values = data.get("values")
    prediction = data.get("prediction")
    probability = data.get("probability")
    message = data.get("message")

    if not isinstance(values, dict):
        return jsonify({"error": "The prediction values must be a JSON object."}), 400

    # Store only the known model inputs, in the model's expected feature order.
    if any(feature not in values for feature in features):
        return jsonify({"error": "One or more required prediction fields are missing."}), 400

    try:
        clean_values = {}
        for feature in features:
            value = float(values[feature])
            if not np.isfinite(value):
                raise ValueError(f"Invalid value for {feature}.")
            clean_values[feature] = value

        prediction_number = int(prediction)
        if prediction_number not in (0, 1):
            raise ValueError("Prediction must be class 0 or class 1.")

        probability_number = None if probability is None else float(probability)
        if probability_number is not None:
            if not np.isfinite(probability_number) or not 0 <= probability_number <= 1:
                raise ValueError("Probability must be between 0 and 1.")

        timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        with get_db_connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO prediction_history
                    (created_at, values_json, prediction, probability, message)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    json.dumps(clean_values),
                    prediction_number,
                    probability_number,
                    str(message)[:500] if message is not None else None,
                ),
            )
            record_id = cursor.lastrowid

        return jsonify({"saved": True, "id": record_id, "created_at": timestamp}), 201
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        app.logger.exception("Could not save prediction history")
        return jsonify({"error": "Could not save prediction history."}), 500


if __name__ == "__main__":
    app.run(debug=True)
