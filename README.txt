HEART DISEASE ANN + FLASK STARTER

IMPORTANT: This starter retrains the ANN and saves the exact preprocessing objects used by Flask. Do not mix a model trained with one preprocessing/scaling setup with a different scaler.

1. Install Python 3.10 or 3.11.
2. Put your ORIGINAL, UNSCALED heart_disease.csv file in this folder. It must have columns:
   age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target
   (UCI original files using 'num' as target are also supported.)
3. Open a terminal in this folder and run:
   python -m venv .venv
   Windows: .venv\Scripts\activate
   macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   python train_model.py
4. Training creates these artifacts:
   heart_disease_ann.keras
   heart_imputer.pkl
   heart_scaler.pkl
   heart_features.pkl
5. Start the website:
   python app.py
6. Open http://127.0.0.1:5000 in your browser.

If you currently only have the scaled heart_disease_preprocessed.csv, download the ORIGINAL unscaled CSV from UCI first. The training script needs raw input values so it can fit and save the same imputer/scaler that the Flask app will use.

The prediction is an educational demonstration, not a medical diagnosis. The form uses the dataset's numerical codes for categorical variables; enter valid codes matching the source dataset.
