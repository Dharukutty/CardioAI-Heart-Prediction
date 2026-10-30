CARDIOAI LIGHTWEIGHT ANN UPDATE

This update replaces TensorFlow/Keras with scikit-learn MLPClassifier.
MLPClassifier is a feed-forward Artificial Neural Network (ANN).
It retains the current simple prediction and prediction-history API.
It does not add login, registration, patient dashboard, or admin dashboard.

STEPS
1. Keep your original, unscaled heart_disease.csv in the project folder.
2. Replace app.py, train_model.py and requirements.txt with these files.
3. In that folder run:
   py -m pip install -r requirements.txt
   py train_model.py
4. Training creates:
   heart_disease_ann.joblib
   heart_imputer.pkl
   heart_scaler.pkl
   heart_features.pkl
5. Upload updated app.py, train_model.py, requirements.txt and those four
   generated files to your GitHub repository.
6. Render Start Command:
   gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 1 --timeout 180 app:app

IMPORTANT: This is a newly trained model. Accuracy and predictions can differ
from the TensorFlow model. Do not use real patient data; this is an educational
demo, not a medical diagnosis.
