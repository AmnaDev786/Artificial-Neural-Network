# app.py
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os

app = Flask(__name__)

# ------------------ Load model, scaler, and features ------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # adjust if needed

model_path = os.path.join(BASE_DIR, "model", "ann_model.h5")
scaler_path = os.path.join(BASE_DIR, "model", "scaler.pkl")
features_path = os.path.join(BASE_DIR, "model", "features.pkl")

# Load ANN model
model = tf.keras.models.load_model(model_path)

# Load scaler
scaler = joblib.load(scaler_path)

# Load feature names (list of all numeric + one-hot column names)
features = joblib.load(features_path)  # should be list of strings

# ------------------ Flask Route ------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None
    input_data = {}  # store submitted values for keeping form filled

    if request.method == "POST":
        # 1️⃣ Collect all form inputs
        input_data = {
            "age": request.form.get("age", ""),
            "balance": request.form.get("balance", ""),
            "day": request.form.get("day", ""),
            "duration": request.form.get("duration", ""),
            "campaign": request.form.get("campaign", ""),
            "pdays": request.form.get("pdays", ""),
            "previous": request.form.get("previous", ""),
            "job": request.form.get("job", ""),
            "marital": request.form.get("marital", ""),
            "education": request.form.get("education", ""),
            "contact": request.form.get("contact", ""),
            "month": request.form.get("month", ""),
            "poutcome": request.form.get("poutcome", "")
        }

        try:
            # 2️⃣ Convert numeric inputs
            numeric_fields = ["age","balance","day","duration","campaign","pdays","previous"]
            numeric_input = {k: float(input_data[k]) if k=="balance" else int(input_data[k]) for k in numeric_fields}

            # 3️⃣ Create zero vector for all features
            input_dict = dict.fromkeys(features, 0)

            # Fill numeric values
            for key in numeric_input:
                input_dict[key] = numeric_input[key]

            # One-hot encode categorical dropdowns
            for col in ["job","marital","education","contact","month","poutcome"]:
                feature_name = f"{col}_{input_data[col]}"
                if feature_name in input_dict:  # check if this one-hot column exists
                    input_dict[feature_name] = 1

            # 4️⃣ Convert to DataFrame and scale
            input_df = pd.DataFrame([input_dict])
            input_scaled = scaler.transform(input_df)

            # 5️⃣ Make prediction
            prob = model.predict(input_scaled)[0][0]
            probability = round(float(prob), 4)
            prediction = "Deposit YES" if prob >= 0.5 else "Deposit NO"

        except Exception as e:
            prediction = f"Error: {str(e)}"
            probability = None

    return render_template("index.html", prediction=prediction, probability=probability, input_data=input_data)

# ------------------ Run Flask App ------------------
if __name__ == "__main__":
    app.run(debug=True)
