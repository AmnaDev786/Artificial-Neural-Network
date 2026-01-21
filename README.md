# Bank Term Deposit Prediction using Artificial Neural Network (ANN)

## 📌 Project Overview
This project implements an **Artificial Neural Network (ANN)**–based system to predict whether a bank customer will subscribe to a **term deposit**.  
It includes a complete backend–frontend pipeline with trained models, preprocessing artifacts, and a simple web interface for prediction.

---

## 📂 Project Structure
```bash
ANNProject/
│
├─ backend/
│   ├─ model/
│   │   ├─ bank_ann_model.keras
│   │   ├─ feature.pkl
│   │   └─ scaler.pkl
│   │
│   ├─ template/
│   │   └─ index.html
│   │
│   └─ app.py
│
├─ model/
│   ├─ ann_model.h5
│   ├─ bank_ann_model.keras
│   ├─ feature.pkl
│   ├─ scaler.pkl
│   └─ threshold.pkl
│
├─ data/
│   └─ bank/
│
├─ frontend/
│   └─ main.py
│
├─ requirements.txt
└─ README.md
```

---

## 🧠 Model Description
- **Model Type:** Fully Connected Feedforward ANN  
- **Architecture:** 256 → 128 → 64 → 1  
- **Activation Functions:**  
  - ReLU (Hidden Layers)  
  - Sigmoid (Output Layer)  
- **Loss Function:** Binary Cross-Entropy  
- **Optimizer:** Adam  
- **Regularization:** Dropout, L2 Regularization, Batch Normalization  
- **Class Imbalance Handling:** SMOTE (applied only to training data)

---

## ⚙️ Data Preprocessing
- One-hot encoding for categorical features  
- Label encoding for target variable  
- Feature scaling using `StandardScaler`  
- Stored preprocessing objects:
  - `feature.pkl`
  - `scaler.pkl`
  - `threshold.pkl` (optimized classification threshold)

---

## 📊 Model Performance
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score, Confusion Matrix  

### Confusion Matrix Summary
| Category | Count |
|--------|------|
| True Deposit | 495 |
| True No Deposit | 462 |
| False Positives | 126 |
| False Negatives | 34 |

The low false-negative rate indicates effective identification of potential deposit customers.

---

## 🌐 Application Workflow
1. User enters customer details via frontend
2. Input data is preprocessed using saved scalers and encoders
3. ANN model predicts deposit probability
4. Optimized threshold determines final output
5. Result is displayed on the interface

---

## 🚀 How to Run the Project

### 1. Clone the Repository
```bash
pip install -r requirements.txt
python backend/app.py
python frontend/main.py
```

## Technologies Used
Python 3.x
TensorFlow / Keras
Scikit-learn
Pandas, NumPy
Flask
HTML

## Use Cases

Bank marketing campaign analysis

Customer subscription prediction

Decision support systems for banking

## Conclusion

This project demonstrates a deployment-ready ANN-based solution for bank term deposit prediction.
Through systematic preprocessing, optimized neural architecture, and threshold tuning, the model achieves strong generalization and real-world applicability.


