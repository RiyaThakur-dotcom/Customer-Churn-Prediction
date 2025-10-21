🧠 Customer Churn Prediction – Machine Learning Project
Predicting telecom customer churn using Python, Scikit-learn, and XGBoost — complete ML pipeline from preprocessing to prediction.

📁 Project Structure
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── customer_churn_model.pkl
├── encoders.pkl
├── churn_prediction.ipynb
└── README.md

🎯 Objective
To predict whether a customer will churn (leave) based on usage patterns and account details.

🧰 Tech Stack
• Python
• Pandas, NumPy
• Matplotlib, Seaborn
• Scikit-learn
• XGBoost
• Imbalanced-learn (SMOTE)

Pickle

📊 Workflow
1️⃣ Data Loading & Cleaning
• Replaced blank TotalCharges values with 0
• Converted data types appropriately
• Removed unnecessary customerID column

2️⃣ Exploratory Data Analysis
• Visualized distributions and relationships
• Checked correlations among numeric features

3️⃣ Preprocessing
• Label Encoding for categorical variables
• SMOTE for handling class imbalance

4️⃣ Model Training
• Trained and evaluated multiple models:
• Decision Tree
• Random Forest
• XGBoost

Used metrics:
• Accuracy
• Confusion Matrix
• Classification Report

5️⃣ Model Saving
with open("customer_churn_model.pkl", "wb") as f:
    pickle.dump({"model": trained_model, "features_names": feature_names}, f)

Encoders saved as:
with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

6️⃣ Prediction Example
import pickle
import pandas as pd

# Load model and encoders
with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)
model = model_data["model"]

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Input sample
input_data = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85
}

df = pd.DataFrame([input_data])
for col, enc in encoders.items():
    if col in df.columns:
        df[col] = enc.transform(df[col])

pred = model.predict(df)[0]
print("Prediction:", "Churn" if pred == 1 else "No Churn")

📈 Results
  • Best Model: Random Forest / XGBoost
  • Accuracy: ~80–85% (depending on tuning)

🌟 Insights
   • Month-to-month contracts, electronic checks, and no internet security → higher churn
   • Long-term contracts and automatic payments → lower churn

📫 Connect:
📍 Author: Riya Thakur
🔗 LinkedIn Profile: https://www.linkedin.com/in/riya-thakur-876571378
💻 GitHub Repository: https://github.com/RiyaThakur-dotcom/Customer-Churn-Prediction
