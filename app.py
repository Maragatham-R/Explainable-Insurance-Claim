import streamlit as st
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Explainable Insurance Claim Prediction",
    layout="wide"
)

# ---------------- TITLE ----------------
st.title("Explainable Insurance Claim Prediction")
st.write("Predict insurance claims using health and lifestyle factors with SHAP explanation")

# ---------------- LOAD MODEL ----------------
model = joblib.load("insurances.pkl")   # <-- Your new trained model

# ---------------- SIDEBAR ----------------
st.sidebar.header("User Input Parameters")

age = st.sidebar.slider("Age", 18, 100, 30)
bmi = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)

gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
smoker = st.sidebar.selectbox("Smoker", ["No", "Yes"])

disease = st.sidebar.selectbox("Disease History", ["No", "Yes"])
diabetes = st.sidebar.selectbox("Diabetes", ["No", "Yes"])

bloodpressure = st.sidebar.slider("Blood Pressure", 60, 200, 80)

regular_ex = st.sidebar.selectbox("Regular Exercise", ["No", "Yes"])

# ---------------- ENCODING ----------------
gender_val = 0 if gender == "Female" else 1
smoker_val = 0 if smoker == "No" else 1
disease_val = 0 if disease == "No" else 1
diabetes_val = 0 if diabetes == "No" else 1
regular_ex_val = 0 if regular_ex == "No" else 1

# ---------------- INPUT DATAFRAME ----------------
user_input = pd.DataFrame(
    [[
        age,
        bmi,
        gender_val,
        smoker_val,
        disease_val,
        bloodpressure,
        diabetes_val,
        regular_ex_val
    ]],
    columns=[
        "age",
        "bmi",
        "gender",
        "smoker",
        "Disease",
        "bloodpressure",
        "diabetes",
        "regular_ex"
    ]
)
shap_vals = np.array(shap_vals).flatten()

# -------- FEATURE INFLUENCE PIE CHART --------
st.subheader("Feature Influence Distribution")

# Use absolute SHAP values
abs_shap = np.abs(shap_vals)

# Convert to percentage
percentages = (abs_shap / abs_shap.sum()) * 100

# Take Top 6 features only (for clarity)
top_n = min(6, len(percentages))
top_idx = np.argsort(percentages)[-top_n:]

pie_labels = np.array(feature_names)[top_idx]
pie_sizes = percentages[top_idx]

fig3, ax3 = plt.subplots()

ax3.pie(
    pie_sizes,
    labels=pie_labels,
    autopct='%1.1f%%',
    startangle=90
)

ax3.axis('equal')
ax3.set_title("Feature Influence on Prediction")

st.pyplot(fig3)
