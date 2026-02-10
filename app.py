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

BP = st.sidebar.slider("Blood Pressure", 60, 200, 80)

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
        BP,
        diabetes_val,
        regular_ex_val
    ]],
    columns=[
        "age",
        "bmi",
        "gender",
        "smoker",
        "Disease",
        "BP",
        "diabetes",
        "regular_ex"
    ]
)

# ---------------- SHAP ----------------
st.subheader("SHAP Explanation")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(user_input)

# Handle binary classifier output
if isinstance(shap_values, list):
    shap_vals = shap_values[1][0]
else:
    shap_vals = shap_values[0]

# Convert to numpy
shap_vals = np.array(shap_vals).flatten()

# Feature names
if hasattr(model, "feature_names_in_"):
    feature_names = list(model.feature_names_in_)
else:
    feature_names = user_input.columns.tolist()

# Match lengths
min_len = min(len(feature_names), len(shap_vals))
shap_vals = shap_vals[:min_len]
feature_names = feature_names[:min_len]

# ---------------- FEATURE INFLUENCE PIE ----------------
st.subheader("Feature Influence Distribution")

abs_shap = np.abs(shap_vals)

percentages = (abs_shap / abs_shap.sum()) * 100

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

# ---------------- BAR CHART ----------------
order = np.argsort(np.abs(shap_vals))
TOP_K = min(8, len(order))
order = order[-TOP_K:]

fig, ax = plt.subplots(figsize=(9, 5))

ax.barh(
    np.array(feature_names)[order],
    shap_vals[order]
)

ax.set_xlabel("SHAP Value (Impact)")
ax.set_title("Top Feature Contributions")

st.pyplot(fig)

st.info(
    "Positive SHAP increases claim risk. Negative SHAP reduces claim risk."
)

