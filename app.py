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

try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(user_input)

        # For classifier
        if isinstance(shap_values, list):
            shap_vals = shap_values[1][0]
        else:
            shap_vals = shap_values[0]

        shap_vals = np.array(shap_vals).flatten()

        # Absolute values
        abs_shap = np.abs(shap_vals)

        # Convert to percentage
        percent = (abs_shap / abs_shap.sum()) * 100

        # -----------------------------
        # Pie Chart
        # -----------------------------

        fig, ax = plt.subplots(figsize=(5,5))

        ax.pie(
            percent,
            labels=features,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize':10}
        )

        ax.set_title("Feature Influence on Prediction", fontsize=12)

        st.pyplot(fig)

    except Exception as e:
        st.error("SHAP calculation failed")
        st.write(e)
