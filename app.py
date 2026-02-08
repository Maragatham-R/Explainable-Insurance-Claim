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

# ---------------- PREDICTION ----------------
if st.sidebar.button("Predict Claim"):

    prediction = model.predict(user_input)[0]
    probability = model.predict_proba(user_input)[0][1]

    # Result
    if prediction == 1:
        st.error(f"Claim Likely (Probability: {probability:.2f})")
    else:
        st.success(f"No Claim (Probability: {probability:.2f})")

    # ---------------- SHAP ----------------
    st.subheader("SHAP Explanation")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(user_input)

    # Handle binary classifier
    if isinstance(shap_values, list):
        shap_vals = shap_values[1][0]
    else:
        shap_vals = shap_values[0]

    shap_vals = np.array(shap_vals).flatten()

    # Feature names
    if hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)
    else:
        feature_names = user_input.columns.tolist()

    # Force same length
    min_len = min(len(feature_names), len(shap_vals))
    shap_vals = shap_vals[:min_len]
    feature_names = feature_names[:min_len]

    # Sort by importance
    order = np.argsort(np.abs(shap_vals))

    TOP_K = min(8, len(order))
    order = order[-TOP_K:]

    # ---------------- PLOT ----------------
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.barh(
        np.array(feature_names)[order],
        shap_vals[order]
    )

    ax.set_xlabel("SHAP Value (Impact on Prediction)")
    ax.set_title("Top Feature Contributions")

    st.pyplot(fig)

    # ---------------- INFO ----------------
    st.markdown("""
    ### 🔍 How to Read This SHAP Chart

    - **Positive Value (+)** → Increases claim risk  
    - **Negative Value (–)** → Reduces claim risk  

    ### 📊 Interpretation
    - Longer bars = More influence
    - Health conditions and habits strongly affect claim probability
    """)

    st.info(
        "This prediction is based on health, lifestyle, and medical history. "
        "Higher probability means higher risk of insurance claim."
    )

