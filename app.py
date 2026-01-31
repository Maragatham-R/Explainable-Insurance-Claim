import streamlit as st
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import numpy as np

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Insurance Claim Prediction",
    layout="wide"
)

st.title("Insurance Claim Prediction with SHAP")

# ---------------- Load Model & Encoder ----------------
model = joblib.load("insurance2.pkl")

# ---------------- Sidebar Inputs ----------------
st.sidebar.header("User Inputs")

age = st.sidebar.slider("Age", 1, 100, 30)
sex = st.sidebar.selectbox("Sex", ["male", "female"])
weight = st.sidebar.slider("Weight (kg)", 30, 150, 60)
bmi = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)
hereditary = st.sidebar.selectbox(
    "Hereditary Disease",
    ["NoDisease", "Epilepsy", "Diabetes", "HeartDisease", "Cancer"]
)
no_of_dependents = st.sidebar.number_input("No. of Dependents", 0, 10, 0)
smoker = st.sidebar.selectbox("Smoker", [0, 1])
city = st.sidebar.selectbox(
    "City",
    ["NewYork", "Boston", "Phildelphia", "Pittsburg", "Buffalo"]
)
bloodpressure = st.sidebar.slider("Blood Pressure", 50, 150, 80)
diabetes = st.sidebar.selectbox("Diabetes", [0, 1])
regular_ex = st.sidebar.selectbox("Regular Exercise", [0, 1])
job_title = st.sidebar.selectbox(
    "Job Title",
    ["Actor", "Engineer", "Academician", "Chef", "HomeMakers"]
)

# ---------------- Input Data (RAW - No Encoding Here) ----------------
user_input = pd.DataFrame(
    [[
        age,
        sex,
        weight,
        bmi,
        hereditary,
        no_of_dependents,
        smoker,
        city,
        bloodpressure,
        diabetes,
        regular_ex,
        job_title
    ]],
    columns=[
        "age",
        "sex",
        "weight",
        "bmi",
        "hereditary_diseases",
        "no_of_dependents",
        "smoker",
        "city",
        "bloodpressure",
        "diabetes",
        "regular_ex",
        "job_title"
    ]
)

# ---------------- Prediction ----------------
if st.sidebar.button("Predict"):

    # Encode input
    X_enc = encoder.transform(user_input)

    # Predict
    prediction = model.predict(X_enc)[0]
    probability = model.predict_proba(X_enc)[0][1]

    if prediction == 1:
        st.error(f"Claim Likely (Probability: {probability:.2f})")
    else:
        st.success(f"No Claim (Probability: {probability:.2f})")

    # ---------------- SHAP ----------------
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_enc)

    # Handle classifier output
    if isinstance(shap_values, list):
        shap_vals = shap_values[1][0]
    else:
        shap_vals = shap_values[0]

    shap_vals = np.array(shap_vals).flatten()

    # Feature Names from Encoder
    if hasattr(encoder, "get_feature_names_out"):
        feature_names = encoder.get_feature_names_out()
    else:
        feature_names = [f"Feature {i}" for i in range(len(shap_vals))]

    # Ensure same length
    min_len = min(len(feature_names), len(shap_vals))
    shap_vals = shap_vals[:min_len]
    feature_names = feature_names[:min_len]

    # Absolute values for pie
    shap_abs = np.abs(shap_vals)

    # Top Features
    TOP_K = min(6, len(shap_abs))
    order = np.argsort(shap_abs)[-TOP_K:]

    top_vals = shap_abs[order]
    top_names = np.array(feature_names)[order]

    # ---------------- Pie Chart ----------------
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.pie(
        top_vals,
        labels=top_names,
        autopct="%1.1f%%",
        startangle=90
    )

    ax.set_title("Top Feature Contribution (SHAP)")
    ax.axis("equal")

    st.pyplot(fig)
