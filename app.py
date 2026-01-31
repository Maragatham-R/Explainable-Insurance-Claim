import streamlit as st
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import numpy as np


# -------- Page Config --------
st.set_page_config(
    page_title="Insurance Claim Prediction",
    layout="wide"
)

st.title("Insurance Claim Prediction System")


# -------- Load Model --------
model = joblib.load("insurance3.pkl")


# -------- Sidebar Inputs --------
st.sidebar.header("User Inputs")

age = st.sidebar.slider("Age", 1, 100, 30)
sex = st.sidebar.selectbox("Sex", ["male", "female"])

weight = st.sidebar.slider("Weight (kg)", 30, 150, 60)

bmi = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)

hereditary = st.sidebar.selectbox(
    "Hereditary Disease",
    ["NoDisease", "Epilepsy", "Diabetes", "HeartDisease", "Cancer"]
)

dependents = st.sidebar.number_input("Dependents", 0, 10, 0)

smoker = st.sidebar.selectbox("Smoker", [0, 1])

city = st.sidebar.selectbox(
    "City",
    ["NewYork", "Boston", "Phildelphia", "Pittsburg", "Buffalo"]
)

bp = st.sidebar.slider("Blood Pressure", 50, 150, 80)

diabetes = st.sidebar.selectbox("Diabetes", [0, 1])

exercise = st.sidebar.selectbox("Regular Exercise", [0, 1])

job = st.sidebar.selectbox(
    "Job Title",
    ["Actor", "Engineer", "Academician", "Chef", "HomeMakers"]
)


# -------- Input Data --------
user_input = pd.DataFrame(
    [[
        age, sex, weight, bmi, hereditary,
        dependents, smoker, city, bp,
        diabetes, exercise, job
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


# -------- Prediction --------
if st.sidebar.button("Predict"):

    pred = model.predict(user_input)[0]
    prob = model.predict_proba(user_input)[0][1]


    if pred == 1:
        st.error(f"Claim Likely (Probability: {prob:.2f})")
    else:
        st.success(f"No Claim (Probability: {prob:.2f})")


    # -------- SHAP --------
    rf_model = model.named_steps["rf"]
    preprocessor = model.named_steps["preprocess"]

    X_trans = preprocessor.transform(user_input)

    explainer = shap.TreeExplainer(rf_model)

    shap_values = explainer.shap_values(X_trans)


    if isinstance(shap_values, list):
        vals = shap_values[1][0]
    else:
        vals = shap_values[0]


    vals = np.abs(vals)

    feature_names = preprocessor.get_feature_names_out()


    # Top 6
    order = np.argsort(vals)[-6:]


    fig, ax = plt.subplots()

    ax.pie(
        vals[order],
        labels=np.array(feature_names)[order],
        autopct="%1.1f%%"
    )

    ax.set_title("Top Feature Impact")

    st.pyplot(fig)
