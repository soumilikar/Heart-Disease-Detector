import streamlit as st
import requests

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

st.title("Heart Disease Prediction")
st.write("Enter the patient's details below to predict the possibility of heart disease.")

st.divider()


st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=50,
        step=1
    )

    sex = st.selectbox(
        "Sex",
        ["M", "F"]
    )

    chest_pain_type = st.selectbox(
    "Chest Pain Type",
    options=["TA", "ATA", "NAP", "ASY"],
    format_func=lambda x: {
        "TA": "TA - Typical Angina",
        "ATA": "ATA - Atypical Angina",
        "NAP": "NAP - Non-Anginal Pain",
        "ASY": "ASY - Asymptomatic"
    }[x]
)

    resting_bp = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        min_value=0,
        max_value=300,
        value=120,
        step=1
    )

    cholesterol = st.number_input(
        "Cholesterol (mg/dl)",
        min_value=0,
        max_value=700,
        value=200,
        step=1
    )

with col2:
    fasting_bs = st.selectbox(
        "Fasting Blood Sugar",
        [0, 1],
        format_func=lambda x: "No (0)" if x == 0 else "Yes (1)"
    )

    resting_ecg = st.selectbox(
    "Resting ECG",
    options=["Normal", "ST", "LVH"],
    format_func=lambda x: {
        "Normal": "Normal - Normal ECG",
        "ST": "ST - ST-T Wave Abnormality",
        "LVH": "LVH - Left Ventricular Hypertrophy"
    }[x])

    max_hr = st.number_input(
        "Maximum Heart Rate",
        min_value=50,
        max_value=250,
        value=150,
        step=1
    )

    exercise_angina = st.selectbox(
    "Exercise-Induced Angina",
    options=["N", "Y"],
    format_func=lambda x: {
        "N": "N - No",
        "Y": "Y - Yes"
    }[x])


    oldpeak = st.number_input(
        "Oldpeak",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )

st_slope = st.selectbox(
    "ST Slope",
    options=["Up", "Flat", "Down"],
    format_func=lambda x: {
        "Up": "Up - Upsloping",
        "Flat": "Flat - Flat",
        "Down": "Down - Downsloping"
    }[x])

st.divider()

if st.button("Predict", type="primary", use_container_width=True):

    input_data = {
        "Age": age,
        "Sex": sex,
        "ChestPainType": chest_pain_type,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "RestingECG": resting_ecg,
        "MaxHR": max_hr,
        "ExerciseAngina": exercise_angina,
        "Oldpeak": oldpeak,
        "ST_Slope": st_slope
    }

    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=input_data
        )

        if response.status_code == 200:

            result = response.json()

            st.subheader("Prediction Result")

            if result["prediction"] == 1:
                st.error("⚠️ Prediction: Heart Disease")
            else:
                st.success("✅ Prediction: Normal")

        else:
            st.error(
                f"API Error: {response.status_code}\n\n{response.text}"
            )

    except requests.exceptions.ConnectionError:
        st.error(
            "Could not connect to the FastAPI server. "
            "Make sure your API is running."
        )