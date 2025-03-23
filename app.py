import streamlit as st
import joblib
import numpy as np

# Load models
@st.cache_resource
def load_model(model_name):
    return joblib.load(model_name)

# Streamlit App
st.title("AI-Powered Disease Diagnosis System")
disease_option = st.sidebar.selectbox(
    "Choose Disease Prediction",
    ["Diabetes Prediction", "Heart Disease Prediction", "Lung Cancer Prediction", "Parkinson's Prediction"]
)

# ------------------- DIABETES PREDICTION -------------------
if disease_option == "Diabetes Prediction":
    st.header("Diabetes Prediction")
    st.markdown("Enter patient details (features from scikit-learn diabetes dataset):")
    
    age = st.number_input("Age", min_value=0, max_value=120, value=50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
    bp = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
    s1 = st.number_input("Total Cholesterol (s1)", min_value=0, max_value=500, value=150)
    s2 = st.number_input("LDL Cholesterol (s2)", min_value=0, max_value=500, value=100)
    s3 = st.number_input("HDL Cholesterol (s3)", min_value=0, max_value=200, value=50)
    s4 = st.number_input("Thyroid Stimulating Hormone (s4)", min_value=0.0, max_value=10.0, value=2.0)
    s5 = st.number_input("Lamotrigine (s5)", min_value=0.0, max_value=10.0, value=3.0)
    s6 = st.number_input("Blood Glucose (s6)", min_value=0, max_value=300, value=100)

    if st.button("Predict Diabetes"):
        # Preprocess sex (0=Male, 1=Female)
        sex = 1 if sex == "Female" else 0
        
        input_data = np.array([
            age, sex, bmi, bp, s1, s2, s3, s4, s5, s6
        ]).reshape(1, -1)
        
        model = load_model("diabetes_model.pkl")
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        st.write(f"**Prediction:** {'Diabetes' if prediction == 1 else 'No Diabetes'}")
        st.write(f"**Probability:** {probability:.2%}")

# ------------------- HEART DISEASE PREDICTION -------------------
elif disease_option == "Heart Disease Prediction":
    st.header("Heart Disease Prediction")
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    chest_pain = st.selectbox("Chest Pain Type (1-4)", [1, 2, 3, 4], index=0)
    resting_bp = st.number_input("Resting Blood Pressure", min_value=50, max_value=200, value=120)
    cholesterol = st.number_input("Cholesterol", min_value=50, max_value=600, value=200)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"])
    resting_ecg = st.selectbox("Resting ECG Results (0=Normal, 1=Abnormal)", [0, 1], index=0)
    max_hr = st.number_input("Max Heart Rate", min_value=50, max_value=200, value=150)
    exercise_angina = st.selectbox("Exercise-Induced Angina (0=No, 1=Yes)", [0, 1], index=0)
    oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0, value=1.0)
    st_slope = st.selectbox("ST Slope (0=Upsloping, 1=Flat, 2=Downsloping)", [0, 1, 2], index=0)

    if st.button("Predict Heart Disease"):
        # Preprocess categorical inputs
        sex = 1 if sex == "Male" else 0
        fasting_bs = 1 if fasting_bs == "Yes" else 0

        input_data = np.array([
            age, sex, chest_pain, resting_bp, cholesterol, fasting_bs,
            resting_ecg, max_hr, exercise_angina, oldpeak, st_slope
        ]).reshape(1, -1)
        
        model = load_model("heart_model_xgb.joblib")
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        st.write(f"**Prediction:** {'Heart Disease' if prediction == 1 else 'No Heart Disease'}")
        st.write(f"**Probability:** {probability:.2%}")

# ------------------- LUNG CANCER PREDICTION -------------------
elif disease_option == "Lung Cancer Prediction":
    st.header("Lung Cancer Prediction")
    yellow_fingers = st.selectbox("Yellow Fingers", ["No", "Yes"])
    anxiety = st.selectbox("Anxiety", ["No", "Yes"])
    peer_pressure = st.selectbox("Peer Pressure", ["No", "Yes"])
    chronic_disease = st.selectbox("Chronic Disease", ["No", "Yes"])
    fatigue = st.selectbox("Fatigue", ["No", "Yes"])
    allergy = st.selectbox("Allergy", ["No", "Yes"])
    wheezing = st.selectbox("Wheezing", ["No", "Yes"])
    alcohol_consuming = st.selectbox("Alcohol Consuming", ["No", "Yes"])
    coughing = st.selectbox("Coughing", ["No", "Yes"])
    swallowing_difficulty = st.selectbox("Swallowing Difficulty", ["No", "Yes"])
    chest_pain = st.selectbox("Chest Pain", ["No", "Yes"])
    anxyelfin = st.selectbox("ANXYELFIN (Unknown Feature)", ["No", "Yes"])

    if st.button("Predict Lung Cancer"):
        # Preprocess categorical inputs
        yellow_fingers = 1 if yellow_fingers == "Yes" else 0
        anxiety = 1 if anxiety == "Yes" else 0
        peer_pressure = 1 if peer_pressure == "Yes" else 0
        chronic_disease = 1 if chronic_disease == "Yes" else 0
        fatigue = 1 if fatigue == "Yes" else 0
        allergy = 1 if allergy == "Yes" else 0
        wheezing = 1 if wheezing == "Yes" else 0
        alcohol_consuming = 1 if alcohol_consuming == "Yes" else 0
        coughing = 1 if coughing == "Yes" else 0
        swallowing_difficulty = 1 if swallowing_difficulty == "Yes" else 0
        chest_pain = 1 if chest_pain == "Yes" else 0
        anxyelfin = 1 if anxyelfin == "Yes" else 0

        input_data = np.array([
            yellow_fingers, anxiety, peer_pressure, chronic_disease, fatigue,
            allergy, wheezing, alcohol_consuming, coughing, swallowing_difficulty,
            chest_pain, anxyelfin
        ]).reshape(1, -1)

        model = load_model("lung_cancer_model.pkl")
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        st.write(f"**Prediction:** {'Lung Cancer Detected' if prediction == 1 else 'No Lung Cancer'}")
        st.write(f"**Probability:** {probability:.2%}")

# ------------------- PARKINSON'S PREDICTION -------------------
elif disease_option == "Parkinson's Prediction":
    st.header("Parkinson's Disease Prediction")
    col1, col2 = st.columns(2)
    with col1:
        mdvp_fo = st.number_input("MDVP:Fo (Hz)", min_value=80.0, max_value=300.0, value=150.0)
        mdvp_fhi = st.number_input("MDVP:Fhi (Hz)", min_value=100.0, max_value=600.0, value=200.0)
        mdvp_flo = st.number_input("MDVP:Flo (Hz)", min_value=50.0, max_value=250.0, value=120.0)
        mdvp_jitter_percent = st.number_input("MDVP:Jitter (%)", min_value=0.0, max_value=5.0, value=0.5)
        mdvp_jitter_abs = st.number_input("MDVP:Jitter (Abs)", min_value=0.0, max_value=0.05, value=0.005)
        mdvp_rap = st.number_input("MDVP:RAP", min_value=0.0, max_value=0.1, value=0.02)
        mdvp_ppq = st.number_input("MDVP:PPQ", min_value=0.0, max_value=0.1, value=0.02)
        jitter_ddp = st.number_input("Jitter:DDP", min_value=0.0, max_value=0.3, value=0.05)
    
    with col2:
        mdvp_shimmer = st.number_input("MDVP:Shimmer", min_value=0.0, max_value=1.0, value=0.1)
        mdvp_shimmer_db = st.number_input("MDVP:Shimmer (dB)", min_value=0.0, max_value=10.0, value=2.0)
        shimmer_apq3 = st.number_input("Shimmer:APQ3", min_value=0.0, max_value=0.1, value=0.02)
        shimmer_apq5 = st.number_input("Shimmer:APQ5", min_value=0.0, max_value=0.1, value=0.02)
        mdvp_apq = st.number_input("MDVP:APQ", min_value=0.0, max_value=0.1, value=0.02)
        shimmer_dda = st.number_input("Shimmer:DDA", min_value=0.0, max_value=1.0, value=0.1)
        nhr = st.number_input("NHR", min_value=0.0, max_value=0.5, value=0.01)
        hnr = st.number_input("HNR", min_value=10.0, max_value=40.0, value=25.0)
        rpde = st.number_input("RPDE", min_value=0.0, max_value=1.0, value=0.5)
        dfa = st.number_input("DFA", min_value=0.5, max_value=1.5, value=0.8)
        spread1 = st.number_input("spread1", min_value=-10.0, max_value=0.0, value=-5.0)
        spread2 = st.number_input("spread2", min_value=0.0, max_value=1.0, value=0.2)
        d2 = st.number_input("D2", min_value=1.0, max_value=4.0, value=2.5)
        ppe = st.number_input("PPE", min_value=0.0, max_value=0.5, value=0.1)

    if st.button("Predict Parkinson's Disease"):
        input_data = np.array([
            mdvp_fo, mdvp_fhi, mdvp_flo, mdvp_jitter_percent, mdvp_jitter_abs,
            mdvp_rap, mdvp_ppq, jitter_ddp, mdvp_shimmer, mdvp_shimmer_db,
            shimmer_apq3, shimmer_apq5, mdvp_apq, shimmer_dda, nhr, hnr,
            rpde, dfa, spread1, spread2, d2, ppe
        ]).reshape(1, -1)
        
        model = load_model("parkinsons_model.pkl")
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        st.write(f"**Prediction:** {'Parkinson\'s Disease' if prediction == 1 else 'No Parkinson\'s Disease'}")
        st.write(f"**Probability:** {probability:.2%}")

else:
    st.warning("Model not yet implemented. Select another option.")