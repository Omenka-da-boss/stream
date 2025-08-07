import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pickle
from notebook.okay import scaler
from notebook.park import scalers




diab_model = pickle.load(open('./saved_models/diabetes.wsv','rb'))

heart_model = pickle.load(open('./saved_models/heart.wsv','rb'))

parks_model = pickle.load(open('./saved_models/park.wsv','rb'))




with st.sidebar:
    selected = option_menu(
        "Multiple Disesease Prediction System",
        ["Diabetes Prediction",
        "Heart Disease Prediction",
        "Parkinsons Prediction"],
        icons=['activity','heart','person lines fill']
        ,default_index=0
    )

def diab_func(sample):
    sample_array = np.asarray(sample)
    sample_reshaped = sample_array.reshape(1,-1)
    standard = scaler.transform(sample_reshaped)
    prediction = diab_model.predict(standard)
    return prediction


def heart_func(sample):
    sample_array = np.asarray(sample)

    sample_array_reshaped = sample_array.reshape(1,-1)

    prediction = heart_model.predict(sample_array_reshaped)

    return prediction


def park_func(sample):
    sample_array = np.asarray(sample)

    sample_reshaped = sample_array.reshape(1,-1)

    std_data = scalers.transform(sample_reshaped)

    prediction = parks_model.predict(std_data)

    return prediction

if selected == "Diabetes Prediction":
    st.title("Diabetes Prediction")
    col1,col2,col3 = st.columns(3)
    with col1:
        Pregnancies = st.number_input("Enter times of Pregnancies")
    with col2:
        Glucose = st.number_input("Enter Glucose level")
    with col3:
        BloodPressure = st.number_input("Enter your BloodPressure")
    with col1:
        SkinThickness = st.number_input("Enter SkinThickness Level")
    with col2:
        Insulin = st.number_input("Enter Insulin Level")
    with col3:
        BMI = st.number_input("Enter BMI rating")
    with col1:
        DiabetesPedigreeFunction = st.number_input("Enter your DiabetesPedigreeFunction") 
    with col2:
        Age = st.number_input("Enter Current Age")

    diab_diagonsis = ""
    
    if st.button("Diabetes Test Result"):
        prediction = diab_func([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
        BMI, DiabetesPedigreeFunction, Age]])
        
        if (prediction == 0):
            st.balloons()
            diab_diagonsis = "This patient does not have diabetes"
        else:
            diab_diagonsis = "This pateint is showing signs of diabetes please check"
    st.success(diab_diagonsis)


if selected == "Heart Disease Prediction":
    st.title("Heart Disease Prediction")
    col1,col2,col3 = st.columns(3)
    with col1:
        age = st.number_input("Enter times of age")
    with col2:
        sex = st.number_input("Enter your gender")
    with col3:
        cp = st.number_input("Enter your cp")
    with col1:
        trtbps = st.number_input("Enter your trtbps mark")
    with col2:
        chol = st.number_input("Enter Cholesterol Level")
    with col3:
        fbs = st.number_input("Enter fbs rating")
    with col1:
        restecg = st.number_input("Enter your restecg level") 
    with col2:
        thalachh = st.number_input("Enter your thalachh level")
    with col3:
        exng = st.number_input("Enter exng rating")
    with col1:
        oldpeak = st.number_input("Enter your oldpeak level") 
    with col2:
        slp = st.number_input("Enter your slp level")
    with col3:
        caa = st.number_input("Enter caa rating")
    with col1:
        thall = st.number_input("Enter your thall level") 
    
    heart_diagonsis = ""
    
    if st.button("Heart Disease Test Result"):
        prediction = heart_func([[age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall]])
        
        if prediction == 0:
            st.balloons()
            heart_diagonsis = "This patient does not have a heart disease"
        else:
            heart_diagonsis = "This pateint is showing signs of heart diseases please check"
    st.success(heart_diagonsis)
    
if selected == "Parkinsons Prediction":
    st.title("Parkinsons Prediction")

    col1,col2,col3 = st.columns(3)
    
    with col1:
        MDVP_fo_hz = st.number_input("Enter your MDVP:Fo(Hz)")
    with col2:
        MDVP_fhi_hz = st.number_input("Enter your MDVP:Fhi(Hz)")
    with col3:
        MDVP_flo_hz = st.number_input("Enter your MDVP:Flo(Hz)")
    with col1:
        MDVP_jitter_per = st.number_input("Enter your MDVP:Jitter(%)")
    with col2:
        MDVP_jitter_abs = st.number_input("Enter your MDVP:Jitter(Abs)")
    with col3:
        MDVP_rap = st.number_input("Enter your MDVP:RAP")
    with col1:
        MDVP_ppq = st.number_input("Enter your MDVP:PPQ")
    with col2:
        jitter_ddp = st.number_input("Enter your Jitter:DDP")
    with col3:
        mdvp_shimmer = st.number_input("Enter your MDVP:Shimmer") 
    with col1:
        mdvp_shimmer_db = st.number_input("Enter your MDVP:Shimmer(dB)")
    with col2:
        shimmer_apq3 = st.number_input("Enter your Shimmer:APQ3")
    with col3:
        shimmer_apq5 = st.number_input("Enter your Shimmer:APQ5")
    with col1:
        mdvp_apq = st.number_input("Enter your MDVP:APQ") 
    with col2:
        shimmer_dda = st.number_input("Enter your Shimmer:DDA")
    with col3:
        nhr = st.number_input("Enter your NHR") 
    with col1:
        hnr = st.number_input("Enter your HNR")   
    with col2:
        rpde = st.number_input("Enter your RPDE")
    with col3:
        dfa = st.number_input("Enter your DFA")
    with col1:
        spread1 = st.number_input("Enter your spread1") 
    with col2:
        spread2 = st.number_input("Enter your spread2")
    with col3:
        d2 = st.number_input("Enter your D2")
    with col1:
        ppe = st.number_input("Enter your PPE") 

    park_diagnosis = ""
    
    if st.button("Parkinsons Test Result"):
        prediction = park_func([[MDVP_fo_hz, MDVP_fhi_hz, MDVP_flo_hz, MDVP_jitter_per,
        MDVP_jitter_abs, MDVP_rap, MDVP_ppq, jitter_ddp, mdvp_shimmer, mdvp_shimmer_db,
        shimmer_apq3,shimmer_apq5, mdvp_apq, shimmer_dda, nhr, hnr, rpde, dfa, spread1,
        spread2, d2, ppe]])

        if prediction == 0:
            st.balloons()
            park_diagnosis = "This patient does not have Parkinsons Disease"
        else:
            park_diagnosis = "This pateint is showing signs of Parkinsons Disease please check"
    st.success(park_diagnosis)
    
   