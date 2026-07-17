import streamlit as dt
import pickle
import numpy as np
import pandas as pd

# Set page configurations
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# App Title and Description
st.title("🎓 Student Performance Predictor")
st.markdown("""
This application uses a Machine Learning model (**K-Neighbors Regressor**) to predict a student's final score based on their daily habits and academic history.
""")
st.divider()

# Load the model safely
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except FileNotFoundError:
    st.error("⚠️ **model.pkl** file not found! Please ensure your model pickle file is named `model.pkl` and placed in the same directory as this script.")
    st.stop()

# Form Layout for Inputs
st.subheader("📊 Enter Student Metrics")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        hours_studied = st.number_input(
            "Hours Studied", 
            min_value=0.0, 
            max_value=24.0, 
            value=5.0, 
            step=0.5,
            help="Average number of hours spent studying per day."
        )
        
        attendance_percent = st.slider(
            "Attendance Percentage (%)", 
            min_value=0.0, 
            max_value=100.0, 
            value=85.0, 
            step=1.0,
            help="Overall class attendance rate."
        )

    with col2:
        sleep_hours = st.number_input(
            "Sleep Hours", 
            min_value=0.0, 
            max_value=24.0, 
            value=7.0, 
            step=0.5,
            help="Average daily hours of sleep."
        )
        
        previous_scores = st.number_input(
            "Previous Exam Score", 
            min_value=0.0, 
            max_value=100.0, 
            value=75.0, 
            step=1.0,
            help="Score achieved in the immediate past evaluation."
        )

    # Submit button
    submit_button = st.form_submit_button(label="🔮 Predict Score")

# Handle prediction logic
if submit_button:
    # Organize input exactly match the model's feature names order
    input_data = pd.DataFrame([{
        'hours_studied': hours_studied,
        'sleep_hours': sleep_hours,
        'attendance_percent': attendance_percent,
        'previous_scores': previous_scores
    }])
    
    # Generate prediction
    try:
        prediction = model.predict(input_data)[0]
        
        # Display the result with a nice layout
        st.success("🎉 Prediction Ready!")
        
        # Metric layout display
        st.metric(
            label="Estimated Final Score", 
            value=f"{prediction:.2f}"
        )
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
