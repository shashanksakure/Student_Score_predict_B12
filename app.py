import streamlit as st  # <-- Fixed typo here!
import pickle
import numpy as np
import pandas as pd

# 1. Page Configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# 2. Custom CSS (Color Palette, Cards, Hover Transitions, & Shadow Effects)
st.markdown("""
    <style>
    /* Global Background Color */
    .stApp {
        background-color: #F8FAFC;
    }
    
    /* Title Styling */
    .title-text {
        font-family: 'Inter', sans-serif;
        color: #0F172A;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
    }
    .subtitle-text {
        color: #64748B;
        text-align: center;
        font-size: 16px;
        margin-bottom: 30px;
    }

    /* Modern Card Layout with Soft Shadows */
    .custom-card {
        background-color: #FFFFFF;
        padding: 28px;
        border-radius: 16px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.05);
        border: 1px solid #E2E8F0;
        margin-bottom: 25px;
    }
    
    /* Highlight Top Border for Action Card */
    .accent-card {
        border-top: 5px solid #3B82F6;
    }
    
    /* Result Card Styling */
    .result-card {
        background-color: #EFF6FF;
        padding: 24px;
        border-radius: 16px;
        border: 1px solid #BFDBFE;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.1);
        text-align: center;
        animation: fadeIn 0.5s ease-out;
    }
    
    /* Typography & Labels */
    label {
        font-weight: 600 !important;
        color: #334155 !important;
        font-size: 14px !important;
    }

    /* Customizing the Streamlit Button */
    div.stButton > button:first-child {
        background-color: #3B82F6;
        color: #FFFFFF;
        border: none;
        padding: 12px 24px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 16px;
        width: 100%;
        box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.2), 0 2px 4px -1px rgba(59, 130, 246, 0.16);
        transition: all 0.2s ease-in-out;
    }
    
    div.stButton > button:first-child:hover {
        background-color: #2563EB;
        box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.3);
        transform: translateY(-1px);
    }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
""", unsafe_allow_html=True)

# 3. App Title & Subtitle
st.markdown("<h1 class='title-text'>🎓 Student Performance Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-text'>Input academic metrics below to generate a predicted final grade using our KNN model.</p>", unsafe_allow_html=True)

# 4. Safely Load Model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except FileNotFoundError:
    st.error("⚠️ **model.pkl** file not found! Please place it in the same directory as this app.")
    st.stop()

# 5. Clean, Styled Vertical Input Layout
st.markdown("<div class='custom-card accent-card'>", unsafe_allow_html=True)
st.subheader("📝 Enter Metrics")

# Input fields mapped directly to the model's features[cite: 1]
hours_studied = st.number_input(
    "📚 Hours Studied", 
    min_value=0.0, max_value=24.0, value=5.0, step=0.5,
    help="Average daily hours spent studying."
)

sleep_hours = st.number_input(
    "😴 Sleep Hours", 
    min_value=0.0, max_value=24.0, value=7.0, step=0.5,
    help="Average daily hours of sleep."
)

attendance_percent = st.slider(
    "🏫 Attendance Percentage", 
    min_value=0.0, max_value=100.0, value=85.0, step=1.0,
    help="Overall class attendance rate."
)

previous_scores = st.number_input(
    "🏆 Previous Exam Score", 
    min_value=0.0, max_value=100.0, value=75.0, step=1.0,
    help="Latest exam grade."
)

st.markdown("</div>", unsafe_allow_html=True)

# Predict Button
predict_clicked = st.button("🔮 Calculate Predicted Score")

# 6. Result Card Section
if predict_clicked:
    # Build dataframe matching expected model feature names[cite: 1]
    input_df = pd.DataFrame([{
        'hours_studied': hours_studied,
        'sleep_hours': sleep_hours,
        'attendance_percent': attendance_percent,
        'previous_scores': previous_scores
    }])
    
    try:
        prediction = model.predict(input_df)[0]
        
        # Display styled output card
        st.markdown(f"""
            <div class='result-card'>
                <p style='color: #1E40AF; font-size: 14px; font-weight: 600; margin: 0; text-transform: uppercase; letter-spacing: 0.05em;'>Prediction Ready</p>
                <h2 style='color: #1E3A8A; font-size: 42px; font-weight: 800; margin: 10px 0 5px 0;'>{prediction:.2f}</h2>
                <p style='color: #60A5FA; font-size: 14px; font-weight: 500; margin: 0;'>Estimated Final Performance Grade</p>
            </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Prediction Error: {e}")
