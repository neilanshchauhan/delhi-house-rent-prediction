# streamlit_app/app.py

import streamlit as st
import pandas as pd
import requests
import os
import logging
import socket
from datetime import datetime
from prometheus_fastapi_instrumentator import Instrumentator
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get API URL from environment variable
API_URL = os.getenv("API_URL", "http://model:8000")

Instrumentator().instrument(app).expose(app)

# Page configuration
st.set_page_config(page_title="Delhi House Rent Predictor", page_icon="🏠", layout="centered")

# Custom CSS
st.markdown("""<style>.main { padding-top: 2rem; } .stButton>button { width: 100%; }</style>""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load data for dropdowns"""
    try:
        df = pd.read_csv("data/processed/final_data.csv")
        return df
    except FileNotFoundError:
        st.error("Data file for dropdowns not found. Please check the Dockerfile COPY command.")
        return None

def check_backend_health():
    """Check if the FastAPI backend is available"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def predict_rent_api(size_sq_ft, bedrooms, locality, property_type):
    """Make rent prediction by calling FastAPI endpoint"""
    predict_url = f"{API_URL.rstrip('/')}/predict"
    logger.info(f"Connecting to API at: {predict_url}")
    
    # --- CORRECTED: Explicitly cast NumPy types to standard Python types ---
    payload = {
        "size_sq_ft": float(size_sq_ft),
        "bedrooms": int(bedrooms),
        "localityName": locality,
        "propertyType": property_type
    }
    
    try:
        response = requests.post(predict_url, json=payload, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        
        result = response.json()
        logger.info(f"Prediction received: {result}")
        
        st.session_state.prediction = {
            'predicted_rent': result.get('predicted_rent'),
            'lower_bound': result.get('confidence_interval', [0,0])[0],
            'upper_bound': result.get('confidence_interval', [0,0])[1],
            'prediction_time': result.get('prediction_time', datetime.now().isoformat())
        }
        st.session_state.error = None

    except requests.exceptions.RequestException as e:
        error_message = f"Error connecting to API: {e}"
        logger.error(error_message)
        st.session_state.error = error_message
        st.session_state.prediction = None

def main():
    st.title("🏠 Delhi House Rent Predictor")
    
    df = load_data()
    if df is None: return

    localities = sorted(df['localityName'].dropna().unique())
    property_types = sorted(df['propertyType'].dropna().unique())
    bedroom_options = sorted(df['bedrooms'].dropna().unique().astype(int))
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            locality = st.selectbox("Select Locality", options=localities)
            property_type = st.selectbox("Property Type", options=property_types)
        with col2:
            bedrooms = st.selectbox("Number of Bedrooms (BHK)", options=bedroom_options)
            size_sq_ft = st.number_input("Property Size (sq. ft.)", min_value=100, value=1000)
        
        submit_button = st.form_submit_button("🔮 Predict Rent")
    
    if submit_button:
        with st.spinner("Calculating..."):
            predict_rent_api(size_sq_ft, bedrooms, locality, property_type)

    if st.session_state.get('error'):
        st.error(f"Prediction Failed: {st.session_state.error}")

    if st.session_state.get('prediction'):
        result = st.session_state.prediction
        st.markdown("---")
        st.markdown("### 📊 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Lower Estimate", f"₹{result['lower_bound']:,.0f}")
        col2.metric("**Predicted Rent**", f"₹{result['predicted_rent']:,.0f}")
        col3.metric("Upper Estimate", f"₹{result['upper_bound']:,.0f}")
        
        st.success(f"Estimated monthly rent for a {bedrooms} BHK {property_type.lower()} in {locality} is **₹{result['predicted_rent']:,.0f}**.")

    st.markdown("<hr>", unsafe_allow_html=True)
    try:
        version = os.getenv("APP_VERSION", "2 .0.0")
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        st.markdown(
            f"""
            <div style="text-align: center; color: gray;">
                <p>Version: {version} | Hostname: {hostname} ({ip_address})</p>
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        logger.warning(f"Could not fetch host details: {e}")

if __name__ == "__main__":
    main()