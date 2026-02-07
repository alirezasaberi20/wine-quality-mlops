"""Streamlit UI for Wine Quality Prediction."""

import streamlit as st
import requests
import os

# Configuration
# Try to get API URL from environment, fallback to localhost
API_URL = os.getenv("API_URL", "http://localhost:8000")

# If running on same host, also try the docker bridge network
import socket
def get_api_url():
    env_url = os.getenv("API_URL")
    if env_url:
        return env_url
    # Try localhost first
    return "http://localhost:8000"

API_URL = get_api_url()

# Page configuration
st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #722F37;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-good {
        background-color: #d4edda;
        border: 2px solid #28a745;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .prediction-bad {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


def check_api_health():
    """Check if API is healthy."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def get_prediction(features: dict) -> dict:
    """Get prediction from API."""
    try:
        response = requests.post(f"{API_URL}/predict", json=features, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<p class="main-header">🍷 Wine Quality Predictor</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar - API Status and Info
    with st.sidebar:
        st.header("📊 System Status")
        
        api_healthy = check_api_health()
        if api_healthy:
            st.success("✅ API Connected")
        else:
            st.error("❌ API Disconnected")
            st.info(f"API URL: {API_URL}")
        
        st.markdown("---")
        st.header("ℹ️ About")
        st.markdown("""
        This application predicts wine quality based on 
        physicochemical properties using a Random Forest classifier.
        
        **Quality Classification:**
        - 🟢 **Good**: Score ≥ 6
        - 🔴 **Bad**: Score < 6
        
        **Model Performance:**
        - Accuracy: ~80%
        - F1 Score: ~80%
        """)
        
        st.markdown("---")
        st.header("🔗 Links")
        st.markdown(f"[API Documentation]({API_URL}/docs)")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🧪 Enter Wine Properties")
        
        # Create input form with two columns
        input_col1, input_col2 = st.columns(2)
        
        with input_col1:
            fixed_acidity = st.number_input(
                "Fixed Acidity (g/dm³)", 
                min_value=0.0, max_value=20.0, value=7.4, step=0.1,
                help="Most acids involved with wine (tartaric acid)"
            )
            volatile_acidity = st.number_input(
                "Volatile Acidity (g/dm³)", 
                min_value=0.0, max_value=2.0, value=0.7, step=0.01,
                help="Amount of acetic acid in wine"
            )
            citric_acid = st.number_input(
                "Citric Acid (g/dm³)", 
                min_value=0.0, max_value=2.0, value=0.0, step=0.01,
                help="Found in small quantities, adds freshness"
            )
            residual_sugar = st.number_input(
                "Residual Sugar (g/dm³)", 
                min_value=0.0, max_value=20.0, value=1.9, step=0.1,
                help="Sugar remaining after fermentation"
            )
            chlorides = st.number_input(
                "Chlorides (g/dm³)", 
                min_value=0.0, max_value=1.0, value=0.076, step=0.001,
                help="Amount of salt in the wine"
            )
            free_sulfur_dioxide = st.number_input(
                "Free Sulfur Dioxide (mg/dm³)", 
                min_value=0.0, max_value=100.0, value=11.0, step=1.0,
                help="Free form of SO2 that prevents microbial growth"
            )
        
        with input_col2:
            total_sulfur_dioxide = st.number_input(
                "Total Sulfur Dioxide (mg/dm³)", 
                min_value=0.0, max_value=300.0, value=34.0, step=1.0,
                help="Total amount of SO2 (free + bound forms)"
            )
            density = st.number_input(
                "Density (g/cm³)", 
                min_value=0.9, max_value=1.1, value=0.9978, step=0.0001,
                format="%.4f",
                help="Density of wine"
            )
            pH = st.number_input(
                "pH", 
                min_value=2.0, max_value=5.0, value=3.51, step=0.01,
                help="Describes acidity on scale of 0-14"
            )
            sulphates = st.number_input(
                "Sulphates (g/dm³)", 
                min_value=0.0, max_value=3.0, value=0.56, step=0.01,
                help="Wine additive contributing to SO2 levels"
            )
            alcohol = st.number_input(
                "Alcohol (%)", 
                min_value=5.0, max_value=20.0, value=9.4, step=0.1,
                help="Percent alcohol content"
            )
        
        # Predict button
        st.markdown("---")
        predict_button = st.button("🔮 Predict Quality", type="primary", use_container_width=True)
    
    with col2:
        st.header("📈 Prediction Result")
        
        if predict_button:
            if not api_healthy:
                st.error("Cannot make prediction - API is not available")
            else:
                # Prepare features
                features = {
                    "fixed_acidity": fixed_acidity,
                    "volatile_acidity": volatile_acidity,
                    "citric_acid": citric_acid,
                    "residual_sugar": residual_sugar,
                    "chlorides": chlorides,
                    "free_sulfur_dioxide": free_sulfur_dioxide,
                    "total_sulfur_dioxide": total_sulfur_dioxide,
                    "density": density,
                    "pH": pH,
                    "sulphates": sulphates,
                    "alcohol": alcohol
                }
                
                with st.spinner("Predicting..."):
                    result = get_prediction(features)
                
                if result:
                    # Display prediction
                    if result["quality_label"] == "good":
                        st.markdown("""
                        <div class="prediction-good">
                            <h2>🟢 GOOD QUALITY</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="prediction-bad">
                            <h2>🔴 BAD QUALITY</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Confidence metrics
                    st.subheader("Confidence Scores")
                    
                    col_good, col_bad = st.columns(2)
                    with col_good:
                        st.metric(
                            "Good Quality Probability", 
                            f"{result['probability_good']:.1%}"
                        )
                    with col_bad:
                        st.metric(
                            "Bad Quality Probability", 
                            f"{result['probability_bad']:.1%}"
                        )
                    
                    # Progress bar for confidence
                    st.progress(result["confidence"], text=f"Model Confidence: {result['confidence']:.1%}")
        else:
            st.info("👆 Enter wine properties and click 'Predict Quality'")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: gray;'>Wine Quality MLOps Project | "
        "Powered by FastAPI + Streamlit</p>", 
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
