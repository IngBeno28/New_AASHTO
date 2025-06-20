# --- Streamlit Config MUST BE FIRST ---
import streamlit as st
st.set_page_config(
    page_title="AASHTO Classifier",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Imports ---
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
import base64
import os
import time
from typing import Optional

# --- Streamlit Cloud Optimized Config ---
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Lightweight for free tier
CACHE_DIR = "/tmp/model_cache"  # Streamlit Cloud compatible cache
MAX_RETRIES = 3  # For model loading retries
RETRY_DELAY = 5  # Seconds between retries

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

# --- Model Import with Robust Error Handling ---
def safe_import_transformers() -> Optional[bool]:
    """Safely import transformers with multiple fallback attempts"""
    for attempt in range(MAX_RETRIES):
        try:
            from transformers import pipeline
            return pipeline
        except ImportError as e:
            if attempt == MAX_RETRIES - 1:
                st.warning(f"Transformers import failed after {MAX_RETRIES} attempts")
                return None
            time.sleep(RETRY_DELAY)
    return None

# Get the pipeline function if available
pipeline = safe_import_transformers()
MODEL_LOADED = pipeline is not None

# --- Load AI Model with Retry Logic ---
@st.cache_resource(ttl=3600)  # Cache for 1 hour
def load_ai_model():
    if not MODEL_LOADED:
        return None
    
    for attempt in range(MAX_RETRIES):
        try:
            model = pipeline(
                "text-generation",
                model=MODEL_NAME,
                device_map="auto",
                model_kwargs={
                    "cache_dir": CACHE_DIR,
                    "torch_dtype": "auto"
                }
            )
            st.success("AI model loaded successfully!")
            return model
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                st.error(f"Model loading failed after {MAX_RETRIES} attempts: {str(e)}")
                return None
            time.sleep(RETRY_DELAY * (attempt + 1))  # Exponential backoff

text_gen = load_ai_model()

# --- Mobile Optimized CSS ---
st.markdown("""
    <style>
    .stNumberInput, .stTextInput {width: 100% !important;}
    .stDownloadButton {width: 100%;}
    @media (min-width: 768px) {
        .stDownloadButton {width: auto;}
    }
    /* Better error message styling */
    .stAlert {padding: 1rem !important;}
    /* Form submit button styling */
    .stFormSubmitButton button {
        background-color: #4CAF50 !important;
        color: white !important;
        font-weight: bold !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- PDF Generator ---
def create_pdf(classification: str, analysis: str, chart_path: Optional[str] = None) -> bytes:
    """Generate PDF report with optional chart"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "AASHTO Soil Classification Report", ln=1)
    pdf.ln(10)
    
    # Classification
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, f"Classification: {classification}", ln=1)
    
    # AI Analysis
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, analysis)
    
    # Add chart if available
    if chart_path and os.path.exists(chart_path):
        pdf.image(chart_path, x=10, w=180)
    
    return pdf.output(dest='S').encode('latin1')

# --- Soil Classification Logic ---
GRANULAR_MATERIALS = ["A-1-a", "A-1-b", "A-3", "A-2-4", "A-2-5", "A-2-6", "A-2-7"]
SILTY_CLAY_MATERIALS = ["A-4", "A-5", "A-6", "A-7"]

def classify_soil(LL: float, PL: float, PI: float, 
                 pass_10: float, pass_40: float, pass_200: float, 
                 is_np: bool) -> str:
    """Classify soil according to AASHTO system"""
    if is_np:
        PI = 0
    if pass_10 <= 50 and pass_40 <= 30 and pass_200 <= 15 and PI <= 6:
        return "A-1-a"
    elif pass_40 <= 50 and pass_200 <= 25 and PI <= 6:
        return "A-1-b"
    elif pass_40 >= 51 and pass_200 <= 10 and PI == 0:
        return "A-3"
    elif pass_200 <= 35 and LL <= 40 and PI <= 10:
        return "A-2-4"
    elif pass_200 <= 35 and LL >= 41 and PI <= 10:
        return "A-2-5"
    elif pass_200 <= 35 and LL <= 40 and PI >= 11:
        return "A-2-6"
    elif pass_200 <= 35 and LL >= 41 and PI >= 11:
        return "A-2-7"
    elif pass_200 >= 36 and LL <= 40 and PI <= 10:
        return "A-4"
    elif pass_200 >= 36 and LL >= 41 and PI <= 10:
        return "A-5"
    elif pass_200 >= 36 and LL <= 40 and PI >= 11:
        return "A-6"
    elif pass_200 >= 36 and LL >= 41 and PI >= 11:
        return "A-7"
    else:
        return "Invalid input or not classifiable"

def classify_material_type(pass_200: float) -> str:
    """Determine if material is granular or silt-clay"""
    return "Granular Material" if pass_200 <= 35 else "Silt-Clay Material"

def identify_constituents_from_classification(classification: str) -> str:
    """Get material constituents based on classification"""
    if classification in ("A-1-a", "A-1-b"):
        return "Stone fragments, Gravel and Sand"
    elif classification == "A-3":
        return "Fine sand"
    elif classification in ("A-2-4", "A-2-5", "A-2-6", "A-2-7"):
        return "Silty or Clayey Gravel and Sand"
    elif classification in ("A-4", "A-5"):
        return "Silty soils"
    elif classification in ("A-6", "A-7"):
        return "Clayey soils"
    return "Unknown"

# --- Streamlit UI ---
with st.form("soil_form"):
    cols = st.columns(2)
    
    with cols[0]:
        st.subheader("Atterberg Limits")
        LL = st.number_input("Liquid Limit (LL)", min_value=0, max_value=100, value=30)
        PL = st.number_input("Plastic Limit (PL)", min_value=0, max_value=100, value=20)
        is_np = st.checkbox("Non-Plastic (N.P)")
        PI = 0 if is_np else max(0, LL - PL)  # Ensure PI isn't negative
        if not is_np:
            st.write(f"Plasticity Index (PI) = **{PI}**")
    
    with cols[1]:
        st.subheader("Sieve Analysis (%)")
        pass_10 = st.number_input("No. 10 (2.0mm)", 0, 100, 50)
        pass_40 = st.number_input("No. 40 (0.425mm)", 0, 100, 30)
        pass_200 = st.number_input("No. 200 (0.075mm)", 0, 100, 15)

    submitted = st.form_submit_button("Classify Soil")

if submitted:
    # --- Classification ---
    classification = classify_soil(LL, PL, PI, pass_10, pass_40, pass_200, is_np)
    mat_type = classify_material_type(pass_200)
    constituents = identify_constituents_from_classification(classification)
    
    st.success(f"**Classification:** {classification}")
    st.info(f"**Material Type:** {mat_type}")
    
    # --- AI Analysis ---
    ai_text = f"Standard properties for {classification}: {constituents}"
    if text_gen:
        with st.spinner("Generating AI insights..."):
            try:
                prompt = f"""Explain AASHTO {classification} soil classification in 50 words for civil engineers. 
                Include: key properties, typical uses in construction, and limitations."""
                result = text_gen(prompt, max_length=150, do_sample=True, temperature=0.7)
                ai_text = result[0]['generated_text']
            except Exception as e:
                st.warning(f"AI analysis failed: {str(e)}")
    
    with st.expander("🧠 AI Analysis", expanded=True):
        st.write(ai_text)
    
    # --- PDF Export ---
    chart_path = None
    try:
        chart_path = "sieves.png"
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(['No.10', 'No.40', 'No.200'], [pass_10, pass_40, pass_200], color='teal')
        ax.set_ylim(0, 100)
        ax.set_ylabel('Percentage Passing (%)')
        ax.set_title('Sieve Analysis Results')
        plt.tight_layout()
        plt.savefig(chart_path, dpi=100)
        plt.close()
        
        pdf = create_pdf(classification, ai_text, chart_path)
        st.download_button(
            label="📄 Download Full Report (PDF)",
            data=pdf,
            file_name=f"soil_report_{classification}.pdf",
            mime="application/pdf"
        )
    except Exception as e:
        st.error(f"PDF generation failed: {str(e)}")
    finally:
        if chart_path and os.path.exists(chart_path):
            os.remove(chart_path)

# --- Deployment Information ---
with st.expander("ℹ️ Deployment Notes"):
    st.markdown(f"""
    **For Streamlit Cloud:**
    - Using lightweight {MODEL_NAME} model (fits free tier memory limits)
    - Model cached in {CACHE_DIR} for faster reloads
    - Robust error handling with {MAX_RETRIES} retry attempts
    - Current status: {"✅ AI Model Loaded" if MODEL_LOADED else "⚠️ Running in limited mode"}
    
    **Tips:**
    - If model fails to load, try refreshing the page
    - For complex soils, the AI analysis provides additional insights
    - All reports include the sieve analysis chart
    """)
