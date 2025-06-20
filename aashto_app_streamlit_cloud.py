import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
from fpdf import FPDF
import base64
import os

# --- Streamlit Cloud Optimized Config ---
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Lightweight for free tier
CACHE_DIR = "/tmp/model_cache"  # Streamlit Cloud compatible cache

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

# --- Load AI Model (Streamlit-Cloud Compatible) ---
@st.cache_resource(ttl=3600)  # Cache for 1 hour
def load_ai():
    return pipeline(
        "text-generation",
        model=MODEL_NAME,
        device_map="auto",
        model_kwargs={"cache_dir": CACHE_DIR}  # Reduces memory spikes
    )

try:
    text_gen = load_ai()
except Exception as e:
    st.error(f"AI model failed to load: {str(e)}")
    text_gen = None

# --- PDF Generator ---
def create_pdf(classification, analysis, chart_path=None):
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
        os.remove(chart_path)  # Cleanup temp file
    
    return pdf.output(dest='S').encode('latin1')

# --- Mobile Optimized UI ---
st.set_page_config(
    page_title="AASHTO Classifier",
    layout="centered",
    initial_sidebar_state="collapsed"
)
st.markdown("""
    <style>
    .stNumberInput, .stTextInput {width: 100% !important;}
    .stDownloadButton {width: 100%;}
    @media (min-width: 768px) {
        .stDownloadButton {width: auto;}
    }
    </style>
""", unsafe_allow_html=True)

# --- Your Existing Classification Logic ---
granular_materials = ["A-1-a", "A-1-b", "A-3", "A-2-4", "A-2-5", "A-2-6", "A-2-7"]
silty_clay_materials = ["A-4", "A-5", "A-6", "A-7"]

def classify_soil(LL, PL, PI, pass_10, pass_40, pass_200, is_np):
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


def classify_material_type(pass_200):
 return "Granular Material" if pass_200 <= 35 else "Silt-Clay Material"

def identify_constituents_from_classification(classification):
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
    else:
        return "Unknown"


# --- Streamlit UI ---
with st.form("soil_form"):
    cols = st.columns(2)
    
    with cols[0]:
        st.subheader("Atterberg Limits")
        LL = st.number_input("Liquid Limit (LL)", min_value=0)
        PL = st.number_input("Plastic Limit (PL)", min_value=0)
        is_np = st.checkbox("Non-Plastic (N.P)")
        PI = 0 if is_np else LL - PL
        if not is_np:
            st.write(f"Plasticity Index (PI) = **{PI}**")
    
    with cols[1]:
        st.subheader("Sieve Analysis (%)")
        pass_10 = st.number_input("No. 10 (2.0mm)", 0, 100)
        pass_40 = st.number_input("No. 40 (0.425mm)", 0, 100)
        pass_200 = st.number_input("No. 200 (0.075mm)", 0, 100)

    submitted = st.form_submit_button("Classify Soil")

if submitted:
    # --- Classification ---
    classification = classify_soil(LL, PL, PI, pass_10, pass_40, pass_200, is_np)
    mat_type = classify_material_type(pass_200)
    constituents = identify_constituents_from_classification(classification)
    
    st.success(f"**Classification:** {classification}")
    st.info(f"**Material Type:** {mat_type}")
    
    # --- AI Analysis (Fallback if model fails) ---
    ai_text = "Enable AI to get analysis"
    if text_gen:
        with st.spinner("Generating AI insights..."):
            try:
                prompt = f"Explain AASHTO {classification} in 50 words for engineers: key properties, uses, and limitations."
                ai_text = text_gen(prompt, max_length=150)[0]['generated_text']
            except Exception as e:
                st.warning(f"AI offline: {str(e)}")
    
    with st.expander("🧠 AI Analysis", expanded=True):
        st.write(ai_text)
    
    # --- PDF Export ---
    chart_path = None
    try:
        chart_path = "sieves.png"
        fig, ax = plt.subplots()
        ax.bar(['No.10', 'No.40', 'No.200'], [pass_10, pass_40, pass_200], color='teal')
        plt.savefig(chart_path, bbox_inches='tight')
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

# --- Streamlit Cloud Specific Tips ---
with st.expander("ℹ️ Deployment Notes"):
    st.markdown("""
    **For Streamlit Cloud:**
    1. Add `transformers`, `torch`, `fpdf`, `matplotlib` to `requirements.txt`
    2. Set `CACHE_DIR` to `/tmp` (writable in cloud)
    3. Model will auto-download on first run (~2 minutes)
    4. Free tier has **1GB RAM** - TinyLlama fits perfectly
    """)
