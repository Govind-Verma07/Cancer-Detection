import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import os
import sys
import base64
from io import BytesIO
import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import run_inference
from utils.config import Config
from utils.report_gen import generate_report

# --- PREMIUM DESIGN ---
st.set_page_config(
    page_title="MammogramAI | Advanced Diagnostic Suite",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Modern "Wow" Aesthetics (Glassmorphism & HSL)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    .main {
        background-color: #05070a;
        background-image: radial-gradient(circle at 50% 50%, #101520 0%, #05070a 100%);
        color: #f8fafc;
    }

    /* Glassmorphism Title Card */
    .title-container {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    .medical-title {
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: 0;
    }

    .subtitle {
        color: #94a3b8;
        font-weight: 300;
        font-size: 1.2rem;
        letter-spacing: 1px;
    }

    /* Metric Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin-top: 2rem;
    }

    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        transition: transform 0.3s ease, background 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.08);
        border-color: #38bdf8;
    }

    .metric-label {
        color: #94a3b8;
        font-size: 0.8rem;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 1px;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        margin-top: 0.5rem;
    }

    /* Result Indicators */
    .status-malignant { color: #f43f5e; } /* Rose-500 */
    .status-benign { color: #10b981; }    /* Emerald-500 */
    .status-absent { color: #38bdf8; }    /* Sky-400 */

    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .stImage, .metric-card {
        animation: fadeIn 0.8s ease-out forwards;
    }

    /* File Uploader override */
    .stFileUploader section {
        background: rgba(56, 189, 248, 0.05) !important;
        border: 2px dashed rgba(56, 189, 248, 0.3) !important;
        border-radius: 16px !important;
    }

    /* Sidebar buttons */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.6rem 1rem;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.4);
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER SECTION ---
st.markdown("""
<div class="title-container">
    <h1 class="medical-title">🧬 MammogramAI</h1>
    <p class="subtitle">Next-Generation Breast Cancer Diagnostic Suite</p>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR & DEMO ---
with st.sidebar:
    st.image("https://img.icons8.com/nolan/256/caduceus.png", width=100)
    st.markdown("### 🏥 Clinical Dashboard")
    st.info(f"🟢 **Core**: PyTorch v{torch.__version__}")
    st.info(f"🔵 **HW Access**: {Config.DEVICE.upper()}")
    
    st.divider()
    
    st.markdown("### 🖼️ Demo Clinical Samples")
    demo_samples = {
        "None": None,
        "Sample Case 001": r"D:\dataset for dl\Dataset DL\TIFF Images\TIFF Images\IMG001.tif",
        "Sample Case 002": r"D:\dataset for dl\Dataset DL\TIFF Images\TIFF Images\IMG002.tif",
        "Sample Case 003": r"D:\dataset for dl\Dataset DL\TIFF Images\TIFF Images\IMG003.tif"
    }
    selected_demo = st.selectbox("Quick-Load High Res TIFF:", list(demo_samples.keys()))
    
    st.divider()
    
    if st.button("🧹 Clear Workspace"):
        st.rerun()

# --- INPUT LOGIC ---
uploaded_file = st.file_uploader("Drag & Drop Mammography Scan here...", type=["tif", "tiff", "jpg", "png", "jpeg"])

input_path = None
if selected_demo != "None":
    input_path = demo_samples[selected_demo]
    st.success(f"Loaded {selected_demo} successfully.")
elif uploaded_file:
    ext = os.path.splitext(uploaded_file.name)[1]
    input_path = f"temp_ui_upload{ext}"
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

# --- ANALYSIS ENGINE ---
if input_path:
    # Side-by-Side Analysis Layout
    st.markdown("---")
    res_col1, res_col2 = st.columns(2)
    
    with st.spinner("⚛️ Analyzing tissue patterns using VGG16+U-Net..."):
        # Run the actual inference
        # This is where the magic happens from src/inference.py
        try:
            results = run_inference(input_path)
            
            if results:
                # 1. DISPLAY IMAGES (SBS)
                with res_col1:
                    st.markdown("#### 📸 Source Scan")
                    case_img = Image.open(input_path)
                    st.image(case_img, use_container_width=True)
                
                with res_col2:
                    st.markdown("#### 🗺️ AI Contour Mapping")
                    st.image(results['visual_result'], use_container_width=True)
                    st.caption("🟢 Benign | 🟡 Stage 1 | 🟠 Stage 2 | 🔴 Stage 3 | 🟤 Stage 4")

                # 2. METRIC DASHBOARD
                st.markdown("### 📊 Automated Diagnostic Metrics")
                
                nature_str = "ABSENT"
                nature_class = "status-absent"
                if results['tumor_present']:
                    is_malignant = any(f['is_malignant'] for f in results['findings'])
                    nature_str = "MALIGNANT" if is_malignant else "BENIGN"
                    nature_class = "status-malignant" if is_malignant else "status-benign"

                # Custom HTML Grid for Metrics
                st.markdown(f"""
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-label">Tumor Nature</div>
                        <div class="metric-value {nature_class}">{nature_str}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">B-T Area Ratio</div>
                        <div class="metric-value">{results['overall_ratio'] * 100:.2f}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Detected Foci</div>
                        <div class="metric-value">{len(results['findings'])}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # 3. DETAILED FINDINGS TABLE
                if results['findings']:
                    st.markdown("#### 🔬 Regional Analysis")
                    for i, finding in enumerate(results['findings']):
                        with st.expander(f"📦 Focus Site #{i+1} [{finding['classification']}]"):
                            inner_col1, inner_col2 = st.columns(2)
                            with inner_col1:
                                st.write(f"**Classification**: {finding['classification']}")
                                st.write(f"**Anatomical Position**: {finding.get('location_pct', 'N/A')}")
                            with inner_col2:
                                st.write(f"**Estimated Size**: {finding.get('area_pixels', 'N/A')} px")
                                st.write(f"**Density Index**: {finding.get('density_index', 'N/A')}")
                
                # 4. EXPORT & REPORTING
                st.markdown("---")
                final_col1, final_col2 = st.columns(2)
                
                with final_col1:
                    report_content = generate_report(results, uploaded_file.name if uploaded_file else selected_demo)
                    st.download_button(
                        label="📄 Export Clinical Report (.md)",
                        data=report_content,
                        file_name=f"MAI_REPORT_{datetime.datetime.now().strftime('%Y%m%d')}.md",
                        mime="text/markdown"
                    )
                
                with final_col2:
                    # Logic to download the segmented image
                    buffered = BytesIO()
                    # Convert BGR to RGB for PIL
                    img_rgb = cv2.cvtColor(results['visual_result'], cv2.COLOR_BGR2RGB)
                    pil_res = Image.fromarray(img_rgb)
                    pil_res.save(buffered, format="JPEG")
                    st.download_button(
                        label="📥 Download Contour Map (.jpg)",
                        data=buffered.getvalue(),
                        file_name="diagnosis_mapping.jpg",
                        mime="image/jpeg"
                    )

            else:
                st.error("Engine failure: Could not generate analysis. Please check model weights.")
                
        except Exception as e:
            st.error(f"Critical System Error: {str(e)}")
            st.error("Ensure 'models/segmentation_model.pth' and 'models/classification_model.pth' are present.")

else:
    # EMPTY STATE
    st.markdown("""
    <div style="text-align: center; padding: 100px; color: #475569;">
        <img src="https://img.icons8.com/flat-round/256/upload--v1.png" width="120" style="opacity: 0.5;">
        <h2 style="font-weight: 300;">Ready for Diagnostic Analysis</h2>
        <p>Drop a mammogram file or select a Demo Case from the sidebar to begin.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("© 2026 MammogramAI Advanced Systems | Research Prototype Deployment")
