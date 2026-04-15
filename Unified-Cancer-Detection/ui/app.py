import streamlit as st
import os
import sys
import numpy as np
from PIL import Image
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.comparison import compare_models
from utils.config import Config

st.set_page_config(page_title="Advanced Cancer Diagnostics", layout="wide", page_icon="🔬")

# UI Styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stHeader {
        background: linear-gradient(135deg, #004e92, #000428);
        color: white;
        padding: 2.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    .stHeader h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    .stHeader p {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    .metric-container {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        flex: 1;
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border-top: 4px solid #004e92;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown('<div class="stHeader"><h1>🧬 Unified Breast Cancer Diagnostic Suite</h1><p>High-Precision Clinical Model Comparison (ResNet50 & VGG16)</p></div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://img.icons8.com/clouds/100/000000/microscope.png", width=100)
st.sidebar.title("Control Panel")
uploaded_file = st.sidebar.file_uploader("Upload Histopathology Scan", type=["jpg", "png", "jpeg", "tif", "tiff"])

# Accuracy Test Button
if st.sidebar.button("Run Accuracy Test (100 Images)"):
    with st.spinner("Running accuracy test on 100 images..."):
        try:
            from accuracy_test import run_accuracy_test
            run_accuracy_test(num_images=100)
            st.sidebar.success("Accuracy test completed!")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

if uploaded_file:
    # Save temp file
    os.makedirs("results", exist_ok=True)
    temp_path = os.path.join("results", "temp_upload.jpg")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.sidebar.success("✅ Image Loaded")
    
    # Run Comparison
    with st.spinner("Processing dual-model inference..."):
        results = compare_models(temp_path)
    
    resnet = results['resnet']
    vgg = results['vgg']
    
    # Load images for display
    resnet_img_path = resnet.get('output_path')
    vgg_img_path = vgg.get('output_path')
    
    # Results Dashboard
    col1, col2 = st.columns(2)
    
    import io
    def load_image_from_path(img_path):
        if img_path and os.path.exists(img_path):
            return Image.open(img_path)
        return None
    
    def img_to_bytes(img):
        if img is None:
            return None
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()
    
    with col1:
        st.markdown(f"### 🟦 ResNet50 Prediction")
        resnet_img = load_image_from_path(resnet_img_path)
        if resnet_img:
            st.image(resnet_img, caption="Segmentation & Classification Map", use_container_width=True)
            st.download_button(
                label="💾 Download ResNet Prediction",
                data=img_to_bytes(resnet_img),
                file_name="resnet_prediction.jpg",
                mime="image/jpeg"
            )
        st.markdown(f"**Tumor Burden Index:** `{resnet.get('overall_ratio', 0):.4f}`")
        st.markdown(f"**Identified Zones:** `{len(resnet.get('findings', []))}`")
        
    with col2:
        st.markdown(f"### 🟧 VGG16 Prediction")
        vgg_img = load_image_from_path(vgg_img_path)
        if vgg_img:
            st.image(vgg_img, caption="Segmentation & Classification Map", use_container_width=True)
            st.download_button(
                label="💾 Download VGG16 Prediction",
                data=img_to_bytes(vgg_img),
                file_name="vgg_prediction.jpg",
                mime="image/jpeg"
            )
        st.markdown(f"**Tumor Burden Index:** `{vgg.get('overall_ratio', 0):.4f}`")
        st.markdown(f"**Identified Zones:** `{len(vgg.get('findings', []))}`")
        
    st.divider()
    
    # Consensus Analysis
    st.header("🧠 Diagnostic Synthesis")
    c1, c2, c3 = st.columns(3)
    
    # Determine consensus color
    consensus = results.get('consensus', 'Unknown')
    consensus_color = "green" if consensus == "Agreement" else "orange"
    
    with c1:
        st.metric("Model Consensus", consensus)
    with c2:
        avg_area = (resnet.get('overall_ratio', 0) + vgg.get('overall_ratio', 0)) / 2
        st.metric("Mean Tumor Area", f"{avg_area:.4f}")
    with c3:
        # Recommendation logic: if they disagree, highlight the one with higher detection area (more conservative)
        if consensus == "Discrepancy":
            rec = "Verification Required"
        else:
            rec = "Consolidated Report"
        st.metric("Clinical Status", rec)
    
    # Conclusion Report
    st.header("📋 Final Conclusion Report")
    ensemble = results.get('ensemble', {})
    conclusion = ensemble.get('conclusion_report', 'No conclusion available')
    st.text_area("Diagnostic Report", conclusion, height=300, disabled=True)
    
    # Detailed Metrics Table
    with st.expander("📂 View Component Analysis Details", expanded=True):
        findings_data = []
        for f in resnet['findings'] + vgg['findings']:
            model_type = "ResNet50" if f in resnet['findings'] else "VGG16"
            findings_data.append({
                "Architecture": model_type,
                "Relative Area": f['area_pixels'],
                "Classification": f['classification'],
                "Malignancy Prob.": f.get('density_index', "N/A"),
                "Coordinates": f['location_pct']
            })
        
        if findings_data:
            df_findings = pd.DataFrame(findings_data)
            st.dataframe(df_findings, use_container_width=True)
        else:
            st.info("Zero malignancy indicators detected in current scan.")

else:
    # Landing Page
    st.info("👋 Welcome! Please upload a medical histopathology image (JPG/PNG) using the sidebar to begin analysis.")
    
    # History Section
    if os.path.exists(Config.COMPARISON_LOG):
        st.markdown("### 📜 Recent Analysis History")
        history = pd.read_csv(Config.COMPARISON_LOG)
        st.dataframe(history.tail(10).sort_index(ascending=False), use_container_width=True)
