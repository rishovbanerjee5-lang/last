import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas

# Set page configuration
st.set_page_config(
    page_title="MedVision - Healthcare CV Demo",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-bottom: 1rem;
    }
    .highlight {
        background-color: #E3F2FD;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    .result-box {
        background-color: #BBDEFB;
        padding: 15px;
        border-radius: 10px;
        margin-top: 15px;
    }
    .analysis-section {
        background-color: #F5F5F5;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown('<h1 class="main-header">MedVision - Healthcare CV Demo</h1>', unsafe_allow_html=True)
st.markdown("""
<div class="highlight">
    Upload medical images (X-ray, MRI, CT scan) for AI-powered analysis and anomaly detection.
    This demo simulates computer vision capabilities for healthcare applications.
</div>
""", unsafe_allow_html=True)

# Sidebar for controls
with st.sidebar:
    st.header("Settings")
    analysis_mode = st.selectbox(
        "Analysis Mode",
        ["Anomaly Detection", "Segmentation", "Measurement", "Comparison"]
    )
    
    sensitivity = st.slider("Detection Sensitivity", 1, 10, 7)
    
    show_heatmap = st.checkbox("Show Heatmap", value=True)
    show_measurements = st.checkbox("Show Measurements", value=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download Result",
            data="",  # This would be the actual image data in a real app
            file_name="analysis_result.jpg",
            mime="image/jpeg",
            disabled=True  # Disabled for demo
        )
    with col2:
        if st.button("Reset"):
            st.experimental_rerun()

# File uploader
uploaded_file = st.file_uploader("Upload Medical Image", type=["jpg", "jpeg", "png", "dcm"])

if uploaded_file is not None:
    # Create columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Display original image
        image = Image.open(uploaded_file)
        st.image(image, caption='Original Medical Image', use_column_width=True)
        
        # Convert to OpenCV format
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    
    with col2:
        # Progress bar for analysis simulation
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(101):
            status_text.text(f"Analyzing... {i}%")
            progress_bar.progress(i)
            time.sleep(0.02)
        
        status_text.text("Analysis Complete!")
        
        # Simulate different types of analysis based on mode
        if analysis_mode == "Anomaly Detection":
            # Draw multiple highlighted regions
            highlighted_img = img_cv2.copy()
            
            # Draw several "anomaly" regions
            anomalies = [
                ((50, 50), (200, 200), "Possible fracture"),
                ((300, 150), (450, 300), "Density irregularity"),
                ((180, 300), (350, 450), "Opacity detected")
            ]
            
            for i, (start, end, label) in enumerate(anomalies):
                color = (0, 0, 255) if i == 0 else (0, 165, 255)  # Red for primary, orange for others
                thickness = 4 if i == 0 else 2
                cv2.rectangle(highlighted_img, start, end, color, thickness)
                
                # Add text label
                cv2.putText(highlighted_img, label, (start[0], start[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            st.image(cv2.cvtColor(highlighted_img, cv2.COLOR_BGR2RGB), 
                    caption="Anomaly Detection Results", use_column_width=True)
            
            # Results summary
            st.markdown("""
            <div class="result-box">
                <h3>Analysis Results</h3>
                <ul>
                    <li>Primary finding: Possible fracture detected with 87% confidence</li>
                    <li>Secondary findings: Two additional areas of interest identified</li>
                    <li>Recommendation: Further evaluation recommended</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        elif analysis_mode == "Segmentation":
            # Create a segmentation mask simulation
            height, width = img_cv2.shape[:2]
            mask = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Draw some random "segmented" regions
            for _ in range(5):
                center = (np.random.randint(100, width-100), np.random.randint(100, height-100))
                axes = (np.random.randint(30, 100), np.random.randint(30, 100))
                angle = np.random.randint(0, 180)
                color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                
                cv2.ellipse(mask, center, axes, angle, 0, 360, color, -1)
            
            # Blend with original image
            alpha = 0.6
            segmented_img = cv2.addWeighted(img_cv2, 1-alpha, mask, alpha, 0)
            
            st.image(cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB), 
                    caption="Tissue Segmentation", use_column_width=True)
            
            # Results summary
            st.markdown("""
            <div class="result-box">
                <h3>Segmentation Results</h3>
                <ul>
                    <li>5 distinct tissue regions identified</li>
                    <li>Region boundaries clearly demarcated</li>
                    <li>Volume calculations available upon request</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Additional analysis section
    st.markdown("""
    <div class="analysis-section">
        <h3>Detailed Analysis Report</h3>
        <p>This simulated analysis demonstrates how computer vision could assist medical professionals in identifying potential issues in medical imagery.</p>
        
        <div class="row">
            <div class="col">
                <h4>Key Metrics</h4>
                <ul>
                    <li>Anomaly Confidence Score: 87%</li>
                    <li>Image Quality: Excellent</li>
                    <li>Processing Time: 2.1 seconds</li>
                </ul>
            </div>
            <div class="col">
                <h4>Potential Diagnoses</h4>
                <ul>
                    <li>Fracture (probability: 65%)</li>
                    <li>Lesion (probability: 42%)</li>
                    <li>Normal variant (probability: 23%)</li>
                </ul>
            </div>
        </div>
        
        <p><em>Note: This is a demonstration only. Always consult with a qualified medical professional for diagnosis.</em></p>
    </div>
    """, unsafe_allow_html=True)

else:
    # Show placeholder and instructions when no image is uploaded
    st.info("Please upload a medical image to begin analysis.")
    
    # Create placeholder image
    placeholder = np.ones((400, 600, 3), dtype=np.uint8) * 255
    cv2.putText(placeholder, "Medical Image Preview", (150, 200), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(placeholder, "Upload an image to begin analysis", (120, 250), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    st.image(placeholder, use_column_width=True)
    
    # Features list
    st.markdown("""
    <div class="analysis-section">
        <h3>Supported Features</h3>
        <div class="row">
            <div class="col">
                <h4>🔍 Anomaly Detection</h4>
                <p>Identify potential abnormalities in medical images with AI-powered analysis.</p>
            </div>
            <div class="col">
                <h4>📊 Segmentation</h4>
                <p>Separate and classify different tissue types and structures.</p>
            </div>
        </div>
        <div class="row">
            <div class="col">
                <h4>📐 Measurement Tools</h4>
                <p>Precise measurement of anatomical structures and anomalies.</p>
            </div>
            <div class="col">
                <h4>📈 Comparison Analysis</h4>
                <p>Track changes over time by comparing current and previous scans.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #757575;">
        <p>MedVision - Healthcare Computer Vision Demo | For demonstration purposes only | Not for clinical use</p>
    </div>
    """
, unsafe_allow_html=True)
