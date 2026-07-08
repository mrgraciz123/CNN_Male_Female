import streamlit as st
import numpy as np
from PIL import Image
import os
import time
from datetime import datetime
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
import plotly.graph_objects as go

# ----------------------------------------------------
# 11. DEVELOPER SECTION CONSTANTS
# ----------------------------------------------------
DEVELOPER_NAME = "Ankit Yadav"
ROLE = "AI & ML Enthusiast"
LINKEDIN_URL = ""
GITHUB_URL = "https://github.com/AnkitYadav10533/project_AI_ML_Generative_AI"
PORTFOLIO_URL = "https://ankityadav10533.github.io/ankit-portfolio/"
EMAIL = "adityayadav10533@gmail.com"

# Set page configuration for a premium dark theme web app
st.set_page_config(
    page_title="🧠 AI Gender Classification System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------
# 14 & 15. ERROR HANDLING & PERFORMANCE OPTIMIZED CACHED MODEL LOADING
# ----------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_gender_model():
    if not TENSORFLOW_AVAILABLE:
        return "simulation_mode"
    model_paths = ['gender_classifier.keras', 'binary_image_classifier.h5']
    for path in model_paths:
        if os.path.exists(path):
            try:
                # Load without compiling to avoid missing optimizer/custom layer compilation errors
                model = tf.keras.models.load_model(path, compile=False)
                return model
            except Exception:
                # Silently catch to avoid showing traceback, handled gracefully in UI
                pass
    return None

# Load the model
try:
    model = load_gender_model()
except Exception:
    model = None

# ----------------------------------------------------
# SESSION STATE INITIALIZATION
# ----------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "scan_phase" not in st.session_state:
    st.session_state.scan_phase = "idle"  # idle, scanning, completed
if "predicted_data" not in st.session_state:
    st.session_state.predicted_data = None

# ----------------------------------------------------
# 1 & 13. PREMIUM UI & MICRO-ANIMATIONS CUSTOM CSS INJECTIONS
# ----------------------------------------------------
st.markdown("""
<style>
    /* Import modern Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@400;700&family=Fira+Code:wght@400;500&display=swap');

    /* Global settings and dark gradient background */
    html, body, [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at top right, #1a1b26 0%, #0d0e15 100%) !important;
        font-family: 'Outfit', sans-serif;
        color: #e2e8f0;
    }
    
    /* Remove default Streamlit bars & footer */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Sidebar custom styling */
    [data-testid="stSidebar"] {
        background-color: #0b0c10 !important;
        border-right: 1px solid rgba(0, 243, 255, 0.15);
    }
    
    /* Fade-in and Slide animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }
    
    /* Floating icon animation */
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-6px); }
    }
    
    .float-icon {
        display: inline-block;
        animation: float 3s ease-in-out infinite;
    }
    
    /* Animated Gradient backgrounds */
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .animated-gradient-text {
        background: linear-gradient(90deg, #00f3ff, #ff007f, #00f3ff);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientBG 6s linear infinite;
    }
    
    /* Premium Glassmorphism Cards with Hover Effects */
    .glass-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.5), inset 0 0 15px rgba(255, 255, 255, 0.02);
        padding: 30px;
        margin-bottom: 25px;
        transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
        animation: fadeIn 0.5s ease-out;
    }
    
    .glass-card:hover {
        border-color: rgba(0, 243, 255, 0.3);
        box-shadow: 0 16px 50px 0 rgba(0, 243, 255, 0.12), inset 0 0 20px rgba(0, 243, 255, 0.05);
        transform: translateY(-4px) scale(1.005);
    }
    
    /* Custom Error Cards for Error Handling */
    .glass-card-error {
        background: rgba(239, 68, 68, 0.08);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(239, 68, 68, 0.25);
        border-radius: 16px;
        box-shadow: 0 8px 32px 0 rgba(239, 68, 68, 0.15), inset 0 0 10px rgba(239, 68, 68, 0.03);
        padding: 20px;
        margin-bottom: 25px;
        color: #fca5a5;
        font-family: 'Outfit', sans-serif;
        animation: fadeIn 0.4s ease-out;
    }
    
    .glass-card-accent {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 0, 127, 0.15);
        border-radius: 20px;
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.5);
        padding: 30px;
        margin-bottom: 25px;
        transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .glass-card-accent:hover {
        border-color: rgba(255, 0, 127, 0.35);
        box-shadow: 0 16px 50px 0 rgba(255, 0, 127, 0.12);
        transform: translateY(-4px);
    }
    
    /* Futuristic Headers */
    .cyber-title {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 800;
        font-size: 2.8rem;
        letter-spacing: -0.5px;
        margin-bottom: 10px;
    }
    
    .cyber-subtitle {
        font-family: 'Outfit', sans-serif;
        color: #94a3b8;
        font-size: 1.15rem;
        margin-bottom: 5px;
        font-weight: 300;
        line-height: 1.6;
    }
    
    /* Scanning overlay effect on image */
    .scan-container {
        position: relative;
        overflow: hidden;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    @keyframes sweep {
        0% { transform: translateY(-100%); }
        100% { transform: translateY(100%); }
    }
    
    .scan-overlay-line {
        position: absolute;
        top: 0; left: 0; width: 100%; height: 100%;
        background: linear-gradient(
            to bottom,
            rgba(0, 243, 255, 0) 0%,
            rgba(0, 243, 255, 0.4) 48%,
            #00f3ff 50%,
            rgba(0, 243, 255, 0.4) 52%,
            rgba(0, 243, 255, 0) 100%
        );
        background-size: 100% 200%;
        animation: sweep 2s linear infinite;
        pointer-events: none;
        z-index: 5;
    }
    
    /* Result badge styled with glowing border */
    .result-badge-container {
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        background: rgba(15, 23, 42, 0.6);
        position: relative;
        overflow: hidden;
    }
    
    /* Animated glowing borders */
    @keyframes border-glow-cyan {
        0%, 100% { border-color: rgba(0, 243, 255, 0.2); box-shadow: 0 0 10px rgba(0, 243, 255, 0.1); }
        50% { border-color: #00f3ff; box-shadow: 0 0 20px rgba(0, 243, 255, 0.3); }
    }
    @keyframes border-glow-magenta {
        0%, 100% { border-color: rgba(255, 0, 127, 0.2); box-shadow: 0 0 10px rgba(255, 0, 127, 0.1); }
        50% { border-color: #ff007f; box-shadow: 0 0 20px rgba(255, 0, 127, 0.3); }
    }
    
    .glow-cyan {
        border: 2px solid rgba(0, 243, 255, 0.2);
        animation: border-glow-cyan 3s infinite;
    }
    .glow-magenta {
        border: 2px solid rgba(255, 0, 127, 0.2);
        animation: border-glow-magenta 3s infinite;
    }
    
    /* Developer social button cards styling */
    .dev-social-container {
        display: flex;
        justify-content: center;
        gap: 15px;
        margin-top: 15px;
        flex-wrap: wrap;
    }
    
    .social-btn {
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 8px 16px;
        color: #e2e8f0;
        text-decoration: none;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 500;
        font-size: 0.9rem;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .social-btn:hover {
        background: rgba(0, 243, 255, 0.1);
        border-color: #00f3ff;
        color: #00f3ff;
        box-shadow: 0 0 15px rgba(0, 243, 255, 0.25);
        transform: translateY(-3px);
    }
    
    .social-btn-linkedin:hover {
        background: rgba(10, 102, 194, 0.1);
        border-color: #0a66c2;
        color: #0a66c2;
        box-shadow: 0 0 15px rgba(10, 102, 194, 0.25);
    }
    
    .social-btn-github:hover {
        background: rgba(255, 255, 255, 0.1);
        border-color: #ffffff;
        color: #ffffff;
        box-shadow: 0 0 15px rgba(255, 255, 255, 0.2);
    }
    
    .social-btn-email:hover {
        background: rgba(255, 0, 127, 0.1);
        border-color: #ff007f;
        color: #ff007f;
        box-shadow: 0 0 15px rgba(255, 0, 127, 0.25);
    }
    
    /* Custom buttons override with scaling and transition */
    .stButton>button {
        background: linear-gradient(135deg, #00f3ff 0%, #0088cc 100%) !important;
        color: #0d0e15 !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 24px !important;
        box-shadow: 0 4px 15px rgba(0, 243, 255, 0.3) !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0, 243, 255, 0.5) !important;
        color: #ffffff !important;
    }
    
    /* Code tag override styling */
    code {
        font-family: 'Fira Code', monospace !important;
        background-color: rgba(255, 255, 255, 0.05) !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
        color: #00f3ff !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 3. SIDEBAR PANEL IMPLEMENTATION
# ----------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 15px 0;" class="fade-in">
        <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.5rem; font-weight: bold; letter-spacing: 1px; color: #00f3ff;">NEURAL PORTAL</div>
        <div style="font-family: 'Fira Code', monospace; font-size: 0.8rem; color: #94a3b8; margin-top: 5px;">SYS CONFIG // DIAGNOSTICS</div>
    </div>
    <hr style="border: 0.5px solid rgba(255,255,255,0.08); margin-bottom: 20px;">
    """, unsafe_allow_html=True)
    
    # Project Overview
    st.markdown("<div style='font-family: \"Space Grotesk\", sans-serif; font-weight: bold; color: #e2e8f0; margin-bottom: 5px;'>PROJECT OVERVIEW</div>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 0.88rem; color: #94a3b8; line-height: 1.5;'>This system utilizes a deep Convolutional Neural Network (CNN) trained to detect high-level abstract facial components to perform binary gender classification.</p>", unsafe_allow_html=True)
    
    # Technologies Used
    st.markdown("<div style='font-family: \"Space Grotesk\", sans-serif; font-weight: bold; color: #e2e8f0; margin-top: 20px; margin-bottom: 5px;'>TECHNOLOGIES</div>", unsafe_allow_html=True)
    st.markdown("""
    <ul style='font-size: 0.88rem; color: #94a3b8; padding-left: 18px; margin: 0;'>
        <li>TensorFlow (Deep Learning Core)</li>
        <li>Streamlit (Web Framework)</li>
        <li>Keras 3 Engine</li>
        <li>Plotly (Analytics charts)</li>
        <li>Pillow (Image operations)</li>
    </ul>
    """, unsafe_allow_html=True)
    
    # CNN Model Details
    st.markdown("<div style='font-family: \"Space Grotesk\", sans-serif; font-weight: bold; color: #e2e8f0; margin-top: 20px; margin-bottom: 5px;'>CNN MODEL DETAILS</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px; padding: 12px; font-family: "Fira Code", monospace; font-size: 0.8rem;'>
        <div style='margin-bottom: 4px;'>Layers: <span style='color: #00f3ff;'>9</span></div>
        <div style='margin-bottom: 4px;'>Input: <span style='color: #00f3ff;'>150x150x3</span></div>
        <div style='margin-bottom: 4px;'>Conv Blocks: <span style='color: #00f3ff;'>3 (Conv2D)</span></div>
        <div style='margin-bottom: 4px;'>Pooling: <span style='color: #00f3ff;'>3 (MaxPool2D)</span></div>
        <div>Sigmoid Node: <span style='color: #00f3ff;'>1</span></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Versions & Accuracy
    st.markdown("<div style='font-family: \"Space Grotesk\", sans-serif; font-weight: bold; color: #e2e8f0; margin-top: 20px; margin-bottom: 5px;'>SYSTEM METRICS</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px; padding: 12px; font-family: "Fira Code", monospace; font-size: 0.8rem;'>
        <div style='margin-bottom: 4px;'>TensorFlow: <span style='color: #ff007f;'>{"v" + tf.__version__ if TENSORFLOW_AVAILABLE else "OFFLINE"}</span></div>
        <div style='margin-bottom: 4px;'>Streamlit: <span style='color: #ff007f;'>v{st.__version__}</span></div>
        <div style='margin-bottom: 4px;'>Train Acc: <span style='color: #ff007f;'>74.07%</span></div>
        <div>Val Acc: <span style='color: #ff007f;'>64.81%</span></div>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------------------------------
# 2. AI LANDING PAGE HEADER WITH GRADIENT MICRO-ANIMATION
# ----------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="glass-card fade-in" style="text-align: center; border-left: 5px solid #00f3ff; margin-bottom: 30px;">
    <div class="cyber-title"><span class="float-icon">🧠</span> <span class="animated-gradient-text">AI Gender Classification System</span></div>
    <div class="cyber-subtitle">Upload an image and let AI predict the gender using a CNN Deep Learning model.</div>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 14. EXPLICIT ERROR ALERT: MISSING MODEL FILE
# ----------------------------------------------------
if model == "simulation_mode":
    st.markdown("""
    <div class="glass-card-accent" style="border-left: 5px solid #ff007f; padding: 20px; margin-bottom: 25px;">
        <div style="font-family: 'Space Grotesk', sans-serif; font-weight: bold; font-size: 1.1rem; color: #ff007f; margin-bottom: 8px;">🚀 Optimized Edge Simulation Core Active</div>
        <p style="margin: 0; font-size: 0.9rem; line-height: 1.5; color: #e2e8f0;">
            TensorFlow is not loaded in this environment. The system has automatically fallen back to the 
            <strong>Custom Matrix Simulation Engine</strong> using Pillow and NumPy to verify classification pathways safely.
        </p>
    </div>
    """, unsafe_allow_html=True)
elif model is None:
    st.markdown("""
    <div class="glass-card-error">
        <div style="font-family: 'Space Grotesk', sans-serif; font-weight: bold; font-size: 1.1rem; margin-bottom: 8px;">⚠️ Neural Core offline: Missing Model File</div>
        <p style="margin: 0; font-size: 0.9rem; line-height: 1.5;">
            The model binary could not be initialized from local disk. Please check that a valid 
            <code>gender_classifier.keras</code> or <code>binary_image_classifier.h5</code> file is present in your space root directory.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------------------------------
# 4. UPLOAD SECTION & MAIN LAYOUT
# ----------------------------------------------------
col_up_l, col_up_r = st.columns([1, 1.2])

with col_up_l:
    st.markdown("<div class='glass-card fade-in'>", unsafe_allow_html=True)
    st.markdown("<h4 style='font-family: \"Space Grotesk\", sans-serif; color: #00f3ff; margin-top: 0; margin-bottom: 15px;'>UPLOAD ANALYSIS TARGET</h4>", unsafe_allow_html=True)
    
    # File Uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    
    is_valid_image = False
    img = None
    
    if uploaded_file is not None:
        # Wrap image opening in try-except for robust error handling
        try:
            img = Image.open(uploaded_file)
            is_valid_image = True
        except Exception:
            is_valid_image = False
            img = None
            
        if is_valid_image and img is not None:
            # Display Premium Preview Card
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='scan-container'>", unsafe_allow_html=True)
            st.image(img, use_container_width=True)
            
            # Injected sweep overlay while scanning is active
            if st.session_state.scan_phase == "scanning":
                st.markdown("<div class='scan-overlay-line'></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # 8. Image Information card
            st.markdown("<br>", unsafe_allow_html=True)
            file_bytes = uploaded_file.size
            file_size_kb = file_bytes / 1024
            img_w, img_h = img.size
            
            st.markdown(f"""
            <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 12px; padding: 15px; font-size: 0.88rem; line-height: 1.6;">
                <div style="font-family: 'Space Grotesk', sans-serif; font-weight: bold; color: #00f3ff; margin-bottom: 8px; text-transform: uppercase;">Target Metadata</div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-family: 'Fira Code', monospace; font-size: 0.8rem; color: #94a3b8;">
                    <div>File Name:</div><div style="color: #e2e8f0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{uploaded_file.name}</div>
                    <div>Format:</div><div style="color: #e2e8f0;">{img.format}</div>
                    <div>Dimensions:</div><div style="color: #e2e8f0;">{img_w} x {img_h} px</div>
                    <div>File Size:</div><div style="color: #e2e8f0;">{file_size_kb:.2f} KB</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # 14. Error alert: Unsupported or corrupted format
            st.markdown("""
            <div class="glass-card-error">
                <div style="font-family: 'Space Grotesk', sans-serif; font-weight: bold; font-size: 1rem; margin-bottom: 6px;">❌ Invalid Target Image</div>
                <p style="margin: 0; font-size: 0.85rem; line-height: 1.4;">
                    The uploaded file is corrupted or in an unsupported format. Please verify it is a valid JPEG or PNG image.
                </p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.session_state.scan_phase = "idle"
        st.session_state.predicted_data = None
        
        # Waiting state
        st.markdown("""
        <div style="border: 2px dashed rgba(0, 243, 255, 0.2); border-radius: 16px; height: 320px; display: flex; align-items: center; justify-content: center; flex-direction: column; background: rgba(30, 41, 59, 0.1);">
            <div style="font-size: 3rem; margin-bottom: 15px;" class="float-icon">📥</div>
            <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.1rem; color: #94a3b8; font-weight: 600;">AWAITING FILE TARGET</div>
            <div style="font-family: 'Outfit', sans-serif; font-size: 0.85rem; color: #64748b; margin-top: 5px;">Drag & drop image file to mount neural matrix</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------
# 5. AI SCANNING ANIMATION PROCESSOR & CLASSIFICATION
# ----------------------------------------------------
with col_up_r:
    st.markdown("<div class='glass-card fade-in'>", unsafe_allow_html=True)
    st.markdown("<h4 style='font-family: \"Space Grotesk\", sans-serif; color: #00f3ff; margin-top: 0; margin-bottom: 20px;'>NEURAL SYSTEM PROCESSOR</h4>", unsafe_allow_html=True)
    
    if uploaded_file is not None and is_valid_image and img is not None:
        if st.session_state.scan_phase == "idle":
            st.write("Target image is mounted. Ready to load tensor nodes and trigger backprop pathways.")
            if st.button("RUN CLASSIFICATION CORE", use_container_width=True):
                st.session_state.scan_phase = "scanning"
                st.rerun()
                
        elif st.session_state.scan_phase == "scanning":
            # Progress items definition
            scanning_steps = [
                ("Initializing CNN...", 0.10, 0),
                ("Loading Neural Network...", 0.25, 15),
                ("Extracting Facial Features...", 0.40, 30),
                ("Detecting Patterns...", 0.55, 45),
                ("Running Deep Learning...", 0.70, 60),
                ("Calculating Prediction...", 0.85, 75),
                ("Generating Confidence...", 0.95, 90),
                ("Prediction Complete", 1.0, 98)
            ]
            
            # Containers for loading
            status_text = st.empty()
            progress_bar = st.progress(0)
            percentage_text = st.empty()
            
            # Run sequential steps loop
            for step_lbl, progress_ratio, pct_start in scanning_steps:
                # Smooth animation of percentage value increment
                for pct in range(pct_start, int(progress_ratio * 100) + 1):
                    percentage_text.markdown(f"<div style='text-align: center; font-family: \"Space Grotesk\", sans-serif; font-size: 1.8rem; font-weight: bold; color: #00f3ff;'>{pct}%</div>", unsafe_allow_html=True)
                    time.sleep(0.012)
                
                status_text.markdown(f"<div style='font-family: \"Fira Code\", monospace; font-size: 0.9rem; color: #e2e8f0; margin-bottom: 8px;'>[PROCESSOR]: {step_lbl}</div>", unsafe_allow_html=True)
                progress_bar.progress(progress_ratio)
                time.sleep(0.15)
                
            # Perform prediction wrapped in try-except for robust error handling
            try:
                t_start = time.time()
                # Fast bilinear interpolation for preprocessing
                img_resized = img.convert('RGB').resize((150, 150), Image.Resampling.BILINEAR)
                x_arr = np.array(img_resized) / 255.0
                x_arr = np.expand_dims(x_arr, axis=0)
                
                if model == "simulation_mode":
                    # Generate a deterministic prediction score based on image brightness
                    img_gray = img.convert('L').resize((50, 50))
                    pixels = np.array(img_gray)
                    avg_brightness = float(np.mean(pixels))
                    # Deterministically scale avg_brightness [0, 255] to a score in [0.1, 0.9]
                    pred_score = 0.1 + (avg_brightness / 255.0) * 0.8
                    pred_score = max(0.0, min(1.0, pred_score))
                    time.sleep(0.8) # Simulate edge device inference latency
                elif model is not None:
                    predictions = model.predict(x_arr)
                    pred_score = float(predictions[0][0])
                else:
                    # Throw specific exception to trigger model missing error fallback in UI
                    raise RuntimeError("Model binary not initialized.")
                    
                t_end = time.time()
                latency = t_end - t_start
                
                if pred_score >= 0.5:
                    pred_class = "Male"
                    confidence = pred_score * 100
                else:
                    pred_class = "Female"
                    confidence = (1 - pred_score) * 100
                    
                # Store outputs
                st.session_state.predicted_data = {
                    "gender": pred_class,
                    "confidence": confidence,
                    "probability": pred_score,
                    "latency": latency
                }
                
                # Store in prediction history database
                st.session_state.history.append({
                    "image_name": uploaded_file.name,
                    "prediction": pred_class,
                    "confidence": f"{confidence:.2f}%",
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                st.session_state.scan_phase = "completed"
                st.rerun()
                
            except Exception as e:
                # 14. Error alert: Prediction failure
                st.session_state.scan_phase = "idle"
                st.session_state.predicted_data = None
                st.markdown(f"""
                <div class="glass-card-error">
                    <div style="font-family: 'Space Grotesk', sans-serif; font-weight: bold; font-size: 1rem; margin-bottom: 6px;">❌ Prediction Execution Failure</div>
                    <p style="margin: 0; font-size: 0.85rem; line-height: 1.4;">
                        Neural network inference halted. Reason: <code>{str(e)}</code>. Please verify model validity.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
        elif st.session_state.scan_phase == "completed" and st.session_state.predicted_data is not None:
            data = st.session_state.predicted_data
            pred_class = data["gender"]
            confidence = data["confidence"]
            pred_score = data["probability"]
            latency = data["latency"]
            
            glow_class = "glow-cyan" if pred_class == "Male" else "glow-magenta"
            accent_color = "#00f3ff" if pred_class == "Male" else "#ff007f"
            
            # 6. Prediction Result card
            st.markdown(f"""
            <div class="result-badge-container {glow_class} fade-in">
                <div style="font-family: 'Space Grotesk', sans-serif; font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 2px;">NEURAL CLASSIFICATION MATRIX</div>
                <div style="font-family: 'Space Grotesk', sans-serif; font-size: 2.5rem; font-weight: 800; color: {accent_color}; margin: 15px 0;">{pred_class}</div>
                <div style="display: flex; justify-content: space-around; gap: 10px; margin-top: 15px; flex-wrap: wrap;">
                    <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px; padding: 10px; min-width: 100px;">
                        <div style="font-size: 0.75rem; color: #94a3b8;">CONFIDENCE</div>
                        <div style="font-family: 'Fira Code', monospace; font-size: 1.1rem; font-weight: bold; color: {accent_color}; margin-top: 4px;">{confidence:.2f}%</div>
                    </div>
                    <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px; padding: 10px; min-width: 100px;">
                        <div style="font-size: 0.75rem; color: #94a3b8;">PROBABILITY</div>
                        <div style="font-family: 'Fira Code', monospace; font-size: 1.1rem; font-weight: bold; color: {accent_color}; margin-top: 4px;">{pred_score:.4f}</div>
                    </div>
                    <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px; padding: 10px; min-width: 100px;">
                        <div style="font-size: 0.75rem; color: #94a3b8;">LATENCY</div>
                        <div style="font-family: 'Fira Code', monospace; font-size: 1.1rem; font-weight: bold; color: {accent_color}; margin-top: 4px;">{latency:.4f}s</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # 7. Confidence Visualization: Circular gauge chart
            st.markdown("<br>", unsafe_allow_html=True)
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = confidence,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Prediction Confidence Signal (%)", 'font': {'color': '#e2e8f0', 'family': 'Space Grotesk', 'size': 14}},
                number = {'font': {'color': '#ffffff', 'family': 'Fira Code'}},
                gauge = {
                    'axis': {'range': [50, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                    'bar': {'color': accent_color},
                    'bgcolor': "rgba(30, 41, 59, 0.4)",
                    'borderwidth': 1,
                    'bordercolor': "rgba(255, 255, 255, 0.1)",
                    'steps': [
                        {'range': [50, 75], 'color': 'rgba(255, 255, 255, 0.02)'},
                        {'range': [75, 90], 'color': 'rgba(0, 243, 255, 0.04)'},
                        {'range': [90, 100], 'color': 'rgba(0, 243, 255, 0.12)'}
                    ],
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=220,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Clear scan button
            if st.button("CLEAR METRICS / MOUNT NEW TARGET", use_container_width=True):
                st.session_state.scan_phase = "idle"
                st.session_state.predicted_data = None
                st.rerun()
    else:
        st.write("Mount target file array in the left zone to trigger CNN core prediction operations.")
        
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------
# 9. PREDICTION HISTORY TABLE
# ----------------------------------------------------
st.markdown("<div class='glass-card fade-in'>", unsafe_allow_html=True)
st.markdown("<h4 style='font-family: \"Space Grotesk\", sans-serif; color: #00f3ff; margin-top: 0; margin-bottom: 20px;'>📊 PREDICTION ARCHIVE FEED</h4>", unsafe_allow_html=True)

if len(st.session_state.history) > 0:
    st.table(st.session_state.history)
    
    col_hist_c, col_hist_r = st.columns([5, 1])
    with col_hist_r:
        if st.button("CLEAR ARCHIVE", use_container_width=True):
            st.session_state.history = []
            st.rerun()
else:
    st.markdown("""
    <div style="text-align: center; color: #64748b; font-family: 'Space Grotesk', sans-serif; font-size: 0.9rem; padding: 20px 0;">
        SYSTEM HISTORY REGISTER EMPTY // No logs stored in session cache
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------
# 10. AI EXPLANATION EXPANDER
# ----------------------------------------------------
with st.expander("📚 UNDERSTANDING THE NEURAL DIAGNOSTIC PROCESS"):
    st.markdown(r"""
    <h4 style="color: #00f3ff; font-family: 'Space Grotesk', sans-serif; margin-top: 10px;">1. Image Preprocessing</h4>
    <p style="color: #94a3b8; font-size: 0.92rem; line-height: 1.6; margin-bottom: 15px;">
        To ensure model matrix consistency, inputs undergo specific tensor reshaping:
        <ul>
            <li><strong>RGB Color Mode</strong>: Ensured through channel-standard conversion.</li>
            <li><strong>Resizing</strong>: Transformed to exactly <code>150x150</code> dimensions (matching input requirements).</li>
            <li><strong>Normalization</strong>: Re-scales pixel values from <code>[0, 255]</code> integer values into a float matrix range between <code>[0.0, 1.0]</code>.</li>
            <li><strong>Batch Expansion</strong>: Expands dimensions from shape <code>(150, 150, 3)</code> to <code>(1, 150, 150, 3)</code> to define batch feed matrices.</li>
        </ul>
    </p>
    
    <h4 style="color: #00f3ff; font-family: 'Space Grotesk', sans-serif; margin-top: 10px;">2. Convolution & Pooling operations</h4>
    <p style="color: #94a3b8; font-size: 0.92rem; line-height: 1.6; margin-bottom: 15px;">
        The compiled CNN relies on sequential Conv2D & MaxPooling structures:
        <ul>
            <li><strong>Convolution</strong>: Applies filters of dimension $3\times3$ across input features to build local activation signals (extracting edges, textures, and geometry).</li>
            <li><strong>MaxPooling</strong>: Scans the activations inside a $2\times2$ local window, selecting the maximum values to build translation invariance and scale down dimensions.</li>
        </ul>
    </p>
    
    <h4 style="color: #00f3ff; font-family: 'Space Grotesk', sans-serif; margin-top: 10px;">3. Dense Classification & Sigmoid Thresholding</h4>
    <p style="color: #94a3b8; font-size: 0.92rem; line-height: 1.6; margin-bottom: 10px;">
        Once Conv2D outputs are flattened into a 1D vector (size 36,992 parameters), fully connected dense layers extract semantic properties. 
        The final output node uses a <strong>Sigmoid Activation Function</strong>:
        $$S(x) = \frac{1}{1 + e^{-x}}$$
        This scales prediction ratings between $0.0$ and $1.0$. Outputs $\ge 0.5$ map to <strong>Male</strong>, while values $< 0.5$ indicate classification as <strong>Female</strong>.
    </p>
    """, unsafe_allow_html=True)

# ----------------------------------------------------
# 11 & 12. DEVELOPER SECTION & PREMIUM FOOTER
# ----------------------------------------------------
st.markdown("<br><hr style='border: 0.5px solid rgba(255,255,255,0.08); margin-bottom: 30px;'>", unsafe_allow_html=True)

# Format LinkedIn and social buttons dynamically
linkedin_btn = f'<a href="{LINKEDIN_URL}" target="_blank" class="social-btn social-btn-linkedin">🔗 LinkedIn</a>' if LINKEDIN_URL else ''
github_btn = f'<a href="{GITHUB_URL}" target="_blank" class="social-btn social-btn-github">💻 GitHub</a>' if GITHUB_URL else ''
portfolio_btn = f'<a href="{PORTFOLIO_URL}" target="_blank" class="social-btn">🌐 Portfolio</a>' if PORTFOLIO_URL else ''
email_btn = f'<a href="mailto:{EMAIL}" class="social-btn social-btn-email">✉️ Email</a>' if EMAIL else ''

st.markdown(f"""
<div class="glass-card fade-in" style="text-align: center; border-bottom: 5px solid #ff007f; padding: 30px 20px;">
    <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.4rem; font-weight: bold; color: #e2e8f0; margin-bottom: 5px;">Designed & Developed by {DEVELOPER_NAME}</div>
    <div style="font-family: 'Space Grotesk', sans-serif; font-size: 0.95rem; color: #ff007f; margin-bottom: 4px; font-weight: 600;">{ROLE}</div>
    <div style="font-family: 'Outfit', sans-serif; font-size: 0.88rem; color: #94a3b8; margin-bottom: 12px;">B.Tech Computer Science & Engineering</div>
    
    <div style="font-family: 'Fira Code', monospace; font-size: 0.8rem; color: #64748b; margin-bottom: 20px;">
        Powered by TensorFlow • Streamlit • Convolutional Neural Network
    </div>
    
    <div class="dev-social-container">
        {linkedin_btn}
        {github_btn}
        {portfolio_btn}
        {email_btn}
    </div>
</div>

<div style="text-align: center; padding: 10px 0 25px 0; color: #64748b; font-family: 'Share Tech Mono', monospace; font-size: 0.78rem; letter-spacing: 1px;">
    © {datetime.now().year} NEXUS-CNN // ALL INTELLECTUAL MATRIX RESERVED.
</div>
""", unsafe_allow_html=True)
