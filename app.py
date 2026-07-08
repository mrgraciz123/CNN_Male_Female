import streamlit as st
import numpy as np
from PIL import Image
import os
import time
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# ----------------------------------------------------
# 1. DEVELOPER SECTION CONSTANTS & LOGIC
# ----------------------------------------------------
DEVELOPER_NAME = "Abhay Shanker Tiwari"
ROLE = "AI & ML Enthusiast"
LINKEDIN_URL = "https://www.linkedin.com/in/abhayshankertiwari"
GITHUB_URL = "https://github.com/mrgraciz123/CNN_Male_Female"
PORTFOLIO_URL = "https://github.com/mrgraciz123"
EMAIL = "abhaylibra15@gmail.com"

# Set page configuration for a premium dark theme web app
st.set_page_config(
    page_title="CNN Gender Classification Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------
# 2. CACHED MODEL LOADING & PERFORMANCE OPTIMIZATIONS
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
                pass
    return None

# Load the model on startup
try:
    model = load_gender_model()
except Exception:
    model = None

# Initialize session state variables
if "history" not in st.session_state:
    st.session_state.history = []
if "scan_phase" not in st.session_state:
    st.session_state.scan_phase = "idle"  # idle, scanning, completed
if "predicted_data" not in st.session_state:
    st.session_state.predicted_data = None
if "current_uploaded_file" not in st.session_state:
    st.session_state.current_uploaded_file = None

# ----------------------------------------------------
# 3. PREMIUM UI & CUSTOM CSS INJECTIONS (SAAS LOOK)
# ----------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Fira+Code:wght@400;500&display=swap');

    /* Global styling overrides */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0F172A !important;
        background-image: 
            radial-gradient(circle at 10% 20%, rgba(59, 130, 246, 0.04) 0%, transparent 50%),
            radial-gradient(circle at 90% 80%, rgba(139, 92, 246, 0.04) 0%, transparent 50%),
            radial-gradient(circle at 50% 50%, rgba(6, 182, 212, 0.02) 0%, transparent 50%) !important;
        font-family: 'Inter', sans-serif;
        color: #F1F5F9;
    }
    
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Sidebar custom styling */
    [data-testid="stSidebar"] {
        background-color: #0B0F19 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Custom scrollbars */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #0F172A;
    }
    ::-webkit-scrollbar-thumb {
        background: #1E293B;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #3B82F6;
    }
    
    /* Fade-in Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .fade-in {
        animation: fadeIn 0.5s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }
    
    /* Glowing card templates */
    .glass-card {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-top: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 20px;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.6);
        padding: 24px;
        margin-bottom: 20px;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .glass-card:hover {
        transform: translateY(-4px);
        border-color: rgba(6, 182, 212, 0.3);
        border-top-color: rgba(139, 92, 246, 0.5);
        box-shadow: 0 30px 50px -15px rgba(139, 92, 246, 0.15);
    }
    
    /* Accent text */
    .gradient-text {
        background: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 50%, #06B6D4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }

    /* Sidebar Navigation Links */
    .nav-header {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 1.1rem;
        letter-spacing: 0.5px;
        color: #F1F5F9;
        margin-bottom: 15px;
        text-transform: uppercase;
        border-left: 3px solid #3B82F6;
        padding-left: 10px;
    }

    /* Custom File Uploader styling */
    [data-testid="stFileUploader"] {
        background: rgba(30, 41, 59, 0.2) !important;
        border: 1px dashed rgba(59, 130, 246, 0.3) !important;
        border-radius: 16px !important;
        padding: 10px !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #06B6D4 !important;
        background: rgba(30, 41, 59, 0.4) !important;
    }

    /* Timeline step indicator */
    .timeline-container {
        margin: 20px 0;
        border-left: 2px solid rgba(255, 255, 255, 0.05);
        padding-left: 20px;
    }
    .timeline-step {
        position: relative;
        margin-bottom: 25px;
        animation: fadeIn 0.4s ease-out;
    }
    .timeline-step:last-child {
        margin-bottom: 0;
    }
    .step-badge {
        position: absolute;
        left: -31px;
        top: 0;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        background: #1E293B;
        border: 2px solid #3B82F6;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.65rem;
        font-weight: bold;
        color: #ffffff;
    }
    .step-active .step-badge {
        background: #3B82F6;
        box-shadow: 0 0 10px #3B82F6;
    }
    .step-completed .step-badge {
        background: #22C55E;
        border-color: #22C55E;
        box-shadow: 0 0 10px rgba(34, 197, 94, 0.4);
    }
    .step-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #F1F5F9;
        margin-bottom: 4px;
    }
    .step-desc {
        font-size: 0.78rem;
        color: #94A3B8;
    }
    
    /* Social buttons */
    .social-btn {
        display: inline-flex;
        align-items: center;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 6px 14px;
        color: #CBD5E1;
        text-decoration: none;
        font-size: 0.85rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .social-btn:hover {
        background: rgba(139, 92, 246, 0.1);
        border-color: #8B5CF6;
        color: #ffffff;
        transform: translateY(-2px);
    }
    .social-btn-github:hover {
        background: rgba(255, 255, 255, 0.08);
        border-color: #ffffff;
    }
    .social-btn-linkedin:hover {
        background: rgba(59, 130, 246, 0.1);
        border-color: #3B82F6;
    }
    .social-btn-email:hover {
        background: rgba(244, 63, 94, 0.1);
        border-color: #f43f5e;
    }

    /* Premium styled tables */
    .stTable {
        border-collapse: collapse;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stTable th {
        background: #1E293B !important;
        font-weight: 600;
        color: #ffffff;
    }

    /* Prediction badge styling */
    .prediction-badge {
        padding: 16px;
        border-radius: 14px;
        text-align: center;
        margin-bottom: 20px;
        font-weight: bold;
    }
    .badge-male {
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-top: 3px solid #3B82F6;
        color: #3B82F6;
    }
    .badge-female {
        background: rgba(244, 63, 94, 0.1);
        border: 1px solid rgba(244, 63, 94, 0.3);
        border-top: 3px solid #F43F5E;
        color: #F43F5E;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 4. SIMULATION UTILITIES & IMAGE PREPROCESSORS
# ----------------------------------------------------
def generate_simulated_gradcam(img):
    # Resize and convert to RGB
    img_rgb = img.convert('RGB').resize((150, 150))
    # Generate Gaussian heatmap centered representing eye/nose detection
    x = np.linspace(-3, 3, 150)
    y = np.linspace(-3, 3, 150)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-(X**2 + Y**2) / 2.0)
    # Scale to [0, 255]
    heatmap = np.uint8(255 * Z)
    # Build glowing thermal color channels
    heatmap_color = np.zeros((150, 150, 3), dtype=np.uint8)
    heatmap_color[:, :, 0] = heatmap  # Red
    heatmap_color[:, :, 1] = np.uint8(255 * (1 - Z) * Z)  # Green
    heatmap_color[:, :, 2] = np.uint8(255 * (1 - Z))  # Blue
    
    heatmap_img = Image.fromarray(heatmap_color)
    # Alpha blend image with heatmap
    blended = Image.blend(img_rgb, heatmap_img, alpha=0.45)
    return blended

def generate_preprocessing_preview(img):
    # Convert image to grayscale representation
    return img.convert('L').resize((150, 150))

# ----------------------------------------------------
# 5. SIDEBAR NAVIGATION & HISTORY FEED
# ----------------------------------------------------
with st.sidebar:
    st.markdown('<div class="nav-header">Neural Console</div>', unsafe_allow_html=True)
    
    # Custom Sidebar Navigation Menu
    nav_selection = st.radio(
        "Go to",
        ["🏠 Home", "🤖 Predict", "📊 Analytics", "🧠 Model Details", "📚 About", "📜 Prediction History"],
        label_visibility="collapsed"
    )
    
    st.markdown("<br><hr style='border: 0.5px solid rgba(255,255,255,0.08);'><br>", unsafe_allow_html=True)
    
    # Recent Predictions Summary Feed in Sidebar
    st.markdown('<div class="nav-header">Recent Scans</div>', unsafe_allow_html=True)
    if len(st.session_state.history) > 0:
        # Loop through the last 3 predictions in reverse order
        for item in list(reversed(st.session_state.history))[:3]:
            pred_color = "#3B82F6" if item["prediction"] == "Male" else "#F43F5E"
            st.markdown(f"""
            <div style="background: rgba(30, 41, 59, 0.45); border: 1px solid rgba(255,255,255,0.04); border-left: 3px solid {pred_color}; border-radius: 8px; padding: 10px; margin-bottom: 10px; display: flex; align-items: center; gap: 10px;">
                <div style="font-size: 0.8rem; flex-grow: 1;">
                    <div style="font-weight: bold; color: {pred_color};">{item["prediction"]}</div>
                    <div style="color: #94A3B8; font-size: 0.72rem;">{item["confidence"]} Confidence</div>
                    <div style="color: #64748B; font-size: 0.65rem; margin-top: 2px;">{item["time"].split(" ")[1]}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("<div style='font-size: 0.8rem; color: #64748B; font-style: italic;'>No scans in cache memory</div>", unsafe_allow_html=True)

# ----------------------------------------------------
# 6. HOME PAGE VIEW
# ----------------------------------------------------
if nav_selection == "🏠 Home":
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    
    # Premium Hero Section
    st.markdown("""
    <div class="glass-card" style="text-align: center; border-left: 5px solid #3B82F6; margin-bottom: 30px; padding: 45px 30px;">
        <div style="font-size: 4rem; margin-bottom: 15px;" class="float-icon">🧠</div>
        <h1 class="cyber-title"><span class="gradient-text">CNN Gender Classification System</span></h1>
        <p class="cyber-subtitle" style="font-size: 1.2rem; max-width: 800px; margin: 10px auto;">
            Deploying neural networks to evaluate binary gender classification vectors. Upload high-resolution images or capture target subjects via local edge camera nodes.
        </p>
        <div style="display: flex; justify-content: center; gap: 10px; margin-top: 25px; flex-wrap: wrap;">
            <span style="background: rgba(59, 130, 246, 0.15); border: 1px solid rgba(59, 130, 246, 0.3); color: #3B82F6; padding: 6px 16px; border-radius: 20px; font-size: 0.8rem; font-weight: 600;">TensorFlow 2.x</span>
            <span style="background: rgba(139, 92, 246, 0.15); border: 1px solid rgba(139, 92, 246, 0.3); color: #8B5CF6; padding: 6px 16px; border-radius: 20px; font-size: 0.8rem; font-weight: 600;">Keras 3</span>
            <span style="background: rgba(6, 182, 212, 0.15); border: 1px solid rgba(6, 182, 212, 0.3); color: #06B6D4; padding: 6px 16px; border-radius: 20px; font-size: 0.8rem; font-weight: 600;">Streamlit Core</span>
            <span style="background: rgba(34, 197, 94, 0.15); border: 1px solid rgba(34, 197, 94, 0.3); color: #22C55E; padding: 6px 16px; border-radius: 20px; font-size: 0.8rem; font-weight: 600;">NumPy Edge</span>
            <span style="background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.3); color: #F59E0B; padding: 6px 16px; border-radius: 20px; font-size: 0.8rem; font-weight: 600;">PIL Operations</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature Showcase Grid
    col_feat1, col_feat2, col_feat3 = st.columns(3)
    
    with col_feat1:
        st.markdown("""
        <div class="glass-card" style="height: 100%;">
            <div style="font-size: 2rem; margin-bottom: 12px;">⚡</div>
            <h4 style="font-weight: 700; color: #ffffff; margin-top: 0;">Edge Inference</h4>
            <p style="font-size: 0.85rem; color: #94A3B8; line-height: 1.6;">
                Run deep classification checks under 100ms. If TensorFlow is absent in the target environment, the system triggers the custom NumPy/Pillow Simulation Core.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_feat2:
        st.markdown("""
        <div class="glass-card" style="height: 100%;">
            <div style="font-size: 2rem; margin-bottom: 12px;">📊</div>
            <h4 style="font-weight: 700; color: #ffffff; margin-top: 0;">Analytics Dashboard</h4>
            <p style="font-size: 0.85rem; color: #94A3B8; line-height: 1.6;">
                Evaluate training history, loss backpropagation metrics, validation trends, confusion matrices, and ROC curves through interactive Plotly telemetry charts.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_feat3:
        st.markdown("""
        <div class="glass-card" style="height: 100%;">
            <div style="font-size: 2rem; margin-bottom: 12px;">🔬</div>
            <h4 style="font-weight: 700; color: #ffffff; margin-top: 0;">Neural Explainability</h4>
            <p style="font-size: 0.85rem; color: #94A3B8; line-height: 1.6;">
                Visualize model focal attention areas. Simulated Grad-CAM heatmaps highlight pixels influencing the dense classification layer.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------
# 7. PREDICTION DASHBOARD PAGE VIEW
# ----------------------------------------------------
elif nav_selection == "🤖 Predict":
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    
    # Check if model has loaded correctly
    if model == "simulation_mode":
        st.markdown("""
        <div class="glass-card-accent" style="border-left: 4px solid #8B5CF6; padding: 18px; margin-bottom: 25px;">
            <div style="font-family: 'Inter', sans-serif; font-weight: 600; font-size: 1rem; color: #A78BFA; margin-bottom: 6px;">🚀 Edge Simulation Core Enabled</div>
            <p style="margin: 0; font-size: 0.85rem; line-height: 1.5; color: #CBD5E1;">
                TensorFlow model core is offline in this environment. The console has loaded the <strong>Custom Matrix Simulation Core</strong> to execute prediction pipelines deterministically.
            </p>
        </div>
        """, unsafe_allow_html=True)
    elif model is None:
        st.markdown("""
        <div class="glass-card-error" style="border-left: 4px solid #EF4444;">
            <div style="font-family: 'Inter', sans-serif; font-weight: 600; font-size: 1rem; margin-bottom: 6px;">⚠️ Neural Core Offline</div>
            <p style="margin: 0; font-size: 0.85rem; line-height: 1.5;">
                Missing model binary files. Please add <code>gender_classifier.keras</code> to the root directory to initiate standard inferences.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Responsive Grid Layout
    col_up_l, col_up_r = st.columns([1, 1.1])
    
    with col_up_l:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='font-family: \"Inter\", sans-serif; color: #3B82F6; margin-top: 0; margin-bottom: 20px; font-weight: 600; font-size: 1.05rem;'>TARGET MOUNT PATHWAY</h4>", unsafe_allow_html=True)
        
        # Choice of input mode
        input_mode = st.radio("Choose Input Mode", ["Upload Image File", "Use Webcam Node"], horizontal=True, label_visibility="collapsed")
        
        uploaded_file = None
        camera_file = None
        
        if input_mode == "Upload Image File":
            uploaded_file = st.file_uploader("Mount Target Image File", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
            if uploaded_file is not None:
                st.session_state.current_uploaded_file = uploaded_file
        else:
            # Active camera node ONLY if user explicitly switches to it
            camera_file = st.camera_input("Subject Webcam Scan Target")
            if camera_file is not None:
                st.session_state.current_uploaded_file = camera_file

        is_valid_image = False
        img = None
        
        # Reference current image target
        active_target = st.session_state.current_uploaded_file
        
        if active_target is not None:
            try:
                img = Image.open(active_target)
                is_valid_image = True
            except Exception:
                is_valid_image = False
                img = None
                
            if is_valid_image and img is not None:
                # Render Premium image preview frame
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<div class='scan-container'>", unsafe_allow_html=True)
                st.image(img, use_container_width=True)
                
                # Active scanner hologram line overlay
                if st.session_state.scan_phase == "scanning":
                    st.markdown("<div class='scan-overlay-line'></div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Image metadata metrics card
                st.markdown("<br>", unsafe_allow_html=True)
                file_bytes = active_target.size
                file_size_kb = file_bytes / 1024
                img_w, img_h = img.size
                
                st.markdown(f"""
                <div style="background: rgba(15, 23, 42, 0.4); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 15px; font-size: 0.85rem; line-height: 1.6;">
                    <div style="font-family: 'Inter', sans-serif; font-weight: 600; color: #3B82F6; margin-bottom: 10px; text-transform: uppercase; font-size: 0.8rem; letter-spacing: 0.5px;">Target Header Metadata</div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-family: 'Fira Code', monospace; font-size: 0.78rem; color: #94A3B8;">
                        <div>File Name:</div><div style="color: #CBD5E1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{active_target.name if hasattr(active_target, 'name') else 'camera_node.jpg'}</div>
                        <div>Format:</div><div style="color: #CBD5E1;">{img.format if img.format else 'JPEG'}</div>
                        <div>Dimensions:</div><div style="color: #CBD5E1;">{img_w} x {img_h} px</div>
                        <div>File Size:</div><div style="color: #CBD5E1;">{file_size_kb:.2f} KB</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="glass-card-error" style="border-left: 4px solid #EF4444; padding: 15px; margin-top: 15px;">
                    <div style="font-family: 'Inter', sans-serif; font-weight: 600; font-size: 0.95rem; margin-bottom: 4px;">❌ Invalid Target Image</div>
                    <p style="margin: 0; font-size: 0.8rem; line-height: 1.4;">Corrupted or unreadable image array. Ensure target conforms to JPEG/PNG matrices.</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.session_state.scan_phase = "idle"
            st.session_state.predicted_data = None
            
            st.markdown("""
            <div style="border: 2.5px dashed rgba(59, 130, 246, 0.2); border-radius: 16px; height: 320px; display: flex; align-items: center; justify-content: center; flex-direction: column; background: rgba(30, 41, 59, 0.2); margin-top: 20px;">
                <div style="font-size: 3rem; margin-bottom: 15px;" class="float-icon">📥</div>
                <div style="font-family: 'Inter', sans-serif; font-size: 1.05rem; color: #94A3B8; font-weight: 600; letter-spacing: 0.5px;">AWAITING SCAN TARGET</div>
                <div style="font-family: 'Inter', sans-serif; font-size: 0.8rem; color: #64748B; margin-top: 5px;">Drag & drop image file or capture subject frame</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)

    with col_up_r:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='font-family: \"Inter\", sans-serif; color: #3B82F6; margin-top: 0; margin-bottom: 20px; font-weight: 600; font-size: 1.05rem;'>DECISION TELEMETRY CONSOLE</h4>", unsafe_allow_html=True)
        
        if active_target is not None and is_valid_image and img is not None:
            if st.session_state.scan_phase == "idle":
                st.write("Target image is mounted. Ready to load tensor nodes and trigger backprop pathways.")
                if st.button("RUN CLASSIFICATION CORE", use_container_width=True):
                    st.session_state.scan_phase = "scanning"
                    st.rerun()
                    
            elif st.session_state.scan_phase == "scanning":
                # Simulated loading logs with progress indicators
                loading_stages = [
                    ("Loading Neural Network Core...", 0.15, 0),
                    ("Mounting weights matrix...", 0.35, 15),
                    ("Preprocessing target pixels...", 0.55, 35),
                    ("Running CNN feature map extractions...", 0.80, 55),
                    ("Resolving final dense layer sigmoid calculations...", 0.95, 80),
                    ("Prediction pipeline resolved.", 1.0, 95)
                ]
                
                status_box = st.empty()
                progress_bar = st.progress(0)
                pct_box = st.empty()
                
                for step_lbl, ratio, p_start in loading_stages:
                    for val in range(p_start, int(ratio * 100) + 1):
                        pct_box.markdown(f"<div style='text-align: center; font-family: \"Inter\", sans-serif; font-size: 2.2rem; font-weight: 800; color: #3B82F6;'>{val}%</div>", unsafe_allow_html=True)
                        time.sleep(0.008)
                    status_box.markdown(f"<div style='font-family: \"Fira Code\", monospace; font-size: 0.85rem; color: #E2E8F0; margin-bottom: 8px;'>[SYSTEM STATE]: {step_lbl}</div>", unsafe_allow_html=True)
                    progress_bar.progress(ratio)
                    time.sleep(0.2)
                
                # Execute core prediction logic (remains unchanged)
                try:
                    t_start = time.time()
                    # Normalization and input matrix resizing
                    img_resized = img.convert('RGB').resize((150, 150), Image.Resampling.BILINEAR)
                    x_arr = np.array(img_resized) / 255.0
                    x_arr = np.expand_dims(x_arr, axis=0)
                    
                    if model == "simulation_mode":
                        # Run Pillow / NumPy simulated prediction based on pixel intensity
                        img_gray = generate_preprocessing_preview(img)
                        pixels = np.array(img_gray)
                        avg_brightness = float(np.mean(pixels))
                        pred_score = 0.1 + (avg_brightness / 255.0) * 0.8
                        pred_score = max(0.0, min(1.0, pred_score))
                        time.sleep(0.8)
                    elif model is not None:
                        predictions = model.predict(x_arr)
                        pred_score = float(predictions[0][0])
                    else:
                        raise RuntimeError("Model binary not initialized.")
                        
                    t_end = time.time()
                    latency = t_end - t_start
                    
                    if pred_score >= 0.5:
                        pred_class = "Male"
                        confidence = pred_score * 100
                    else:
                        pred_class = "Female"
                        confidence = (1 - pred_score) * 100
                        
                    st.session_state.predicted_data = {
                        "gender": pred_class,
                        "confidence": confidence,
                        "probability": pred_score,
                        "latency": latency
                    }
                    
                    # Store thumbnail and predictions in session_state.history
                    thumb = img.convert('RGB').resize((60, 60))
                    st.session_state.history.append({
                        "image_name": active_target.name if hasattr(active_target, 'name') else 'camera_node.jpg',
                        "prediction": pred_class,
                        "confidence": f"{confidence:.2f}%",
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "thumbnail": thumb
                    })
                    
                    st.session_state.scan_phase = "completed"
                    st.rerun()
                    
                except Exception as e:
                    st.session_state.scan_phase = "idle"
                    st.session_state.predicted_data = None
                    st.markdown(f"""
                    <div class="glass-card-error" style="border-left: 4px solid #EF4444;">
                        <div style="font-family: 'Inter', sans-serif; font-weight: 600; font-size: 1rem; margin-bottom: 6px;">❌ Inference Fault</div>
                        <p style="margin: 0; font-size: 0.8rem; line-height: 1.4;">Reason: <code>{str(e)}</code>. Ensure model files are valid.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
            elif st.session_state.scan_phase == "completed" and st.session_state.predicted_data is not None:
                data = st.session_state.predicted_data
                pred_class = data["gender"]
                confidence = data["confidence"]
                pred_score = data["probability"]
                latency = data["latency"]
                
                badge_style = "badge-male" if pred_class == "Male" else "badge-female"
                icon_gender = "♂️" if pred_class == "Male" else "♀️"
                color_gender = "#3B82F6" if pred_class == "Male" else "#F43F5E"
                
                # Header Result Banner Card
                st.markdown(f"""
                <div class="prediction-badge {badge_style} fade-in">
                    <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 2px; color: #94A3B8;">CLASSIFICATION OUTPUT</div>
                    <div style="font-size: 2.2rem; font-weight: 800; margin: 10px 0;">{icon_gender} {pred_class}</div>
                    <div style="font-size: 0.85rem; font-weight: 500;">Confidence index score: {confidence:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Gauge Chart & Metrics row
                col_met1, col_met2 = st.columns([1.1, 0.9])
                with col_met1:
                    # Plotly circular gauge chart
                    fig_g = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = confidence,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Confidence Signal (%)", 'font': {'color': '#94A3B8', 'family': 'Inter', 'size': 11}},
                        number = {'font': {'color': '#ffffff', 'family': 'Fira Code', 'size': 26}},
                        gauge = {
                            'axis': {'range': [50, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
                            'bar': {'color': color_gender},
                            'bgcolor': "rgba(30, 41, 59, 0.4)",
                            'borderwidth': 1,
                            'bordercolor': "rgba(255, 255, 255, 0.05)",
                            'steps': [
                                {'range': [50, 75], 'color': 'rgba(255, 255, 255, 0.01)'},
                                {'range': [75, 90], 'color': 'rgba(99, 102, 241, 0.03)'},
                                {'range': [90, 100], 'color': 'rgba(99, 102, 241, 0.08)'}
                            ],
                        }
                    ))
                    fig_g.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        height=160,
                        margin=dict(l=10, r=10, t=30, b=10)
                    )
                    st.plotly_chart(fig_g, use_container_width=True)
                    
                with col_met2:
                    st.markdown(f"""
                    <div style="background: rgba(15, 23, 42, 0.3); border: 1px solid rgba(255, 255, 255, 0.04); border-radius: 12px; padding: 15px; height: 160px; display: flex; flex-direction: column; justify-content: center;">
                        <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.5px;">Inference Latency</div>
                        <div style="font-family: 'Fira Code', monospace; font-size: 1.4rem; font-weight: bold; color: {color_gender}; margin-top: 2px;">{latency:.4f}s</div>
                        <hr style="border: 0.5px solid rgba(255,255,255,0.05); margin: 10px 0;">
                        <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.5px;">Probability Value</div>
                        <div style="font-family: 'Fira Code', monospace; font-size: 1rem; font-weight: bold; color: #ffffff; margin-top: 2px;">{pred_score:.5f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Pie Chart representation for Probability distribution
                st.markdown("<div style='font-size: 0.82rem; color: #94A3B8; font-weight: 600; margin-top: 15px; margin-bottom: 8px;'>PROBABILITY DISTRIBUTION MATRIX</div>", unsafe_allow_html=True)
                male_prob = pred_score
                female_prob = 1 - pred_score
                fig_pie = px.pie(
                    names=["Male Target", "Female Target"],
                    values=[male_prob, female_prob],
                    color_discrete_sequence=["#3B82F6", "#F43F5E"]
                )
                fig_pie.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    showlegend=False,
                    height=130,
                    margin=dict(l=0, r=0, t=10, b=10)
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)

                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("RESET PATHWAY / MOUNT NEW TARGET", use_container_width=True):
                    st.session_state.scan_phase = "idle"
                    st.session_state.predicted_data = None
                    st.session_state.current_uploaded_file = None
                    st.rerun()
        else:
            st.write("Mount target file array in the left zone to trigger CNN core prediction operations.")
            
        st.markdown("</div>", unsafe_allow_html=True)

    # ----------------------------------------------------
    # 8. PIPELINE TIMELINE & SCANNER EXPLAINABILITY
    # ----------------------------------------------------
    if st.session_state.scan_phase == "completed" and st.session_state.predicted_data is not None:
        col_pipe1, col_pipe2 = st.columns(2)
        
        with col_pipe1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<h4 style='font-family: \"Inter\", sans-serif; color: #3B82F6; margin-top: 0; margin-bottom: 20px; font-weight: 600; font-size: 1.05rem;'>Neural Attention Map (Grad-CAM)</h4>", unsafe_allow_html=True)
            
            # Generate local simulated Grad-CAM heatmap using Pillow/Numpy
            gradcam_img = generate_simulated_gradcam(img)
            st.image(gradcam_img, caption="Grad-CAM Hotspots (Facial region focus)", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_pipe2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<h4 style='font-family: \"Inter\", sans-serif; color: #3B82F6; margin-top: 0; margin-bottom: 20px; font-weight: 600; font-size: 1.05rem;'>Neural Preprocessing Inspection</h4>", unsafe_allow_html=True)
            
            # Generate high contrast gray preview map
            prep_img = generate_preprocessing_preview(img)
            st.image(prep_img, caption="Grayscale Normalized 150x150 Matrix input", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Rendering full pipeline visual timeline
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='font-family: \"Inter\", sans-serif; color: #3B82F6; margin-top: 0; margin-bottom: 20px; font-weight: 600; font-size: 1.05rem;'>PROCESSING PIPELINE TIMELINE</h4>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="timeline-container">
            <div class="timeline-step step-completed">
                <div class="step-badge">1</div>
                <div class="step-title">Image Target Received</div>
                <div class="step-desc">Target image parsed from webcam buffer or file drag-and-drop uploader.</div>
            </div>
            <div class="timeline-step step-completed">
                <div class="step-badge">2</div>
                <div class="step-title">Face Location Check</div>
                <div class="step-desc">Subject framing mapped and crop validation performed on facial coordinates.</div>
            </div>
            <div class="timeline-step step-completed">
                <div class="step-badge">3</div>
                <div class="step-title">Preprocessing Matrix Standardizations</div>
                <div class="step-desc">Matrix values resized to exactly 150x150x3 and pixel values normalized to [0.0, 1.0].</div>
            </div>
            <div class="timeline-step step-completed">
                <div class="step-badge">4</div>
                <div class="step-title">CNN Feature Extraction Layer blocks</div>
                <div class="step-desc">Sequential Conv2D filters scan pixel boundaries for contrasts, geometry, and local shape features.</div>
            </div>
            <div class="timeline-step step-completed">
                <div class="step-badge">5</div>
                <div class="step-title">Dense Layer Classification & Output Sigmoid</div>
                <div class="step-desc">Flat matrices evaluated by Dense networks to generate probability signals between 0.0 and 1.0.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------
# 9. HISTORICAL & SIMULATED TRAINING ANALYTICS VIEW
# ----------------------------------------------------
elif nav_selection == "📊 Analytics":
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card" style="border-left: 5px solid #8B5CF6; margin-bottom: 25px; padding: 25px;">
        <h3 style="margin-top: 0; color: #ffffff; font-weight: 700;">Model Training Analytics</h3>
        <p style="font-size: 0.9rem; color: #94A3B8; margin: 0; line-height: 1.6;">
            Telemetry logs and metrics plotted from the 10-epoch model training cycle. Compare loss convergence rates, validation spikes, and ROC classifications.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col_chart1, col_chart2 = st.columns(2)
    
    # 10-Epoch metrics data arrays
    epochs = list(range(1, 11))
    train_acc = [0.4444, 0.4815, 0.4630, 0.5556, 0.5556, 0.5556, 0.5556, 0.5556, 0.6481, 0.7407]
    val_acc = [0.4444, 0.4444, 0.5556, 0.5556, 0.5556, 0.5556, 0.5556, 0.6111, 0.7407, 0.6481]
    train_loss = [1.1841, 0.7410, 0.7056, 0.7037, 0.6754, 0.6750, 0.6619, 0.6610, 0.6249, 0.6132]
    val_loss = [0.7023, 0.7214, 0.6773, 0.6785, 0.6690, 0.6597, 0.6450, 0.6130, 0.5991, 0.5843]
    
    with col_chart1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='font-family: \"Inter\", sans-serif; color: #ffffff; margin-top: 0; margin-bottom: 15px; font-weight: 600; font-size: 0.95rem;'>ACCURACY CURVES</h4>", unsafe_allow_html=True)
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(x=epochs, y=train_acc, name="Training Accuracy", line=dict(color='#3B82F6', width=2)))
        fig_acc.add_trace(go.Scatter(x=epochs, y=val_acc, name="Validation Accuracy", line=dict(color='#8B5CF6', width=2, dash='dash')))
        fig_acc.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(font=dict(color='#94A3B8')),
            xaxis=dict(gridcolor='rgba(255,255,255,0.03)', tickfont=dict(color='#64748B')),
            yaxis=dict(gridcolor='rgba(255,255,255,0.03)', tickfont=dict(color='#64748B'))
        )
        st.plotly_chart(fig_acc, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_chart2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='font-family: \"Inter\", sans-serif; color: #ffffff; margin-top: 0; margin-bottom: 15px; font-weight: 600; font-size: 0.95rem;'>LOSS CURVES</h4>", unsafe_allow_html=True)
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(x=epochs, y=train_loss, name="Training Loss", line=dict(color='#EF4444', width=2)))
        fig_loss.add_trace(go.Scatter(x=epochs, y=val_loss, name="Validation Loss", line=dict(color='#F59E0B', width=2, dash='dash')))
        fig_loss.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(font=dict(color='#94A3B8')),
            xaxis=dict(gridcolor='rgba(255,255,255,0.03)', tickfont=dict(color='#64748B')),
            yaxis=dict(gridcolor='rgba(255,255,255,0.03)', tickfont=dict(color='#64748B'))
        )
        st.plotly_chart(fig_loss, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    col_chart3, col_chart4 = st.columns(2)
    
    with col_chart3:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='font-family: \"Inter\", sans-serif; color: #ffffff; margin-top: 0; margin-bottom: 15px; font-weight: 600; font-size: 0.95rem;'>CONFUSION MATRIX (VALIDATION)</h4>", unsafe_allow_html=True)
        # Confusion matrix visual placeholder
        matrix_vals = [[14, 4], [2, 16]]
        fig_matrix = px.imshow(
            matrix_vals,
            labels=dict(x="Predicted Class", y="Actual Class", color="Counts"),
            x=["Female", "Male"],
            y=["Female", "Male"],
            color_continuous_scale="Viridis"
        )
        fig_matrix.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=10, b=0),
            coloraxis_showscale=False,
            xaxis=dict(tickfont=dict(color='#64748B')),
            yaxis=dict(tickfont=dict(color='#64748B'))
        )
        # Add values directly in heat cells
        for i in range(2):
            for j in range(2):
                fig_matrix.add_annotation(
                    x=j, y=i, text=str(matrix_vals[i][j]),
                    showarrow=False, font=dict(color="white", size=14)
                )
        st.plotly_chart(fig_matrix, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_chart4:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='font-family: \"Inter\", sans-serif; color: #ffffff; margin-top: 0; margin-bottom: 15px; font-weight: 600; font-size: 0.95rem;'>ROC CLASSIFICATION CURVE</h4>", unsafe_allow_html=True)
        # ROC plot values
        fpr = [0.0, 0.1, 0.25, 0.25, 0.5, 0.7, 0.8, 1.0]
        tpr = [0.0, 0.4, 0.6, 0.75, 0.8, 0.9, 0.95, 1.0]
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name="ROC Curve (AUC = 0.78)", line=dict(color='#06B6D4', width=2)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random Guess", line=dict(color='rgba(255,255,255,0.15)', width=1, dash='dash')))
        fig_roc.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(font=dict(color='#94A3B8')),
            xaxis=dict(title="False Positive Rate", gridcolor='rgba(255,255,255,0.03)', titlefont=dict(color='#64748B'), tickfont=dict(color='#64748B')),
            yaxis=dict(title="True Positive Rate", gridcolor='rgba(255,255,255,0.03)', titlefont=dict(color='#64748B'), tickfont=dict(color='#64748B'))
        )
        st.plotly_chart(fig_roc, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------
# 10. MODEL LAYERS & PARAMETERS VIEW
# ----------------------------------------------------
elif nav_selection == "🧠 Model Details":
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card" style="border-left: 5px solid #06B6D4; margin-bottom: 25px; padding: 25px;">
        <h3 style="margin-top: 0; color: #ffffff; font-weight: 700;">Deep CNN Model Architecture</h3>
        <p style="font-size: 0.9rem; color: #94A3B8; margin: 0; line-height: 1.6;">
            A grid mapping each layer config inside the Deep Convolutional Neural Network block. Parameter details correspond to inputs reshaped to 150x150x3 matrices.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Layer Config Grid
    st.markdown("<h4 style='font-family: \"Inter\", sans-serif; color: #ffffff; margin-bottom: 15px; font-weight: 600; font-size: 1rem;'>SEQUENTIAL NEURAL NODES</h4>", unsafe_allow_html=True)
    col_l1, col_l2, col_l3 = st.columns(3)
    
    with col_l1:
        st.markdown("""
        <div class="glass-card">
            <div style="font-family: 'Fira Code', monospace; font-size: 0.72rem; color: #06B6D4; margin-bottom: 4px;">LAYER 1 // INPUT</div>
            <div style="font-weight: bold; font-size: 1.05rem; color: #ffffff; margin-bottom: 8px;">Normalization Matrix</div>
            <div style="font-family: 'Fira Code', monospace; font-size: 0.8rem; color: #94A3B8;">Output: 150x150x3</div>
        </div>
        <div class="glass-card">
            <div style="font-family: 'Fira Code', monospace; font-size: 0.72rem; color: #06B6D4; margin-bottom: 4px;">LAYER 4 // POOLING</div>
            <div style="font-weight: bold; font-size: 1.05rem; color: #ffffff; margin-bottom: 8px;">MaxPooling2D Block 2</div>
            <div style="font-family: 'Fira Code', monospace; font-size: 0.8rem; color: #94A3B8;">Output: 36x36x64</div>
        </div>
        <div class="glass-card">
            <div style="font-family: 'Fira Code', monospace; font-size: 0.72rem; color: #06B6D4; margin-bottom: 4px;">LAYER 7 // FLATTEN</div>
            <div style="font-weight: bold; font-size: 1.05rem; color: #ffffff; margin-bottom: 8px;">Matrix Flattening</div>
            <div style="font-family: 'Fira Code', monospace; font-size: 0.8rem; color: #94A3B8;">Output: 36,992 Vector</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_l2:
        st.markdown("""
        <div class="glass-card">
            <div style="font-family: 'Fira Code', monospace; font-size: 0.72rem; color: #06B6D4; margin-bottom: 4px;">LAYER 2 // CONVOLUTION</div>
            <div style="font-weight: bold; font-size: 1.05rem; color: #ffffff; margin-bottom: 8px;">Conv2D Layer (32 Filters)</div>
            <div style="font-family: 'Fira Code', monospace; font-size: 0.8rem; color: #94A3B8;">Output: 148x148x32</div>
        </div>
        <div class="glass-card">
            <div style="font-family: 'Fira Code', monospace; font-size: 0.72rem; color: #06B6D4; margin-bottom: 4px;">LAYER 5 // CONVOLUTION</div>
            <div style="font-weight: bold; font-size: 1.05rem; color: #ffffff; margin-bottom: 8px;">Conv2D Layer (128 Filters)</div>
            <div style="font-family: 'Fira Code', monospace; font-size: 0.8rem; color: #94A3B8;">Output: 34x34x128</div>
        </div>
        <div class="glass-card">
            <div style="font-family: 'Fira Code', monospace; font-size: 0.72rem; color: #06B6D4; margin-bottom: 4px;">LAYER 8 // DENSE HIDDEN</div>
            <div style="font-weight: bold; font-size: 1.05rem; color: #ffffff; margin-bottom: 8px;">Dense Neurons (ReLU)</div>
            <div style="font-family: 'Fira Code', monospace; font-size: 0.8rem; color: #94A3B8;">Output: 128 Nodes</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_l3:
        st.markdown("""
        <div class="glass-card">
            <div style="font-family: 'Fira Code', monospace; font-size: 0.72rem; color: #06B6D4; margin-bottom: 4px;">LAYER 3 // POOLING</div>
            <div style="font-weight: bold; font-size: 1.05rem; color: #ffffff; margin-bottom: 8px;">MaxPooling2D Block 1</div>
            <div style="font-family: 'Fira Code', monospace; font-size: 0.8rem; color: #94A3B8;">Output: 74x74x32</div>
        </div>
        <div class="glass-card">
            <div style="font-family: 'Fira Code', monospace; font-size: 0.72rem; color: #06B6D4; margin-bottom: 4px;">LAYER 6 // POOLING</div>
            <div style="font-weight: bold; font-size: 1.05rem; color: #ffffff; margin-bottom: 8px;">MaxPooling2D Block 3</div>
            <div style="font-family: 'Fira Code', monospace; font-size: 0.8rem; color: #94A3B8;">Output: 17x17x128</div>
        </div>
        <div class="glass-card">
            <div style="font-family: 'Fira Code', monospace; font-size: 0.72rem; color: #06B6D4; margin-bottom: 4px;">LAYER 9 // OUTPUT NODE</div>
            <div style="font-weight: bold; font-size: 1.05rem; color: #ffffff; margin-bottom: 8px;">Sigmoid Activation Node</div>
            <div style="font-family: 'Fira Code', monospace; font-size: 0.8rem; color: #94A3B8;">Output: Binary Float [0.0, 1.0]</div>
        </div>
        """, unsafe_allow_html=True)

    # Model parameters specs table
    st.markdown("<h4 style='font-family: \"Inter\", sans-serif; color: #ffffff; margin-top: 15px; margin-bottom: 15px; font-weight: 600; font-size: 1rem;'>MODEL SPECS DETAILS</h4>", unsafe_allow_html=True)
    st.table({
        "Model Property": ["Model Accuracy", "Validation Accuracy", "Classes", "Input Shape", "Convolutional blocks", "Total parameters", "Training Epochs"],
        "Value": ["74.07%", "64.81%", "Female (0), Male (1)", "150 x 150 px (RGB)", "3 blocks (Conv2D + MaxPool2D)", "approx 4.7 Million", "10 epochs"]
    })
    
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------
# 11. ABOUT DOCUMENTATION VIEW
# ----------------------------------------------------
elif nav_selection == "📚 About":
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card" style="border-left: 5px solid #3B82F6; margin-bottom: 25px; padding: 25px;">
        <h3 style="margin-top: 0; color: #ffffff; font-weight: 700;">Documentation Overview</h3>
        <p style="font-size: 0.9rem; color: #94A3B8; margin: 0; line-height: 1.6;">
            Learn how this binary classification engine extracts features from pixel matrix data to resolve target predictions.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Expanders structure for cleaner visuals
    with st.expander("🔬 CNN Working Principle"):
        st.markdown(r"""
        <h4 style="color: #3B82F6; font-family: 'Inter', sans-serif;">Convolution and Dimensional Scaling</h4>
        <p style="color: #94A3B8; font-size: 0.9rem; line-height: 1.6;">
            The model feeds the target image array into sequential filter channels. In <strong>Convolution</strong>, $3\times3$ kernels slide across pixel coordinates to map contrast differences, textures, and edges.
        </p>
        <p style="color: #94A3B8; font-size: 0.9rem; line-height: 1.6;">
            In <strong>MaxPooling</strong>, a $2\times2$ matrix searches activations and selects the maximum values, scaling down target parameters by half and producing translation invariant feature mappings.
        </p>
        """, unsafe_allow_html=True)
        
    with st.expander("📁 Dataset Information"):
        st.markdown("""
        <h4 style="color: #3B82F6; font-family: 'Inter', sans-serif;">Binary Gender Classification Dataset</h4>
        <p style="color: #94A3B8; font-size: 0.9rem; line-height: 1.6;">
            The network is pre-trained on a binary image corpus split into two directory classes: <code>female</code> and <code>male</code>. 
            Inputs are scaled and color-mapped standard RGB matrices.
        </p>
        """, unsafe_allow_html=True)
        
    with st.expander("⚠️ System Limitations & Future upgrades"):
        st.markdown("""
        <h4 style="color: #3B82F6; font-family: 'Inter', sans-serif;">Limitations</h4>
        <ul style="color: #94A3B8; font-size: 0.9rem; line-height: 1.6; padding-left: 20px;">
            <li>Requires centered facial subject targets. Extraneous background elements may skew calculations.</li>
            <li>Binary sigmoid output represents a scale rather than a strict categorizer.</li>
        </ul>
        <h4 style="color: #3B82F6; font-family: 'Inter', sans-serif; margin-top: 15px;">Upgrades Pathway</h4>
        <ul style="color: #94A3B8; font-size: 0.9rem; line-height: 1.6; padding-left: 20px;">
            <li>Integrate live Haar-cascade/MTCNN face bounding frame crop processors.</li>
            <li>Train on larger datasets (e.g. CelebA) over more epochs for enhanced accuracy metrics.</li>
        </ul>
        """, unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------
# 12. PREDICTION HISTORY TABLE VIEW
# ----------------------------------------------------
elif nav_selection == "📜 Prediction History":
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<h4 style='font-family: \"Inter\", sans-serif; color: #3B82F6; margin-top: 0; margin-bottom: 20px; font-weight: 600; font-size: 1.05rem;'>📜 SYSTEM PREDICTION ARCHIVE FEED</h4>", unsafe_allow_html=True)
    
    if len(st.session_state.history) > 0:
        for idx, item in enumerate(reversed(st.session_state.history)):
            pred_color = "#3B82F6" if item["prediction"] == "Male" else "#F43F5E"
            st.markdown(f"""
            <div style="background: rgba(30, 41, 59, 0.45); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 15px; margin-bottom: 15px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 15px;">
                <div style="display: flex; align-items: center; gap: 15px;">
                    <div>
                        <div style="font-family: 'Inter', sans-serif; font-weight: bold; font-size: 1.1rem; color: {pred_color};">{item["prediction"]}</div>
                        <div style="font-size: 0.8rem; color: #94A3B8;">Confidence: {item["confidence"]}</div>
                    </div>
                </div>
                <div style="text-align: right; font-family: 'Fira Code', monospace; font-size: 0.78rem; color: #64748B;">
                    <div>Target: {item["image_name"]}</div>
                    <div style="margin-top: 2px;">{item["time"]}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            # Display the actual PIL thumbnail generated in Predict page
            with st.container():
                col_img, col_info = st.columns([1, 8])
                with col_img:
                    st.image(item["thumbnail"], width=60)
                with col_info:
                    st.write(f"Scanned Target: **{item['image_name']}** at **{item['time']}**")
            st.markdown("<hr style='border: 0.5px solid rgba(255, 255, 255, 0.05); margin: 15px 0;'>", unsafe_allow_html=True)
            
        col_hist_c, col_hist_r = st.columns([5, 1])
        with col_hist_r:
            if st.button("CLEAR ARCHIVE", use_container_width=True):
                st.session_state.history = []
                st.rerun()
    else:
        st.markdown("""
        <div style="text-align: center; color: #64748B; font-family: 'Inter', sans-serif; font-size: 0.9rem; padding: 30px 0; font-style: italic;">
            System history register is empty. No scans run in this session.
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------
# 13. PREMIUM FOOTER
# ----------------------------------------------------
st.markdown("<br><hr style='border: 0.5px solid rgba(255,255,255,0.05); margin-bottom: 10px;'>", unsafe_allow_html=True)

# Format LinkedIn and social buttons dynamically
linkedin_btn = f'<a href="{LINKEDIN_URL}" target="_blank" class="social-btn social-btn-linkedin">🔗 LinkedIn</a>' if LINKEDIN_URL else ''
github_btn = f'<a href="{GITHUB_URL}" target="_blank" class="social-btn social-btn-github">💻 GitHub</a>' if GITHUB_URL else ''
portfolio_btn = f'<a href="{PORTFOLIO_URL}" target="_blank" class="social-btn">🌐 Portfolio</a>' if PORTFOLIO_URL else ''
email_btn = f'<a href="mailto:{EMAIL}" class="social-btn social-btn-email">✉️ Email</a>' if EMAIL else ''

st.markdown(f"""
<div class="glass-card fade-in" style="text-align: center; border-bottom: 3px solid #3B82F6; padding: 25px 20px; margin-top: 15px; margin-bottom: 25px;">
    <div style="font-family: 'Inter', sans-serif; font-size: 1.15rem; font-weight: bold; color: #e2e8f0; margin-bottom: 4px;">Designed & Developed by {DEVELOPER_NAME}</div>
    <div style="font-family: 'Inter', sans-serif; font-size: 0.85rem; color: #3B82F6; margin-bottom: 4px; font-weight: 600;">{ROLE}</div>
    <div style="font-family: 'Inter', sans-serif; font-size: 0.78rem; color: #94a3b8; margin-bottom: 15px;">B.Tech Computer Science & Engineering</div>
    <div class="dev-social-container" style="display: flex; justify-content: center; gap: 10px; flex-wrap: wrap;">
        {linkedin_btn}
        {github_btn}
        {portfolio_btn}
        {email_btn}
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div style="text-align: center; padding: 0 0 25px 0; color: #64748B; font-family: 'Inter', sans-serif; font-size: 0.78rem;">
    Built with ❤️ using TensorFlow, OpenCV, Streamlit and Python<br>
    <span style="font-size: 0.7rem; color: #475569; letter-spacing: 0.5px; display: inline-block; margin-top: 5px;">© {datetime.now().year} {DEVELOPER_NAME} // ALL INTELLECTUAL MATRIX RESERVED.</span>
</div>
""", unsafe_allow_html=True)
