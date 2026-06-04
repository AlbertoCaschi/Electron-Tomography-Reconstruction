import os
import streamlit as st
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from config import CONFIG
from models.vae import TomographyVAE
from inference import run_streamlit_inference 

# --------------------------------------------------------
# 1. Page Configuration & Theme
# --------------------------------------------------------
st.set_page_config(
    page_title="Electron Tomography Reconstruction",
    layout="wide", # UPGRADED: Changed to wide layout for desktop optimization
    initial_sidebar_state="collapsed"
)

# UPGRADED: Comprehensive modern CSS with animations and cards
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .subtitle { 
        font-size: 1.15rem; 
        color: #A0AEC0; 
        margin-bottom: 2rem; 
        line-height: 1.6; 
        animation: fadeIn 1.2s ease-in-out;
    }
    
    /* Modern Card Layouts for Text */
    .glass-card {
        background: rgba(17, 25, 40, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: fadeIn 1.5s ease-in-out;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
    }
    
    .section-header { 
        font-size: 1.4rem; 
        font-weight: 600; 
        color: #E2E8F0; 
        margin-bottom: 1rem; 
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    /* Animated Info Box */
    .info-box { 
        background: linear-gradient(90deg, rgba(30,58,138,0.3) 0%, rgba(15,23,42,0.1) 100%); 
        border-left: 4px solid #3B82F6; 
        padding: 1.2rem; 
        border-radius: 6px; 
        margin-bottom: 1.5rem; 
        color: #E2E8F0;
        animation: pulseBorder 2.5s infinite;
    }
            
    /* Digital Data Flow Wrapper & Animation */
    .title-wrapper {
        position: relative;
        overflow: hidden;
        padding: 10px 0; /* Breathing room for the animation */
        border-radius: 8px; /* Soft edges for the background */
    }

    /* The glowing data stream */
    .title-wrapper::before {
        content: '';
        position: absolute;
        top: 0;
        left: -20%; /* Start outside the left edge */
        width: 15%; /* Width of the beam */
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0, 242, 254, 0.15), rgba(79, 172, 254, 0.25), transparent);
        transform: skewX(-25deg); /* Angle it for a sense of speed */
        animation: dataSweep 1.8s cubic-bezier(0.25, 1, 0.5, 1) forwards; /* Plays exactly once */
        pointer-events: none; /* Prevents it from blocking text selection */
        z-index: 0;
    }

    /* Keep the text above the background animation */
    .main-title { 
        position: relative;
        z-index: 1;
        font-size: 2.8rem; 
        font-weight: 800; 
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem; 
        animation: fadeIn 1s ease-in-out;
    }

    @keyframes dataSweep {
        0% { left: -20%; opacity: 0; }
        10% { opacity: 1; }
        80% { opacity: 1; }
        100% { left: 120%; opacity: 0; }
    }
    
    /* Keyframe Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulseBorder {
        0% { border-left-color: #3B82F6; }
        50% { border-left-color: #60A5FA; box-shadow: -4px 0px 10px rgba(59, 130, 246, 0.2); }
        100% { border-left-color: #3B82F6; }
    }
    
    /* Style tweaks for Streamlit UI elements */
    div[data-testid="stRadio"] > label { font-weight: 600; color: #94A3B8; }
    div[data-testid="stSelectbox"] > label { font-weight: 600; color: #94A3B8; }
    </style>
""", unsafe_allow_html=True)

# --------------------------------------------------------
# 2. Secure Private Model Loading
# --------------------------------------------------------
@st.cache_resource
def load_private_model():
    """Fetches the private PyTorch model weights from Hugging Face."""
    try:
        token = st.secrets["HF_TOKEN"]
        
        model_path = hf_hub_download(
            repo_id="albertocaschi/VAEResNet_Tomography", 
            filename="VAEResNet.pth", 
            token=token
        )
        
        device = torch.device('cpu')
        checkpoint = torch.load(model_path, map_location=device)
        
        model = TomographyVAE(
            latent_dim=CONFIG["latent_dim"],
            target_size=CONFIG["target_size"],
            resnet_type=CONFIG["resnet_type"],
            freeze_early_layers=CONFIG["freeze_early_layers"]
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model
        
    except Exception as e:
        st.error(f"Failed to load model from Hugging Face. Error: {e}")
        return None

# Load model globally into cache
model = load_private_model()

# --------------------------------------------------------
# 3. Header Area (Full Width)
# --------------------------------------------------------
st.markdown("""
    <div class="title-wrapper">
        <div class="main-title">AI-Powered Electron Tomography</div>
    </div>
""", unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Bypassing traditional analytical reconstruction artifacts using a deep '
    'generative framework for high-fidelity 3D volumes from limited-angle data.</div>', 
    unsafe_allow_html=True
)

st.divider()

# --------------------------------------------------------
# 4. Context & Theory Section (Row 1)
# --------------------------------------------------------
col_intro, col_arch = st.columns(2, gap="large")

with col_intro:
    st.markdown("""
        <div class="glass-card">
            <div class="section-header">🧠 Project Overview</div>
            <p style="color: #CBD5E1; line-height: 1.7;">
                Developed for university research, this application addresses the critical challenge of missing-wedge 
                artifacts in electron tomography. By simulating physical hardware constraints, we can observe how our 
                neural network compensates for lost projection data.
            </p>
            <div class="section-header" style="margin-top: 1.5rem;">⚙️ Model Architecture</div>
            <p style="color: #CBD5E1; line-height: 1.7;">
                The core framework relies on a <strong>Variational Autoencoder (VAE)</strong> paired with a pre-trained 
                <strong>ResNet-18</strong> backbone acting as the feature extraction encoder. The network maps degraded 
                projection spaces to an optimized latent distribution, allowing the decoder to generate structurally 
                accurate tomograms even under severe missing wedge constraints.
            </p>
        </div>
    """, unsafe_allow_html=True)

with col_arch:
    try:
        arch_image = Image.open("assets/architecture.png")
        st.image(arch_image, caption="Figure 1: VAE Network pipeline with ResNet-18 Feature Extractor.", use_container_width=True)
    except FileNotFoundError:
        # Better looking fallback placeholder
        st.info("🖼️ **Architecture diagram placeholder:** Place 'architecture.png' in your 'assets/' folder to display the network pipeline here.")

st.divider()

# --------------------------------------------------------
# 5. Interactive Workspace Section (Row 2)
# --------------------------------------------------------
col_config, col_exec = st.columns(2, gap="large")

# --- LEFT COLUMN: Configuration ---
with col_config:
    st.markdown('<div class="section-header">🎛️ Simulation Configuration</div>', unsafe_allow_html=True)
    
    st.markdown("""
        <div class="info-box">
            <strong>💡 Acquisition Notice:</strong> Input files must represent a full, continuous sinogram 
            spanning from <strong>-90° to +90°</strong> with a 1° step resolution (181 total projections). 
            The configuration below simulates physical hardware constraints by dropping projections.
        </div>
    """, unsafe_allow_html=True)

    input_mode = st.radio(
        "Choose Sinogram Source Data:",
        ["Use a Preloaded Example File", "Upload Custom .mrc File"],
        horizontal=True
    )

    mrc_file_path = None

    if input_mode == "Use a Preloaded Example File":
        example_choice = st.selectbox(
            "Select an example dataset:",
            ["2 Squares", "Catalyst"]
        )
        filename = "2_squares.mrc" if "2 Squares" in example_choice else "catalyst.mrc"
        mrc_file_path = os.path.join("assets", filename)
        
        if os.path.exists(mrc_file_path):
            st.success(f"✅ **{example_choice}** initialized and ready.")
        else:
            st.warning(f"⚠️ Placeholder: Upload '{filename}' to your repository's 'assets/' folder.")
            mrc_file_path = None

    else:
        uploaded_file = st.file_uploader("Upload an experimental .mrc sinogram", type=["mrc"])
        if uploaded_file is not None:
            mrc_file_path = os.path.join("assets", "temp_upload.mrc")
            # Ensure assets dir exists for temp upload
            os.makedirs("assets", exist_ok=True) 
            with open(mrc_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success("✅ Custom .mrc file uploaded and parsed successfully.")

    st.markdown("<br>", unsafe_allow_html=True) # visual spacer
    
    config_choice = st.selectbox(
        "Select Missing Wedge Hardware Simulation Limit:",
        [
            "±50° Wedge Limit (5° Step)",
            "±50° Wedge Limit (10° Step)",
            "±50° Wedge Limit (20° Step)",
            "±40° Wedge Limit (5° Step)",
            "±40° Wedge Limit (10° Step)",
            "±40° Wedge Limit (20° Step)"
        ]
    )

    # Dictionary mapping config choice to parameters
    config_map = {
        "±50° Wedge Limit (5° Step)": {'range': (-50, 50), 'step': 5},
        "±50° Wedge Limit (10° Step)": {'range': (-50, 50), 'step': 10},
        "±50° Wedge Limit (20° Step)": {'range': (-50, 50), 'step': 20},
        "±40° Wedge Limit (5° Step)": {'range': (-40, 40), 'step': 5},
        "±40° Wedge Limit (10° Step)": {'range': (-40, 40), 'step': 10},
        "±40° Wedge Limit (20° Step)": {'range': (-40, 40), 'step': 20}
    }
    acquisition_config = config_map[config_choice]


# --- RIGHT COLUMN: Execution & Results ---
with col_exec:
    st.markdown('<div class="section-header">🔬 Inference & Results</div>', unsafe_allow_html=True)
    
    if mrc_file_path and os.path.exists(mrc_file_path):
        
        # Wrapped in a visually distinct container
        with st.container():
            st.markdown("<div style='padding-top: 10px;'></div>", unsafe_allow_html=True)
            run_btn = st.button("🚀 Run Tomographic Reconstruction", use_container_width=True, type="primary")
            
            if run_btn:
                output_image_path = os.path.join("assets", "latest_reconstruction_result.png")
                
                with st.spinner("🧠 Executing neural reconstruction algorithms... Please hold."):
                    run_streamlit_inference(
                        model = model,
                        input_mrc_path = mrc_file_path,
                        output_image_path = output_image_path,
                        is_complete=True,
                        acquisition_config = acquisition_config,
                        threshold = 0.05
                    )
                    
                if os.path.exists(output_image_path):
                    st.success("✨ Reconstruction complete!")
                    
                    result_img = Image.open(output_image_path)
                    st.image(result_img, caption="Reconstruction Output: Input Sinogram vs Neural Networks Solution", use_container_width=True)
                    
                    with open(output_image_path, "rb") as file:
                        st.download_button(
                            label="📥 Download High-Res Result Image",
                            data=file,
                            file_name="tomography_reconstruction_result.png",
                            mime="image/png",
                            use_container_width=True
                        )
                else:
                    st.error("❌ Inference executed, but output image could not be located.")
            else:
                # Empty state UI
                st.info("👈 System ready. Configure parameters on the left and initialize reconstruction.")
    else:
        st.warning("⚠️ Please select or upload a valid .mrc file dataset to unlock execution parameters.")