import os
import requests
import streamlit as st
import torch
from PIL import Image
import shutil

# --- NEW: Import Gradio Client for remote API execution ---
from gradio_client import Client, handle_file

import sys
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

# --- VAE Imports remain for local execution ---
from VAEResNet.config import CONFIG as VAE_CONFIG
from VAEResNet.models.vae import TomographyVAE
from VAEResNet.inference import run_streamlit_inference as vae_inference

# --- cDDPM local imports have been removed to save memory ---

st.set_page_config(
    page_title="Electron Tomography Reconstruction",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
            
    /* 1. Base Container Setup */
    [data-testid="stAppViewContainer"] {
        background-color: #0B1120; 
        z-index: 0; /* Establishes a stacking context for backgrounds */
    }
    
    /* 2. Default Background Layer (Tab 1: VAE) */
    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: fixed;
        top: 0; left: 0; width: 100vw; height: 100vh;
        background-image: radial-gradient(ellipse 150% 150% at 0% 100%, 
            rgba(102, 213, 250, 0.40) 0%,
            transparent 75%
        );
        z-index: -1;
        opacity: 1;
        transition: opacity 1.2s ease-in-out;
        pointer-events: none;
    }

    /* 3. New Background Layer (Tab 2: Diffusion) */
    /* I shifted this one to the bottom-left (0% 100%) and gave it a soft purple/magenta tint to differentiate the model */
    [data-testid="stAppViewContainer"]::after {
        content: "";
        position: fixed;
        top: 0; left: 0; width: 100vw; height: 100vh;
        background-image: radial-gradient(ellipse 150% 150% at 100% 100%, 
            rgba(167, 139, 250, 0.40) 0%,
            transparent 75%
        );
        z-index: -1;
        opacity: 0; /* Hidden by default */
        transition: opacity 1.2s ease-in-out;
        pointer-events: none;
    }

    /* 4. The Trigger: If the second tab is selected, swap the opacities! */
    [data-testid="stAppViewContainer"]:has(div[data-testid="stTabs"] button:nth-of-type(2)[aria-selected="true"])::before {
        opacity: 0;
    }
    
    [data-testid="stAppViewContainer"]:has(div[data-testid="stTabs"] button:nth-of-type(2)[aria-selected="true"])::after {
        opacity: 1;
    }
    
            
    [data-testid="stHeader"] {
        background: transparent;
    }
    
    .subtitle { 
        font-size: 1.15rem; 
        color: #A0AEC0; 
        margin-bottom: 2rem; 
        line-height: 1.6; 
        animation: fadeIn 1.2s ease-in-out;
    }
    
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
    
    .info-box { 
        background: linear-gradient(90deg, rgba(30,58,138,0.3) 0%, rgba(15,23,42,0.1) 100%); 
        border-left: 4px solid #3B82F6; 
        padding: 1.2rem; 
        border-radius: 6px; 
        margin-bottom: 1.5rem; 
        color: #E2E8F0;
    }
            
    .title-wrapper {
        position: relative;
        overflow: hidden;
        padding: 10px 0;
        border-radius: 8px;
    }

    .title-wrapper::before {
        content: '';
        position: absolute;
        top: 0;
        left: -30%;
        width: 15%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0, 242, 254, 0.15), rgba(79, 172, 254, 0.25), transparent);
        transform: skewX(-25deg);
        animation: dataSweep 2.5s cubic-bezier(0.25, 1, 0.5, 1) forwards;
        animation-delay: 1s;
        pointer-events: none;
        z-index: 0;
    }

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
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulseBorder {
        0% { border-left-color: #3B82F6; }
        50% { border-left-color: #60A5FA; box-shadow: -4px 0px 10px rgba(59, 130, 246, 0.2); }
        100% { border-left-color: #3B82F6; }
    }
    
    div[data-testid="stRadio"] > label { font-weight: 600; color: #94A3B8; }
    div[data-testid="stSelectbox"] > label { font-weight: 600; color: #94A3B8; }
    
    /* 1. Base Tab Styling (Targeting the inner <p> tag for font size) */
    div[data-testid="stTabs"] button p {
        font-size: 1.35rem !important; /* Force the larger font size */
        font-weight: 600 !important;   /* Force the bold weight */
    }

    /* 2. Base Tab Color and Transition */
    div[data-testid="stTabs"] button {
        color: #A0AEC0 !important; /* Default unselected color */
        transition: color 0.3s ease;
    }

    /* 3. Hover State (Ensure both button and inner text change color) */
    div[data-testid="stTabs"] button:hover,
    div[data-testid="stTabs"] button:hover p {
        color: #ffffff !important; /* Brilliant white on hover */
    }

    /* 4. Selected Tab Text Color */
    div[data-testid="stTabs"] button[aria-selected="true"],
    div[data-testid="stTabs"] button[aria-selected="true"] p {
        color: #65b9f8 !important; /* Neon blue for the active text */
    }

    /* 5. Selected Tab Underline Color */
    div[data-baseweb="tab-highlight"] {
        background-color: #65b9f8 !important; /* Neon blue for the underline */
        height: 3px !important; /* Slightly thicker underline */
    }
            
    /* 6. Action Button Text Color */
    div[data-testid="stButton"] button p,
    div[data-testid="stDownloadButton"] button p {
        color: #ffffff !important; 
        font-weight: 600 !important;
    }
    </style>
""", unsafe_allow_html=True)


# Model loading into cache - ONLY FOR VAE
@st.cache_resource(show_spinner=False)
def load_private_model(model_type):
    """Fetches the private PyTorch model bypassing the HF library."""
    
    status = st.empty()
    
    try:
        if "HF_TOKEN" not in st.secrets:
            status.error("HF_TOKEN is missing from Streamlit secrets!")
            return None
            
        token = st.secrets["HF_TOKEN"]
        headers = {"Authorization": f"Bearer {token}"}
        
        if model_type == "VAE":
            url = "https://huggingface.co/albertocaschi/VAEResNet_Tomography/resolve/main/VAEResNet.pth"
            local_path = "/tmp/VAEResNet.pth"
        else:
            status.error(f"Unknown model type: {model_type}")
            return None
        
        if not os.path.exists(local_path):
            status.info(f"Loading {model_type} model...")
            
            response = requests.get(url, headers=headers, stream=True)
            
            if response.status_code != 200:
                status.error(f"HTTP Error {response.status_code}: Check if your token is valid and has 'Read' access.")
                return None
                
            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        status.info(f"Loading {model_type} into PyTorch...")
        
        device = torch.device('cpu')
        
        checkpoint = torch.load(local_path, map_location=device, weights_only=True) 
        
        if model_type == "VAE":
            model = TomographyVAE(
                latent_dim=VAE_CONFIG["latent_dim"],
                target_size=VAE_CONFIG["target_size"],
                resnet_type=VAE_CONFIG["resnet_type"],
                freeze_early_layers=VAE_CONFIG["freeze_early_layers"]
            ).to(device)
            
            model.load_state_dict(checkpoint['model_state_dict'])

        model.eval()
        
        status.success(f"{model_type} loaded successfully!")
        status.empty() 
        
        return model
        
    except Exception as e:
        status.error(f"Failed to load model. Error: {e}")
        print(f"DEBUG ERROR: {e}")
        return None

# Only the VAE model is loaded locally now
vae_model = load_private_model("VAE")

## APP

# Title
st.markdown("""
    <div class="title-wrapper">
        <div class="main-title">AI-Powered Electron Tomography Reconstruction</div>
    </div>
""", unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Alberto Caschi - University of Udine</div>', 
    unsafe_allow_html=True
)

# --- NEW NAVIGATION BAR ---
tab1, tab2 = st.tabs(["VAE model", "Diffusion Model"])

with tab1:
    # Overview
    col_intro, col_arch = st.columns(2, gap="large")

    with col_intro:
        st.markdown("""
            <div class="glass-card">
                <div class="section-header">Method Overview</div>
                <p style="color: #CBD5E1; line-height: 1.7; text-align: justify; text-justify: inter-word;">
                    This method addresses the electron-tomography missing wedge and projections by training a VAE-based sinogram inpainting model to infer absent sinogram data. Using 2500 synthetic samples with noise and augmentations, the network learns physically consistent projections that improve FBP reconstruction, enabling 2D slice recovery from severely incomplete inputs.
                </p>
                <div class="section-header" style="margin-top: 1.5rem;">Model Architecture</div>
                <p style="color: #CBD5E1; line-height: 1.7; text-align: justify; text-justify: inter-word;">
                    This model is a <strong>Variational Autoencoder</strong> built on a pre-trained <strong>ResNet-18</strong> that compresses each sinogram into a 64-number representation (latent vector). A decoder then expands it back using smooth upsampling steps to avoid visual artifacts. Training uses smart regularization so the network learns real geometric patterns instead of memorizing examples.
                </p>
                <p style="color: #B5C3D2; line-height: 1.7; font-size: 0.7rem;">
                    Check the slides for additional information.
                </p>
            </div>
        """, unsafe_allow_html=True)

    with col_arch:
        st.markdown("<div style='margin-top: 3.5rem;'></div>", unsafe_allow_html=True)
        try:
            arch_image = Image.open("assets/vae_architecture.png")
            st.image(arch_image, caption="VAE Network pipeline with ResNet-18 Feature Extractor.", use_container_width=True)
        except FileNotFoundError:
            st.info("**Architecture diagram placeholder:** Place 'vae_architecture.png' in your 'assets/' folder to display the network pipeline here.")

    st.divider()

    ## MODEL TESTING PART
    col_config, col_exec = st.columns(2, gap="large")

    # LEFT COLUMN: Configuration
    with col_config:
        st.markdown('<div class="section-header">Simulation Configuration</div>', unsafe_allow_html=True)

        input_mode = st.radio(
            "Choose sinogram source data:",
            ["Use a preloaded example file", "Upload custom .mrc file"],
            horizontal=True,
            key="radio_tab1"
        )

        if input_mode == "Upload custom .mrc file":
            st.markdown("""
                <div class="info-box">
                    <strong>Please note:</strong>
                    <ul>
                    <li>Upload a full sinogram (-90° to +90°, 1° steps). The app will simulate the missing wedge and projections you choose below.</li>
                    <li>Input sinograms must have the following size: <strong>[362, 181]</strong>.</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)

        mrc_file_path = None

        if input_mode == "Use a preloaded example file":
            example_choice = st.selectbox(
                "Select an example sinogram:",
                [
                    "Rectangle",
                    "Oval 1",
                    "Oval 2",
                    "Rectangle + Oval",
                    "Circle",
                    "2 Squares",
                    "Catalyst"
                ],
                key="sb_example_tab1"
            )

            if "Rectangle" in example_choice and "Oval" in example_choice:
                filename = "rect_oval.mrc"
            elif "Oval 1" in example_choice:
                filename = "oval1.mrc"
            elif "Oval 2" in example_choice:
                filename = "oval2.mrc"
            elif "Rectangle" in example_choice:
                filename = "rectangle.mrc"
            elif "Circle" in example_choice:
                filename = "circle.mrc"
            elif "2 Squares" in example_choice:
                filename = "2_squares.mrc"
            elif "Catalyst" in example_choice:
                filename = "catalyst.mrc"

            mrc_file_path = os.path.join("assets", filename)
            
            if os.path.exists(mrc_file_path):
                st.success(f"**{example_choice}** initialized and ready.")
            else:
                st.warning(f"Placeholder: upload '{filename}' to your repository's 'assets/' folder.")
                mrc_file_path = None

        else:
            uploaded_file = st.file_uploader("Upload an experimental .mrc sinogram", type=["mrc"], key="uploader_tab1")
            if uploaded_file is not None:
                filename = "uploaded.mrc"
                mrc_file_path = os.path.join("assets", filename)
                os.makedirs("assets", exist_ok=True) 
                with open(mrc_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success("Custom .mrc file uploaded and parsed successfully.")

        st.markdown("<br>", unsafe_allow_html=True)
        
        config_choice = st.selectbox(
            "Select missing wedge and projection simulation:",
            [
                "±50° Wedge (5° Step)",
                "±50° Wedge (10° Step)",
                "±50° Wedge (20° Step)",
                "±40° Wedge (5° Step)",
                "±40° Wedge (10° Step)",
                "±40° Wedge (20° Step)"
            ],
            key="sb_wedge_tab1"
        )

        config_map = {
            "±50° Wedge (5° Step)": {'range': (-50, 50), 'step': 5},
            "±50° Wedge (10° Step)": {'range': (-50, 50), 'step': 10},
            "±50° Wedge (20° Step)": {'range': (-50, 50), 'step': 20},
            "±40° Wedge (5° Step)": {'range': (-40, 40), 'step': 5},
            "±40° Wedge (10° Step)": {'range': (-40, 40), 'step': 10},
            "±40° Wedge (20° Step)": {'range': (-40, 40), 'step': 20}
        }
        acquisition_config = config_map[config_choice]


    # --- RIGHT COLUMN: Inference and results ---
    with col_exec:
        st.markdown('<div class="section-header">Inference & Results</div>', unsafe_allow_html=True)
        
        if mrc_file_path and os.path.exists(mrc_file_path):
            
            with st.container():
                st.markdown("<div style='padding-top: 10px;'></div>", unsafe_allow_html=True)
                run_btn = st.button("🚀 Run Tomographic Reconstruction", use_container_width=True, type="primary", key="btn_run_tab1")
                
                if run_btn:
                    output_image_path = os.path.join("assets", "full_reconstruction_result.png")
                    output_fbp_path = os.path.join("assets", "fbp_reconstruction_result.png")
                    
                    loading_placeholder = st.empty()

                    loading_placeholder.markdown("""
                        <style>
                        .custom-loader-container {
                            display: flex;
                            flex-direction: column;
                            align-items: center;
                            justify-content: center;
                            padding: 3rem;
                            background: rgba(17, 25, 40, 0.4);
                            border-radius: 12px;
                            border: 1px solid rgba(0, 242, 254, 0.2);
                            margin: 2rem 0;
                        }
                        
                        .scanner-track {
                            width: 80%;
                            height: 4px;
                            background: rgba(255, 255, 255, 0.1);
                            border-radius: 4px;
                            position: relative;
                            overflow: hidden;
                            margin-top: 1.5rem;
                        }
                        
                        /* The moving glowing beam */
                        .scanner-beam {
                            position: absolute;
                            top: 0;
                            left: -50%;
                            width: 50%;
                            height: 100%;
                            background: linear-gradient(90deg, transparent, #00f2fe, #4facfe, transparent);
                            animation: scan 2s infinite linear;
                        }
                        
                        .loader-text {
                            color: #00f2fe;
                            font-weight: 600;
                            font-size: 1.1rem;
                            letter-spacing: 2px;
                            margin-top: 1rem;
                            animation: pulseText 2s infinite ease-in-out;
                        }
                        
                        /* Animations */
                        @keyframes scan {
                            0% { left: -50%; }
                            100% { left: 100%; }
                        }
                        
                        @keyframes pulseText {
                            0%, 100% { opacity: 0.5; text-shadow: 0 0 0 transparent; }
                            50% { opacity: 1; text-shadow: 0 0 10px rgba(0, 242, 254, 0.6); }
                        }
                        </style>
                        
                        <div class="custom-loader-container">
                            <div style="font-size: 2rem; animation: pulseText 2s infinite;">RECONSTRUCTING SINOGRAM</div>
                            <div class="loader-text">Please wait...</div>
                            <div class="scanner-track">
                                <div class="scanner-beam"></div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    vae_inference(
                        model = vae_model,
                        input_mrc_path = mrc_file_path,
                        output_image_path = output_image_path,
                        output_fbp_path = output_fbp_path,
                        is_complete=True,
                        acquisition_config = acquisition_config,
                        threshold = 0.05
                    )
                    
                    loading_placeholder.empty()
                    
                    if os.path.exists(output_image_path):
                        st.success("Reconstruction complete!")
                        
                        result_img = Image.open(output_image_path)
                        st.image(result_img, caption="Reconstruction output: input sinogram and neural networks solution", use_container_width=False)
                        
                        with open(output_fbp_path, "rb") as file:
                            st.download_button(
                                label="📥  Download FBP reconstruction (PNG)",
                                data=file,
                                file_name=f"{filename[:-4]}_reconstruction_result.png",
                                mime="image/png",
                                use_container_width=True,
                                key="btn_dl_tab1"
                            )
                    else:
                        st.error("Inference executed, but output image could not be located.")
                else:
                    st.info("System ready. Configure parameters on the left and initialize reconstruction.")
        else:
            st.warning("Please select or upload a valid .mrc file.")

with tab2:

    # Overview
    col_intro, col_arch = st.columns(2, gap="large")

    with col_intro:
        st.markdown("""
            <div class="glass-card">
                <div class="section-header">Method Overview</div>
                <p style="color: #CBD5E1; line-height: 1.7; text-align: justify; text-justify: inter-word;">
                    This method addresses the electron-tomography missing wedge problem by training a generative diffusion model to reconstruct 2D slices, using incomplete Filtered Back Projection (FBP) images as deterministic spatial constraints. Using 2,500 synthetic samples with noise and augmentations, the network learns to map physically consistent features. The mathematical framework relies on a Gaussian diffusion process that systematically adds and removes noise over a scheduled progression (such as a linear or cosine schedule) across a fixed number of timesteps (typically 1,000). By learning the reverse of this degradation, the method enables high-fidelity 2D slice recovery from severely incomplete projection inputs.
                <div class="section-header" style="margin-top: 1.5rem;">Model Architecture</div>
                <p style="color: #CBD5E1; line-height: 1.7; text-align: justify; text-justify: inter-word;">
                    This model is a <strong>Conditional Denoising Diffusion Probabilistic Model</strong> (<strong>cDDPM</strong>) built upon a 4-level U-Net backbone. Rather than transforming a noisy image directly into a clear version in a single pass, the network is trained to iteratively predict the specific noise tensor at each timestep.  To know exactly which object needs to be reconstructed, the model employs an "early fusion" conditioning mechanism: the current noisy image and the static, incomplete 2D FBP reconstruction are concatenated directly at the channel level before entering the network.  The U-Net architecture is optimized for both local and global awareness: it utilizes standard convolutional blocks for extracting fine structural details at higher resolutions, and selectively injects self-attention exclusively at its deepest bottleneck (Level 4) to ensure global structural coherence without overwhelming memory. Because of this robust, structurally-guided architecture, the model is able to generalize and successfully reconstruct any image with a 60-40 projection wedge, handling any number of projections inside this range.
                </p>
                <p style="color: #B5C3D2; line-height: 1.7; font-size: 0.7rem;">
                    Check the slides for additional information.
                </p>
            </div>
        """, unsafe_allow_html=True)

    with col_arch:
        st.markdown("<div style='margin-top: 3.5rem;'></div>", unsafe_allow_html=True)
        try:
            arch_image = Image.open("assets/cddpm_architecture.png")
            st.image(arch_image, caption="U-Net cDDPM model architecture", use_container_width=True)
        except FileNotFoundError:
            st.info("**Architecture diagram placeholder:** Place 'cddpm_architecture.png' in your 'assets/' folder to display the network pipeline here.")

    st.divider()

    ## MODEL TESTING PART
    col_config, col_exec = st.columns(2, gap="large")

    # LEFT COLUMN: Configuration
    with col_config:
        st.markdown('<div class="section-header">Simulation Configuration</div>', unsafe_allow_html=True)

        input_mode = st.radio(
            "Choose sinogram source data:",
            ["Use a preloaded example file", "Upload custom .mrc file"],
            horizontal=True,
            key="radio_tab2"
        )

        if input_mode == "Upload custom .mrc file":
            st.markdown("""
                <div class="info-box">
                    <strong>Please note:</strong>
                    <ul>
                    <li>Upload a full sinogram (-90° to +90°, 1° steps). The app will simulate the missing wedge and projections you choose below.</li>
                    <li>Input sinograms must have the following size: <strong>[362, 181]</strong>.</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)

        mrc_file_path = None

        if input_mode == "Use a preloaded example file":
            example_choice = st.selectbox(
                "Select an example sinogram:",
                [
                    "Rectangle",
                    "Oval 1",
                    "Oval 2",
                    "Rectangle + Oval",
                    "Circle",
                    "2 Squares",
                    "Catalyst"
                ],
                key="sb_example_tab2"
            )

            if "Rectangle" in example_choice and "Oval" in example_choice:
                filename = "rect_oval.mrc"
            elif "Oval 1" in example_choice:
                filename = "oval1.mrc"
            elif "Oval 2" in example_choice:
                filename = "oval2.mrc"
            elif "Rectangle" in example_choice:
                filename = "rectangle.mrc"
            elif "Circle" in example_choice:
                filename = "circle.mrc"
            elif "2 Squares" in example_choice:
                filename = "2_squares.mrc"
            elif "Catalyst" in example_choice:
                filename = "catalyst.mrc"

            mrc_file_path = os.path.join("assets", filename)
            
            if os.path.exists(mrc_file_path):
                st.success(f"**{example_choice}** initialized and ready.")
            else:
                st.warning(f"Placeholder: upload '{filename}' to your repository's 'assets/' folder.")
                mrc_file_path = None

        else:
            uploaded_file = st.file_uploader("Upload an experimental .mrc sinogram", type=["mrc"], key="uploader_tab2")
            if uploaded_file is not None:
                filename = "uploaded.mrc"
                mrc_file_path = os.path.join("assets", filename)
                os.makedirs("assets", exist_ok=True) 
                with open(mrc_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success("Custom .mrc file uploaded and parsed successfully.")

        st.markdown("<br>", unsafe_allow_html=True)
        
        config_choice = st.selectbox(
            "Select missing wedge and projection simulation:",
            [
                "±50° Wedge (5° Step)",
                "±50° Wedge (10° Step)",
                "±50° Wedge (20° Step)",
                "±40° Wedge (5° Step)",
                "±40° Wedge (10° Step)",
                "±40° Wedge (20° Step)"
            ],
            key="sb_wedge_tab2"
        )

        config_map = {
            "±50° Wedge (5° Step)": {'range': (-50, 50), 'step': 5},
            "±50° Wedge (10° Step)": {'range': (-50, 50), 'step': 10},
            "±50° Wedge (20° Step)": {'range': (-50, 50), 'step': 20},
            "±40° Wedge (5° Step)": {'range': (-40, 40), 'step': 5},
            "±40° Wedge (10° Step)": {'range': (-40, 40), 'step': 10},
            "±40° Wedge (20° Step)": {'range': (-40, 40), 'step': 20}
        }
        acquisition_config = config_map[config_choice]


    # --- RIGHT COLUMN: Inference and results ---
    with col_exec:
        st.markdown('<div class="section-header">Inference & Results</div>', unsafe_allow_html=True)
        
        if mrc_file_path and os.path.exists(mrc_file_path):
            
            with st.container():
                st.markdown("<div style='padding-top: 10px;'></div>", unsafe_allow_html=True)
                run_btn = st.button("🚀 Run Tomographic Reconstruction", use_container_width=True, type="primary", key="btn_run_tab2")
                
                if run_btn:
                    # Defining local destinations for the returned backend images
                    output_image_path = os.path.join("assets", "full_reconstruction_result.png")
                    output_fbp_path = os.path.join("assets", "fbp_reconstruction_result.png")
                    
                    loading_placeholder = st.empty()

                    loading_placeholder.markdown("""
                        <style>
                        .custom-loader-container {
                            display: flex;
                            flex-direction: column;
                            align-items: center;
                            justify-content: center;
                            padding: 3rem;
                            background: rgba(17, 25, 40, 0.4);
                            border-radius: 12px;
                            border: 1px solid rgba(0, 242, 254, 0.2);
                            margin: 2rem 0;
                        }
                        
                        .scanner-track {
                            width: 80%;
                            height: 4px;
                            background: rgba(255, 255, 255, 0.1);
                            border-radius: 4px;
                            position: relative;
                            overflow: hidden;
                            margin-top: 1.5rem;
                        }
                        
                        /* The moving glowing beam */
                        .scanner-beam {
                            position: absolute;
                            top: 0;
                            left: -50%;
                            width: 50%;
                            height: 100%;
                            background: linear-gradient(90deg, transparent, #00f2fe, #4facfe, transparent);
                            animation: scan 2s infinite linear;
                        }
                        
                        .loader-text {
                            color: #00f2fe;
                            font-weight: 600;
                            font-size: 1.1rem;
                            letter-spacing: 2px;
                            margin-top: 1rem;
                            animation: pulseText 2s infinite ease-in-out;
                        }
                        
                        /* Animations */
                        @keyframes scan {
                            0% { left: -50%; }
                            100% { left: 100%; }
                        }
                        
                        @keyframes pulseText {
                            0%, 100% { opacity: 0.5; text-shadow: 0 0 0 transparent; }
                            50% { opacity: 1; text-shadow: 0 0 10px rgba(0, 242, 254, 0.6); }
                        }
                        </style>
                        
                        <div class="custom-loader-container">
                            <div style="font-size: 2rem; animation: pulseText 2s infinite;">RECONSTRUCTING 2D SLICE</div>
                            <div class="loader-text">Please wait (may take a few minutes)...</div>
                            <div class="scanner-track">
                                <div class="scanner-beam"></div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    try:
                        # --- NEW: Call the remote Hugging Face API instead of local execution ---
                        
                        # IMPORTANT: Replace the string below with your actual Hugging Face Space repository ID!
                        hf_space_client = Client("albertocaschi/tomo-cddpm-backend")
                        
                        # Send the file and configuration to the API.
                        # Note: This expects your Gradio app to return two file paths: (full_reconstruction, fbp_reconstruction)
                        result_image_temp, fbp_image_temp = hf_space_client.predict(
                            mrc_file=handle_file(mrc_file_path),
                            config_selection=config_choice,
                            api_name="/predict"
                        )
                        
                        # Move the returned temporary files to the asset folder
                        shutil.copy(result_image_temp, output_image_path)
                        shutil.copy(fbp_image_temp, output_fbp_path)
                        
                        loading_placeholder.empty()
                        
                        if os.path.exists(output_image_path):
                            st.success("Reconstruction complete!")
                            
                            result_img = Image.open(output_image_path)
                            st.image(result_img, caption="Reconstruction output: input sinogram and neural networks solution", use_container_width=False)
                            
                            with open(output_fbp_path, "rb") as file:
                                st.download_button(
                                    label="📥  Download FBP reconstruction (PNG)",
                                    data=file,
                                    file_name=f"{filename[:-4]}_reconstruction_result.png",
                                    mime="image/png",
                                    use_container_width=True,
                                    key="btn_dl_tab2"
                                )
                        else:
                            st.error("Inference executed on external GPU, but output image could not be loaded locally.")
                            
                    except Exception as e:
                        loading_placeholder.empty()
                        st.error(f"Failed to communicate with Hugging Face ZeroGPU space: {e}")
                else:
                    st.info("System ready. Configure parameters on the left and initialize reconstruction.")
        else:
            st.warning("Please select or upload a valid .mrc file.")