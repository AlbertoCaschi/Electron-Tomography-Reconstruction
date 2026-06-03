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
    # page_icon="🔬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for a clean, scientific aesthetic
st.markdown("""
    <style>
    .main-title { font-size: 2.4rem; font-weight: 700; color: #0F172A; margin-bottom: 0.5rem; }
    .subtitle { font-size: 1.1rem; color: #475569; margin-bottom: 2rem; line-height: 1.6; }
    .section-header { font-size: 1.6rem; font-weight: 600; color: #1E3A8A; margin-top: 2rem; margin-bottom: 1rem; border-bottom: 2px solid #E2E8F0; padding-bottom: 0.5rem; }
    .info-box { background-color: #F8FAFC; border-left: 4px solid #3B82F6; padding: 1rem; border-radius: 0.375rem; margin-bottom: 1.5rem; }
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
        
        # 1. Download the file from Hugging Face
        model_path = hf_hub_download(
            repo_id="albertocaschi/VAEResNet_Tomography", 
            filename="VAEResNet.pth", 
            token=token
        )
        
        # 2. Set device to CPU for Streamlit Cloud
        device = torch.device('cpu')
        
        # 3. Load the checkpoint dictionary
        checkpoint = torch.load(model_path, map_location=device)
        
        # 4. Initialize the model skeleton
        model = TomographyVAE(
            latent_dim=CONFIG["latent_dim"],
            target_size=CONFIG["target_size"],
            resnet_type=CONFIG["resnet_type"],
            freeze_early_layers=CONFIG["freeze_early_layers"]
        ).to(device)
        
        # 5. Load the weights into the skeleton
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model
        
    except Exception as e:
        st.error(f"Failed to load model from Hugging Face. Error: {e}")
        return None

# Load model globally into cache
model = load_private_model()

# --------------------------------------------------------
# 3. Header & Project Introduction
# --------------------------------------------------------
st.markdown('<div class="main-title">AI-Powered Electron Tomography Reconstruction</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">This application leverages a deep generative framework to reconstruct high-fidelity '
    '3D volumes from limited-angle or missing-wedge electron tomography data. Developed for university research '
    'to bypass traditional analytical reconstruction artifacts.</div>', 
    unsafe_allow_html=True
)

st.markdown('<div class="section-header">Model Architecture</div>', unsafe_allow_html=True)
st.markdown(
    "The core framework relies on a **Variational Autoencoder (VAE)** paired with a pre-trained **ResNet-18** backbone "
    "acting as the feature extraction encoder. The network maps degraded projection spaces to an optimized latent distribution, "
    "allowing the decoder to generate structurally accurate tomograms even under severe missing wedge constraints."
)

# Display Architecture Diagram
try:
    arch_image = Image.open("assets/architecture.png")
    st.image(arch_image, caption="Figure 1: VAE Network pipeline with ResNet-18 Feature Extractor.", use_container_width=True)
except FileNotFoundError:
    st.warning("Architecture diagram placeholder: Place 'architecture.png' in your 'assets/' folder to display it here.")

# --------------------------------------------------------
# 4. Interactive Test Section
# --------------------------------------------------------
st.markdown('<div class="section-header">Interactive Model Testing</div>', unsafe_allow_html=True)

# Important User Reminder Box
st.markdown("""
    <div class="info-box">
        <strong>💡 Acquisition Notice:</strong> Input files must represent a full, continuous sinogram 
        spanning from <strong>-90° to +90°</strong> with a 1° step resolution (181 total projections). 
        The simulation configuration selected below will automatically drop projections to simulate 
        specific physical hardware constraints.
    </div>
""", unsafe_allow_html=True)

# Data Input Selection
input_mode = st.radio(
    "Choose Sinogram Source Data:",
    ["Use a Preloaded Example File", "Upload Custom .mrc File"]
)

mrc_file_path = None
uploaded_file_bytes = None

if input_mode == "Use a Preloaded Example File":
    example_choice = st.selectbox(
        "Select an example file:",
        [
            "2 Squares",
            "Catalyst",
        ]
    )
    # Define file paths based on choice
    filename = "2_squares.mrc" if "2 Squares" in example_choice else "catalyst.mrc"
    mrc_file_path = os.path.join("assets", filename)
    
    if os.path.exists(mrc_file_path):
        st.success(f"Selected {example_choice} initialized successfully.")
    else:
        st.info(f"Placeholder: Upload '{filename}' to your repository's 'assets/' folder to activate this choice.")
        mrc_file_path = None

else:
    uploaded_file = st.file_uploader("Upload an experimental .mrc sinogram", type=["mrc"])
    if uploaded_file is not None:
        # Save uploaded file temporarily to pass to the inference script
        mrc_file_path = os.path.join("assets", "temp_upload.mrc")
        with open(mrc_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("Custom .mrc file uploaded and parsed successfully.")

# Configuration Dropdown for Missing Wedge Simulation
config_choice = st.selectbox(
    "Select Missing Wedge Hardware Simulation Configuration:",
    [
        "±50° Wedge Limit (5° Step)",
        "±50° Wedge Limit (10° Step)",
        "±50° Wedge Limit (20° Step)",
        "±40° Wedge Limit (5° Step)",
        "±40° Wedge Limit (10° Step)",
        "±40° Wedge Limit (20° Step)"
    ]
)

# Parse configurations down to lists/variables needed by your python logic
if "±50° Wedge Limit (5° Step)" in config_choice:
    acquisition_config = {'range': (-50, 50), 'step': 5}
elif "±50° Wedge Limit (10° Step)" in config_choice:
    acquisition_config = {'range': (-50, 50), 'step': 10}
elif "±50° Wedge Limit (20° Step)" in config_choice:
    acquisition_config = {'range': (-50, 50), 'step': 20}
elif "±40° Wedge Limit (5° Step)" in config_choice:
    acquisition_config = {'range': (-40, 40), 'step': 5}
elif "±40° Wedge Limit (10° Step)" in config_choice:
    acquisition_config = {'range': (-40, 40), 'step': 10}
elif "±40° Wedge Limit (20° Step)" in config_choice:
    acquisition_config = {'range': (-40, 40), 'step': 20}

# --------------------------------------------------------
# 5. Inference Execution & Output Visualizations
# --------------------------------------------------------
if mrc_file_path and os.path.exists(mrc_file_path):
    st.markdown("### Execution")
    
    if st.button("🚀 Run Tomographic Reconstruction"):
        # Temporary output image path created by run_inference
        output_image_path = os.path.join("assets", "latest_reconstruction_result.png")
        
        with st.spinner("Executing neural reconstruction algorithms... Please hold."):
            # Call your external module function
            # Pass model, file path, missing angles, and degree intervals
            run_streamlit_inference(
                model = model,
                input_mrc_path = mrc_file_path,
                output_image_path = "assets/latest_reconstruction_result.png",
                is_complete=True,
                acquisition_config = acquisition_config,
                threshold = 0.05
            )
            
        if os.path.exists(output_image_path):
            st.success("Reconstruction complete!")
            
            # Display results image
            result_img = Image.open(output_image_path)
            st.image(result_img, caption="Reconstruction Output: Input Sinogram vs Neural Networks Solution", use_container_width=True)
            
            # Local Download Button for stakeholders
            with open(output_image_path, "rb") as file:
                st.download_button(
                    label="📥 Download Result Image",
                    data=file,
                    file_name="tomography_reconstruction_result.png",
                    mime="image/png"
                )
        else:
            st.error("Inference executed, but output image generation path could not be located.")
else:
    st.info("Please select or upload a valid .mrc file data set to unlock execution parameters.")