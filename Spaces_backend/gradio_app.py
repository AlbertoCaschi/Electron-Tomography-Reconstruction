import os
import gradio as gr
import spaces
import torch
from PIL import Image
import requests

# Import your custom cDDPM modules (ensure folders are uploaded to the Space repo)
from cDDPM.config import CONFIG as CDDPM_CONFIG_v1
from cDDPM.models.diffusion import GaussianDiffusion as GaussianDiffusion_v1
from cDDPM.models.unet import ConditionalUNet as ConditionalUNet_v1
from cDDPM.inference import run_streamlit_inference as cddpm_inference_v1

from cDDPM_v2.config import CONFIG as CDDPM_CONFIG_v2
from cDDPM_v2.models.diffusion import GaussianDiffusion as GaussianDiffusion_v2
from cDDPM_v2.models.unet import ConditionalUNet as ConditionalUNet_v2
from cDDPM_v2.inference import run_streamlit_inference as cddpm_inference_v2

# --- Load Models Globally (Cached on Space Startup) ---
print("Loading cDDPM models into GPU...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

token = os.getenv("HF_TOKEN")
headers = {"Authorization": f"Bearer {token}"} if token else {}

# Load Model v1 (Original)
url_v1 = "https://huggingface.co/albertocaschi/cDDPM_Tomography/resolve/main/cDDPM.pt"
local_path_v1 = "/tmp/cDDPM.pt"

if not os.path.exists(local_path_v1):
    response = requests.get(url_v1, headers=headers, stream=True)
    with open(local_path_v1, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

checkpoint_v1 = torch.load(local_path_v1, map_location=device, weights_only=True)
model = ConditionalUNet_v1(CDDPM_CONFIG_v1).to(device)
model.load_state_dict(checkpoint_v1['model_state_dict'])
model.eval()

# Load Model v2
url_v2 = "https://huggingface.co/albertocaschi/cDDPM_v2_Tomography/resolve/main/cDDPM_v2.pt"
local_path_v2 = "/tmp/cDDPM_v2.pt"

if not os.path.exists(local_path_v2):
    response = requests.get(url_v2, headers=headers, stream=True)
    with open(local_path_v2, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

checkpoint_v2 = torch.load(local_path_v2, map_location=device, weights_only=True)
model_v2 = ConditionalUNet_v2(CDDPM_CONFIG_v2).to(device)
model_v2.load_state_dict(checkpoint_v2['model_state_dict'])
model_v2.eval()

print("Models loaded successfully!")

# --- Define the GPU-accelerated Inference Function for v1 ---
@spaces.GPU
def predict_v1(mrc_file, config_selection):
    # Map the config selection string back to your dictionary structure
    config_map = {
        "±50° Wedge (5° Step)": {'range': (-50, 50), 'step': 5},
        "±50° Wedge (10° Step)": {'range': (-50, 50), 'step': 10},
        "±50° Wedge (20° Step)": {'range': (-50, 50), 'step': 20},
        "±40° Wedge (5° Step)": {'range': (-40, 40), 'step': 5},
        "±40° Wedge (10° Step)": {'range': (-40, 40), 'step': 10},
        "±40° Wedge (20° Step)": {'range': (-40, 40), 'step': 20},
        "±30° Wedge (5° Step)": {'range': (-30, 30), 'step': 5},
        "±30° Wedge (10° Step)": {'range': (-30, 30), 'step': 10},
        "±30° Wedge (15° Step)": {'range': (-30, 30), 'step': 15}
    }
    acquisition_config = config_map.get(config_selection, {'range': (-50, 50), 'step': 5})
    
    output_image_path = "/tmp/full_reconstruction_result.png"
    output_fbp_path = "/tmp/fbp_reconstruction_result.png"
    
    # Run your original inference function
    cddpm_inference_v1(
        model=model,
        test_file=mrc_file.name if hasattr(mrc_file, 'name') else mrc_file,
        output_image_path=output_image_path,
        output_fbp_path=output_fbp_path,
        acquisition_config=acquisition_config
    )
    
    return output_image_path, output_fbp_path

# --- Define the GPU-accelerated Inference Function for v2 ---
@spaces.GPU
def predict_v2(mrc_file, config_selection):
    config_map = {
        "±50° Wedge (5° Step)": {'range': (-50, 50), 'step': 5},
        "±50° Wedge (10° Step)": {'range': (-50, 50), 'step': 10},
        "±50° Wedge (20° Step)": {'range': (-50, 50), 'step': 20},
        "±40° Wedge (5° Step)": {'range': (-40, 40), 'step': 5},
        "±40° Wedge (10° Step)": {'range': (-40, 40), 'step': 10},
        "±40° Wedge (20° Step)": {'range': (-40, 40), 'step': 20},
        "±30° Wedge (5° Step)": {'range': (-30, 30), 'step': 5},
        "±30° Wedge (10° Step)": {'range': (-30, 30), 'step': 10},
        "±30° Wedge (15° Step)": {'range': (-30, 30), 'step': 15},
        "±20° Wedge (5° Step)": {'range': (-20, 20), 'step': 5},
        "±20° Wedge (10° Step)": {'range': (-20, 20), 'step': 10},
        "±20° Wedge (15° Step)": {'range': (-20, 20), 'step': 15}
    }
    acquisition_config = config_map.get(config_selection, {'range': (-50, 50), 'step': 5})
    
    # Use distinct temporary paths to avoid write conflicts if run concurrently
    output_image_path = "/tmp/full_reconstruction_result_v2.png"
    output_sirt_path = "/tmp/sirt_reconstruction_result_v2.png"
    
    # Run the v2 inference function
    cddpm_inference_v2(
        model=model_v2,
        test_file=mrc_file.name if hasattr(mrc_file, 'name') else mrc_file,
        output_image_path=output_image_path,
        output_sirt_path=output_sirt_path,
        acquisition_config=acquisition_config
    )
    
    return output_image_path, output_sirt_path

# --- Set up the Headless Gradio Interface ---
# Create interfaces for both models
interface_v1 = gr.Interface(
    fn=predict_v1,
    inputs=[gr.File(label="MRC File"), gr.Textbox(label="Config Selection")],
    outputs=[gr.Image(type="filepath", label="Full Reconstruction"), gr.Image(type="filepath", label="FBP Reconstruction")]
)

interface_v2 = gr.Interface(
    fn=predict_v2,
    inputs=[gr.File(label="MRC File"), gr.Textbox(label="Config Selection")],
    outputs=[gr.Image(type="filepath", label="Full Reconstruction (v2)"), gr.Image(type="filepath", label="SIRT Reconstruction (v2)")]
)

# Combine them into a single tabbed interface
demo = gr.TabbedInterface(
    [interface_v1, interface_v2],
    ["Model v1", "Model v2"]
)

if __name__ == "__main__":
    demo.launch()