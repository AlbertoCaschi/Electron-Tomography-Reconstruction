import os
import gradio as gr
import spaces
import torch
from PIL import Image
import requests

# Import your custom cDDPM modules (ensure folders are uploaded to the Space repo)
from cDDPM.config import CONFIG as CDDPM_CONFIG
from cDDPM.models.diffusion import GaussianDiffusion
from cDDPM.models.unet import ConditionalUNet
from cDDPM.inference import run_streamlit_inference as cddpm_inference

# --- Load Model Globally (Cached on Space Startup) ---
print("Loading cDDPM model into GPU...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Example loading logic using your secret token
token = os.getenv("HF_TOKEN")
headers = {"Authorization": f"Bearer {token}"}
url = "https://huggingface.co/albertocaschi/cDDPM_Tomography/resolve/main/cDDPM.pt"
local_path = "/tmp/cDDPM.pt"

if not os.path.exists(local_path):
    response = requests.get(url, headers=headers, stream=True)
    with open(local_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

checkpoint = torch.load(local_path, map_location=device, weights_only=True)
model = ConditionalUNet(CDDPM_CONFIG).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("Model loaded successfully!")

# --- Define the GPU-accelerated Inference Function ---
@spaces.GPU
def predict(mrc_file, config_selection):
    # Map the config selection string back to your dictionary structure
    config_map = {
        "±50° Wedge (5° Step)": {'range': (-50, 50), 'step': 5},
        "±50° Wedge (10° Step)": {'range': (-50, 50), 'step': 10},
        "±50° Wedge (20° Step)": {'range': (-50, 50), 'step': 20},
        "±40° Wedge (5° Step)": {'range': (-40, 40), 'step': 5},
        "±40° Wedge (10° Step)": {'range': (-40, 40), 'step': 10},
        "±40° Wedge (20° Step)": {'range': (-40, 40), 'step': 20}
    }
    acquisition_config = config_map.get(config_selection, {'range': (-50, 50), 'step': 5})
    
    output_image_path = "/tmp/full_reconstruction_result.png"
    output_fbp_path = "/tmp/fbp_reconstruction_result.png"
    
    # Run your original inference function
    cddpm_inference(
        model=model,
        test_file=mrc_file.name if hasattr(mrc_file, 'name') else mrc_file,
        output_image_path=output_image_path,
        output_fbp_path=output_fbp_path,
        acquisition_config=acquisition_config
    )
    
    return output_image_path, output_fbp_path

# --- Set up the Headless Gradio Interface ---
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.File(label="MRC File"),
        gr.Textbox(label="Config Selection")
    ],
    outputs=[
        gr.Image(type="filepath", label="Full Reconstruction"),
        gr.Image(type="filepath", label="FBP Reconstruction")
    ]
)

if __name__ == "__main__":
    demo.launch()