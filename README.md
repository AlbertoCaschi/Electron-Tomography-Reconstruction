# AI-Powered Electron Tomography Reconstruction

## About the Project
This project addresses the electron-tomography missing wedge and projection problems by training a VAE-based sinogram inpainting model to infer absent sinogram data. Using 2,500 synthetic samples with noise and augmentations, the network learns physically consistent projections that improve Filtered Back Projection (FBP) reconstruction. Ultimately, this method enables accurate 2D slice recovery even from severely incomplete inputs.

---

## Installation and Setup

1. **Clone the repository:**
```bash
   git clone [https://github.com/AlbertoCaschi/Electron-Tomography-Reconstruction.git](https://github.com/AlbertoCaschi/Electron-Tomography-Reconstruction.git)

```

2. **Navigate to the core directory and install dependencies:**
The required libraries are listed in the `VAEResNet` folder. It is recommended to use a virtual environment.
```bash
cd Electron-Tomography-Reconstruction/VAEResNet
pip install -r requirements.txt

```



---

## Training the Model

All training parameters and settings are managed via a Python dictionary inside the `config.py` file.

1. Open `VAEResNet/config.py` and adjust the training settings to your needs.
2. Start the training process by running:
```bash
python config.py

```



**Training Features & Outputs:**

* **Resumable:** You can stop the training process and safely restart it at any time.
* **Logs:** The model evaluates its reconstruction performance every *n* epochs. You can find these visualizations and the loss curves in the `logs/` folder.
* **Checkpoints:** The latest and best-performing model weights are automatically saved in the `checkpoints/` folder.

---

## Local Inference

To reconstruct a 2D slice from a sinogram using a trained model, use the `inference.py` script.

1. Ensure you have your desired model weights selected.
2. If you want to test the model immediately, sample sinograms are provided in the `dataset/test/` directory.
3. Run the inference script:
```bash
python inference.py

```


*(Note: You will need to select the specific model and input sinogram within the script).*

4. The reconstructed outputs will be saved automatically in the `dataset/reconstructions/` folder.
