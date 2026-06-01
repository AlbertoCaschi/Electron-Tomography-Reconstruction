import streamlit as st
from PIL import Image
import numpy as np
import time

# 1. Page Configuration (Modern Layout)
st.set_page_config(
    page_title="ET Reconstruction Lab",
    page_icon="🔬",
    layout="wide", # Use full width for side-by-side comparison
    initial_sidebar_state="expanded"
)

# 2. Add custom CSS for modern aesthetics (Optional)
st.markdown("""
<style>
    .big-font { font-size: 24px !important; font-weight: bold; }
    .reportview-container .main .block-container { padding-top: 1rem; }
    div[data-testid="stToolbar"] { display: none;} /* Hide top toolbar */
</style>
""", unsafe_allow_html=True)

# 3. Main Title and Intro
st.markdown('<p class="big-font">Electron Tomography Reconstruction Dashboard</p>', unsafe_allow_html=True)
st.markdown("---")

# 4. Sidebar: Inputs and Configuration
with st.sidebar:
    st.header("Project Configuration")
    st.write("University Project: [Your University Name]")
    
    # Required Input A: Upload Sinogram
    st.subheader("1. Input Data")
    uploaded_sinogram = st.file_uploader(
        "Upload limited-angle sinogram (.png, .jpg)", 
        type=["png", "jpg", "jpeg"]
    )
    
    # Required Input B: Missing Wedge Dropdown
    st.subheader("2. Imaging Parameters")
    missing_wedge_config = st.selectbox(
        "Select Missing Wedge Configuration",
        ("± 30 Degrees (60° total)", 
         "± 40 Degrees (80° total)", 
         "± 45 Degrees (90° total)",
         "± 50 Degrees (100° total)")
    )
    
    # Action Button
    run_button = st.button("Start AI Reconstruction", type="primary")

# 5. Main Content Area: Displays and Results
if uploaded_sinogram is not None:
    
    # We use columns to layout the input vs. output side-by-side
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Sinogram")
        sinogram_image = Image.open(uploaded_sinogram)
        
        # We need to process this image for the model (e.g., convert to gray NumPy array)
        # We do this here, but keep the PIL image for displaying
        sinogram_array = np.array(sinogram_image.convert('L')) # Convert to 8-bit Gray
        
        st.image(sinogram_image, caption="Uploaded Limited-Angle Sinogram", use_column_width=True)
        st.write(f"Sinogram Dimensions: {sinogram_array.shape[1]} (Angles) x {sinogram_array.shape[0]} (Detectors)")
        
    with col2:
        st.subheader("AI Reconstruction Result")
        
        if run_button:
            # 6. Connecting the Inputs to your Model (Zero-Install Step)
            with st.spinner(f"Reconstructing volume (assuming {missing_wedge_config})..."):
                
                # We simulate time taken for inference
                # In your real code, this is where you call your function
                # e.g., result_array = my_trained_model.predict(sinogram_array, angle_config=missing_wedge_config)
                
                # --- [INSERT YOUR MODEL CODE HERE] ---
                # Example Placeholder: Replace this with your actual reconstruction call.
                # Since we don't have the real model, we will simulate a result:
                time.sleep(3) # Mocking model runtime
                
                # Create a sample "reconstructed" result from the input (mock logic)
                mock_result_arr = np.sqrt(np.mean(sinogram_array)) * np.ones((512, 512), dtype=np.uint8)
                # Apply a slight gradient for visual difference
                x, y = np.meshgrid(np.linspace(0, 255, 512), np.linspace(0, 255, 512))
                mock_result_arr = (mock_result_arr + (x+y)/2).astype(np.uint8)
                
                # Final output must be converted back to an Image for display
                reconstructed_image = Image.fromarray(mock_result_arr)
                # --- [END OF MOCK CODE] ---

                # 7. REQUIRED DISPLAY of final result (.png image)
                st.success("Reconstruction Complete!")
                st.image(
                    reconstructed_image, 
                    caption=f"AI Reconstruction (Missing Wedge: {missing_wedge_config})", 
                    use_column_width=True
                )
                
                # Option to download the reconstructed image
                # (Optional, but very helpful for scientific collaborators)
                st.download_button(
                    label="Download Reconstructed PNG",
                    data=reconstructed_image.tobytes(), # Need to properly format bytes
                    file_name="et_reconstruction.png",
                    mime="image/png"
                )
        else:
            st.info("Configure settings in the sidebar and click 'Start AI Reconstruction' to view the output.")
            st.warning("Your final reconstruction (.png) will be displayed in this panel.")

else:
    # Initial state when the app loads
    st.info("👈 Please upload a limited-angle sinogram in the sidebar to get started.")
    # You can include an example image or description here.