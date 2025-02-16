import streamlit as st
import os
import tempfile
from zipfile import ZipFile
import numpy as np
import torch
from monai.transforms import LoadImage, EnsureChannelFirst, Orientation
from monai.bundle import ConfigParser, download

def create_zip_file(directory_path, output_filename):
    """Create a ZIP file from the directory contents"""
    with ZipFile(output_filename, 'w') as zipf:
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, directory_path)
                zipf.write(file_path, arcname)

def process_image(input_file, output_directory):
    """Process a single medical image file without VTK visualization"""
    os.makedirs(output_directory, exist_ok=True)
    
    # Load and preprocess the image
    preprocessing_pipeline = LoadImage(image_only=True)
    CT = preprocessing_pipeline(input_file)
    
    # Add channel dimension if needed
    if len(CT.shape) == 3:
        CT = np.expand_dims(CT, 0)
    
    # Convert to tensor
    CT = torch.from_numpy(CT).float()
    
    # Download and load the model
    datadir = os.path.dirname(os.path.abspath(__file__))
    model_name = "wholeBody_ct_segmentation"
    download(name=model_name, bundle_dir=datadir)
    model_path = os.path.join(datadir, 'model_lowres.pt')
    
    # Load model configuration
    config_path = os.path.join(datadir, 'inference.json')
    config = ConfigParser()
    config.read_config(config_path)
    
    # Set up preprocessing and model
    preprocessing = config.get_parsed_content("preprocessing")
    data = preprocessing({'image': CT})
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = config.get_parsed_content("network")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Run inference
    with torch.no_grad():
        data['pred'] = model(data['image'].unsqueeze(0).to(device))
        data['pred'] = data['pred'][0]
    
    # Save predictions
    pred_path = os.path.join(output_directory, 'predictions.npy')
    np.save(pred_path, data['pred'].cpu().numpy())
    
    # Save metadata
    metadata = {
        'input_shape': list(CT.shape),
        'output_shape': list(data['pred'].shape),
        'device_used': str(device)
    }
    
    import json
    metadata_path = os.path.join(output_directory, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    return data['pred']

def main_app():
    st.title("Medical Image Segmentation")
    st.write("Upload a medical image file (.nii format) for segmentation")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a .nii file", type=['nii'])
    
    if uploaded_file is not None:
        with tempfile.TemporaryDirectory() as temp_input_dir, \
             tempfile.TemporaryDirectory() as temp_output_dir:
            
            # Save uploaded file
            input_path = os.path.join(temp_input_dir, uploaded_file.name)
            with open(input_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            
            try:
                with st.spinner('Processing... This may take several minutes.'):
                    # Process the image
                    predictions = process_image(input_path, temp_output_dir)
                    
                    # Create visualization of results (optional)
                    st.write("Segmentation complete!")
                    
                    # Show some basic statistics
                    st.write("Segmentation Statistics:")
                    st.write(f"- Number of classes detected: {len(np.unique(predictions.argmax(0)))}")
                    st.write(f"- Output shape: {predictions.shape}")
                    
                    # Create ZIP file of results
                    zip_path = os.path.join(temp_input_dir, 'results.zip')
                    create_zip_file(temp_output_dir, zip_path)
                    
                    # Provide download button
                    with open(zip_path, 'rb') as f:
                        st.download_button(
                            label="Download Results",
                            data=f.read(),
                            file_name="segmentation_results.zip",
                            mime="application/zip"
                        )
                    
                    st.success('Processing complete! Click above to download your results.')
            
            except Exception as e:
                st.error("An error occurred during processing!")
                st.error(str(e))
                st.info("Please ensure your input file is valid and try again.")

if __name__ == "__main__":
    main_app()