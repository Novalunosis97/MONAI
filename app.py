import streamlit as st
import os
import tempfile
from zipfile import ZipFile
from segmentation import main  # Import the main function from your segmentation code

def create_zip_file(directory_path, output_filename):
    with ZipFile(output_filename, 'w') as zipf:
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, directory_path)
                zipf.write(file_path, arcname)

def main_app():
    st.title("3D Medical Image Segmentation")
    st.write("Upload a medical image file (.nii format) for 3D segmentation")

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
                    # Call the main function from your segmentation code
                    main(input_path, temp_output_dir)

                    # Create ZIP file of results
                    zip_path = os.path.join(temp_input_dir, 'results.zip')
                    create_zip_file(temp_output_dir, zip_path)

                    # Provide download button
                    with open(zip_path, 'rb') as f:
                        st.download_button(
                            label="Download Processed Files",
                            data=f.read(),
                            file_name="segmentation_results.zip",
                            mime="application/zip"
                        )

                    st.success('Processing complete! Click above to download your results.')

            except Exception as e:
                st.error(f'An error occurred during processing: {str(e)}')

if __name__ == "__main__":
    main_app()