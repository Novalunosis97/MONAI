import streamlit as st
import os
import tempfile
import subprocess
from zipfile import ZipFile

def install_system_dependencies():
    """Install required system libraries for VTK"""
    try:
        subprocess.run(['apt-get', 'update'], check=True)
        subprocess.run(['apt-get', 'install', '-y', 'libxrender1'], check=True)
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"Failed to install system dependencies: {str(e)}")
        return False

def check_dependencies():
    """Check if all required Python packages are installed"""
    required_packages = ['monai', 'torch', 'vtk', 'numpy', 'skimage']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_python_dependencies(packages):
    """Install missing Python packages"""
    try:
        for package in packages:
            subprocess.run(['pip', 'install', package], check=True)
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"Failed to install Python dependencies: {str(e)}")
        return False

def create_zip_file(directory_path, output_filename):
    """Create a ZIP file from the directory contents"""
    with ZipFile(output_filename, 'w') as zipf:
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, directory_path)
                zipf.write(file_path, arcname)

def main_app():
    st.title("3D Medical Image Segmentation")
    st.write("Upload a medical image file (.nii format) for 3D segmentation")

    # Check and install dependencies
    with st.spinner("Checking dependencies..."):
        # First install system dependencies
        if not install_system_dependencies():
            st.error("Failed to install required system libraries. Please contact support.")
            return

        # Then check and install Python packages
        missing_packages = check_dependencies()
        if missing_packages:
            st.info(f"Installing required packages: {', '.join(missing_packages)}")
            if not install_python_dependencies(missing_packages):
                st.error("Failed to install required Python packages. Please contact support.")
                return

    # Only import segmentation after ensuring dependencies are installed
    try:
        from segmentation import main as segmentation_main
    except ImportError as e:
        st.error(f"Failed to import segmentation module: {str(e)}")
        st.info("Please ensure the segmentation.py file is in the correct location.")
        return

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
                    # Call the segmentation function
                    segmentation_main(input_path, temp_output_dir)

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
                st.info("Please check the logs for more details and ensure all dependencies are properly installed.")

if __name__ == "__main__":
    main_app()