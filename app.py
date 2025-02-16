import streamlit as st
import os
import tempfile
from zipfile import ZipFile
import sys
from pathlib import Path

def check_vtk_dependencies():
    """Check if VTK and its system dependencies are available"""
    try:
        import vtk
        # Try to create a simple VTK object to test functionality
        renderer = vtk.vtkRenderer()
        return True
    except ImportError:
        return "VTK is not installed. Please install it with: pip install vtk"
    except Exception as e:
        if "libXrender" in str(e):
            return ("Missing system library: libXrender. Please install it with:\n"
                   "Ubuntu/Debian: sudo apt-get install libxrender1\n"
                   "CentOS/RHEL: sudo yum install libXrender\n"
                   "macOS: brew install xquartz")
        return f"VTK dependency error: {str(e)}"

def check_python_dependencies():
    """Check if required Python packages are installed"""
    dependencies = {
        'monai': 'monai',
        'torch': 'torch',
        'numpy': 'numpy',
        'scikit-image': 'skimage'
    }
    
    missing = []
    for package, import_name in dependencies.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(package)
    
    return missing

def create_zip_file(directory_path, output_filename):
    """Create a ZIP file from the directory contents"""
    with ZipFile(output_filename, 'w') as zipf:
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, directory_path)
                zipf.write(file_path, arcname)

def check_segmentation_file():
    """Check if segmentation.py exists in the correct location"""
    current_dir = Path(__file__).parent
    segmentation_path = current_dir / "segmentation.py"
    return segmentation_path.exists()

def main_app():
    st.title("3D Medical Image Segmentation")
    
    # Check Python dependencies
    missing_packages = check_python_dependencies()
    if missing_packages:
        st.error("Missing required Python packages!")
        st.code(f"pip install {' '.join(missing_packages)}", language="bash")
        st.info("Please install the missing packages and restart the application.")
        return

    # Check VTK and system dependencies
    vtk_status = check_vtk_dependencies()
    if vtk_status is not True:
        st.error("VTK dependency check failed!")
        st.text(vtk_status)
        return

    # Check if segmentation.py exists
    if not check_segmentation_file():
        st.error("segmentation.py not found!")
        st.info("Please ensure segmentation.py is in the same directory as this Streamlit app.")
        return

    # Only import segmentation after all checks pass
    try:
        from segmentation import main as segmentation_main
    except ImportError as e:
        st.error(f"Failed to import segmentation module: {str(e)}")
        st.info("Please check that segmentation.py contains a properly defined 'main' function.")
        return

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
                st.error("An error occurred during processing!")
                st.error(str(e))
                st.info("Please check that your input file is valid and all dependencies are properly installed.")

if __name__ == "__main__":
    main_app()