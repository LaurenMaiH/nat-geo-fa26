import streamlit as st
import os 
from pathlib import Path
import shutil
import tempfile

# Configure Streamlit to allow larger file uploads
st.set_page_config(
    page_title="Sea Animals Video Processor",
    layout="wide",
    initial_sidebar_state="expanded"
)

def process_video(input_path):
    """
    Process video file. Currently returns the same video.
    Future: Add ML model processing here.
    """
    # For now, just return the input path (same video)
    return input_path

st.title("Sea Animals Video Processor")

uploaded_file = st.file_uploader("Upload a video file (.mov or .mp4)", type=["mov", "mp4"])

if uploaded_file is not None:

    if st.button("Process Video"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_input:
            tmp_input.write(uploaded_file.getbuffer())
            input_path = tmp_input.name
        
        with st.spinner("Processing video..."):
            output_path = process_video(input_path)
        
        st.success("✓ Video processed!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Download")
            with open(output_path, "rb") as f:
                st.download_button(
                    label="Processed Video Ready to Download! :)",
                    data=f.read(),
                    file_name="processed_video.mp4",
                    mime="video/mp4"
                )
        
        with col2:
            st.subheader("Video Preview")
            st.video(output_path)
        
        os.unlink(input_path)