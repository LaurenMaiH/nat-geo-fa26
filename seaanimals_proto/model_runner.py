import streamlit as st
import os 
from pathlib import Path
import shutil
import tempfile
import subprocess
from model_pipeline import process_video_with_model

# Configure Streamlit to allow larger file uploads
st.set_page_config(
    page_title="Sea Animals Video Processor",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.set_page_config(layout="wide")
st.image("natgeobanner.png", use_container_width=True)

def process_video(input_path):
    """
    Process video file with ML model and convert to compatible format.
    """
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

    #running YOLO model processing 
    process_video_with_model(input_path, temp_output)

    # For now, just return the input path (same video)
    final_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    subprocess.run([
        "ffmpeg", "-i", temp_output,
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-c:a", "aac",
        "-y", final_output
    ], capture_output=True, check=True)

    os.unlink(temp_output) # clean temp file
    return final_output 

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
                    mime="video/mp4",
                    use_container_width=True
                )
            st.markdown("""
                <style>
                div.stDownloadButton > button 
                {height: 100px; font-size: 50px;} </style>""",
                 unsafe_allow_html=True)

        with col2:
            st.subheader("Video Preview")
            st.video(output_path)
