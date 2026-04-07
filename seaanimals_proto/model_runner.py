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
st.image("natgeobanner.png", use_container_width=True)

def process_video(input_path, progress_bar, status_text):
    """
    Process video file with ML model and convert to compatible format.
    """
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

    def update_ui(percent, text):
        progress_bar.progress(percent)
        status_text.text(text)

    # running YOLO model processing with live UI updates
    process_video_with_model(input_path, temp_output, progress_callback=update_ui)

    update_ui(1.0, "Wrapping up final video formatting...")
    
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
        
        st.info("Initializing models and starting pipeline...")
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        
        output_path = process_video(input_path, progress_bar, status_text)
        
        # Clean up the UI elements once processing finishes
        status_text.empty()
        progress_bar.empty()
        
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