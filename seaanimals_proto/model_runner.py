import streamlit as st
import os 
from pathlib import Path
import shutil
import tempfile
import subprocess
import pandas as pd
from model_pipeline import process_video_with_model

# Configure Streamlit to allow larger file uploads
st.set_page_config(
    page_title="Sea Animals Video Processor",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.image("natgeobanner.png", use_container_width=True)

# ==========================================
# INITIALIZE SESSION STATE 
# ==========================================
if "is_processed" not in st.session_state:
    st.session_state.is_processed = False
if "final_video_path" not in st.session_state:
    st.session_state.final_video_path = None
if "final_csv_path" not in st.session_state:
    st.session_state.final_csv_path = None
if "final_zip_path" not in st.session_state:
    st.session_state.final_zip_path = None
if "max_n" not in st.session_state:
    st.session_state.max_n = 0
if "last_uploaded_key" not in st.session_state:
    st.session_state.last_uploaded_key = None

# ---------------------------------------------------------
# UI SIDEBAR FILTERS
# ---------------------------------------------------------
st.sidebar.header("Model Settings & Filters")
st.sidebar.markdown("Adjust these parameters to fine-tune the tracking pipeline.")

# --- THE MODEL SELECTION BRIDGE ---
selected_model = st.sidebar.radio(
    "Choose AI Model:",
    ("Default (YOLO-World)", "Fine-Tuned (Trained On JAMSTEC Dataset)")
)
st.sidebar.markdown("---")

conf_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.15, step=0.05)
iou_threshold = st.sidebar.slider("IOU (Overlap) Threshold", min_value=0.0, max_value=1.0, value=0.70, step=0.05)
apply_clahe = st.sidebar.checkbox("Apply CLAHE", value=True)
trim_black = st.sidebar.checkbox("Trim Black Frames", value=True)
frame_skip = st.sidebar.number_input("Frame Skip (1 = every frame, 3 = skip 2)", min_value=1, max_value=10, value=2)

def process_video(input_path, progress_bar, status_text, model_choice, original_filename):
    """
    Process video file with ML model and convert to compatible format.
    """
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    temp_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name
    temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip").name

    def update_ui(percent, text):
        progress_bar.progress(percent)
        status_text.text(text)

    # Running YOLO model processing with live UI updates and the selected model
    max_animals_counted = process_video_with_model(
        input_path=input_path, 
        output_path=temp_output, 
        model_choice=model_choice,
        original_filename=original_filename, # <-- Passes the original filename to the backend
        progress_callback=update_ui,
        confidence=conf_threshold,
        iou=iou_threshold,
        apply_clahe=apply_clahe,
        trim_black=trim_black,
        frame_skip=frame_skip,
        csv_path=temp_csv,
        zip_path=temp_zip
    )

    update_ui(1.0, "Wrapping up final video formatting...")
    
    final_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    subprocess.run([
        "ffmpeg", "-i", temp_output,
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-c:a", "aac",
        "-y", final_output
    ], capture_output=True, check=True)

    os.unlink(temp_output) 
    
    return final_output, temp_csv, temp_zip, max_animals_counted

# ---------------------------------------------------------
# MAIN APP FLOW
# ---------------------------------------------------------
st.title("Sea Animals Video Processor")

uploaded_file = st.file_uploader("Upload a video file (.mov or .mp4)", type=["mov", "mp4"])

if uploaded_file is not None:
    current_upload_key = f"{uploaded_file.name}:{uploaded_file.size}"
    if st.session_state.last_uploaded_key != current_upload_key:
        st.session_state.last_uploaded_key = current_upload_key
        st.session_state.is_processed = False
        st.session_state.final_video_path = None
        st.session_state.final_csv_path = None
        st.session_state.final_zip_path = None
        st.session_state.max_n = 0

    if st.button("Process Video"):
        original_name = uploaded_file.name # <-- Grab the real name right here!
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_input:
            tmp_input.write(uploaded_file.getbuffer())
            input_path = tmp_input.name
        
        st.info(f"Initializing {selected_model} and starting pipeline...")
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        
        try:
            # Pass original_name into process_video
            output_vid, output_csv, output_zip, max_n = process_video(input_path, progress_bar, status_text, selected_model, original_name)
            
            # Save results to session state
            st.session_state.final_video_path = output_vid
            st.session_state.final_csv_path = output_csv
            st.session_state.final_zip_path = output_zip
            st.session_state.max_n = max_n
            st.session_state.is_processed = True
            st.success("✓ Video processed!")
        except Exception as e:
            st.session_state.is_processed = False
            st.error(f"Video processing failed: {e}")
        finally:
            # Clean up the UI elements once processing finishes
            status_text.empty()
            progress_bar.empty()
        
    # ==========================================
    # RENDER UI OUTSIDE THE BUTTON BLOCK
    # ==========================================
    if st.session_state.is_processed:

        output_vid = st.session_state.final_video_path
        output_csv = st.session_state.final_csv_path
        output_zip = st.session_state.final_zip_path
        max_n = st.session_state.max_n

        if not output_vid or not os.path.exists(output_vid):
            st.error("Processed video is no longer available. Please run processing again.")
            st.session_state.is_processed = False
            st.stop()

        # Display the MaxN Metric
        st.markdown("---")
        st.metric(label="MaxN (Maximum Animals in a Single Frame)", value=max_n)
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Download Assets")
            
            # Video Download
            with open(output_vid, "rb") as f:
                # <-- Ensure downloaded video gets the original name + "_processed"
                base_name = uploaded_file.name.rsplit('.', 1)[0]
                export_name = f"{base_name}_processed.mp4"
                
                st.download_button(
                    label="🎥 Download Processed Video",
                    data=f.read(),
                    file_name=export_name,
                    mime="video/mp4",
                    use_container_width=True
                )
                
            # CSV Download
            if os.path.exists(output_csv) and os.path.getsize(output_csv) > 0:
                with open(output_csv, "rb") as f:
                    st.download_button(
                        label="📊 Download Cluster Data (CSV)",
                        data=f.read(),
                        file_name="animal_detections.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            # ZIP Download (Cropped Bounding Boxes)
            if os.path.exists(output_zip) and os.path.getsize(output_zip) > 0:
                with open(output_zip, "rb") as f:
                    st.download_button(
                        label="🖼️ Download Representative Crops (ZIP)",
                        data=f.read(),
                        file_name="representative_crops.zip",
                        mime="application/zip",
                        use_container_width=True
                    )

            st.markdown("""<style>div.stDownloadButton > button {height: 60px; font-size: 18px;}</style>""", unsafe_allow_html=True)

        with col2:
            st.subheader("Video Preview")
            st.video(output_vid)

        # ==========================================
        # DEMO PREVIEW TABS
        # ==========================================
        st.markdown("---")
        st.subheader("Live Data Extraction Preview")
        
        # Create tabs to keep the UI clean
        tab1, tab2 = st.tabs(["📊 Cluster Data (CSV)", "🖼️ Representative Crops"])
        
        # TAB 1: Interactive CSV Preview
        with tab1:
            if os.path.exists(output_csv) and os.path.getsize(output_csv) > 0:
                # Load the CSV using Pandas
                df = pd.read_csv(output_csv)
                # Display an interactive, scrollable table
                st.dataframe(df, use_container_width=True, height=300)
                st.caption(f"Currently tracking {len(df)} total detection instances across all frames.")
            else:
                st.info("No data available to preview.")
                
        # TAB 2: Image Gallery of the ZIP Crops
        with tab2:
            if os.path.exists(output_zip) and os.path.getsize(output_zip) > 0:
                import zipfile
                
                with zipfile.ZipFile(output_zip, 'r') as zf:
                    # Find all the .jpg files inside the zip
                    image_names = [name for name in zf.namelist() if name.endswith('.jpg')]
                    
                    if image_names:
                        st.write("Previewing first 6 unique species detections:")
                        # Create dynamic columns for the gallery (max 6 images)
                        num_images_to_show = min(6, len(image_names))
                        cols = st.columns(num_images_to_show)
                        
                        for i in range(num_images_to_show):
                            img_name = image_names[i]
                            # Read the image bytes directly from the zip without extracting to disk
                            with zf.open(img_name) as img_file:
                                img_bytes = img_file.read()
                                
                                # The file name is usually like 'id5_fish_f30.jpg'
                                # Let's extract just the class name for the caption
                                try:
                                    label = img_name.split('_')[1].capitalize()
                                except:
                                    label = "Cropped Image"
                                cols[i].image(img_bytes, caption=label, use_container_width=True)
                    else:
                        st.info("No images were cropped in this run.")
            else:
                st.info("No image crops available to preview.")