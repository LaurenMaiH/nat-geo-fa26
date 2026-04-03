import ultralytics
from ultralytics import YOLOWorld
import cv2
import numpy as np
import os
import yaml
import shutil

# Setup YOLO World model (initialize once)
model = YOLOWorld('yolov8x-worldv2.pt')
custom_vocabulary = ["fish", "eel", "ray", "shark", "jellyfish", "animal", "shrimp", "crab", "lobster", "isopod", "octopus", "squid", "mollusk", "crustacean", "animal cluster"]
model.set_classes(custom_vocabulary)

# Setup bytetrack config (initialize once)
yaml_path = os.path.join(os.path.dirname(ultralytics.__file__), "cfg", "trackers", "bytetrack.yaml")
shutil.copy(yaml_path, "bytetrack_custom.yaml")
with open("bytetrack_custom.yaml", "r") as f:
    config = yaml.safe_load(f)
config["track_buffer"] = 120
config["match_thresh"] = 0.85
with open("bytetrack_custom.yaml", "w") as f:
    yaml.dump(config, f)

def skip_black_frames(cap, threshold=10, require_consecutive=3):
    consecutive_bright = 0
    first_real_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        brightness = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()
        if brightness > threshold:
            consecutive_bright += 1
        else:
            consecutive_bright = 0
        if consecutive_bright >= require_consecutive:
            first_real_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - require_consecutive
            break
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_real_frame)
    return first_real_frame

def clahe_L_median(frame, clahe):
    median_blur_frame = cv2.medianBlur(frame, 3)
    # Add missing clahe_on_l_channel_LAB function if needed
    return median_blur_frame

def perform_inference(video_path, preprocess, output_path="output_video.mp4", skip_black=(10, 3), sample_interval=3, confidence=0.15, iou=0.7, bg_thresh=40, tile_grid_size=(8,8)):
    N_BG_SAMPLES = 20
    MIN_BLOB_AREA = 500
    INFER_SIZE = 1280

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)

    first_real_frame = 0
    if skip_black:
        black_thresh, require_consecutive = skip_black
        first_real_frame = skip_black_frames(cap, black_thresh, require_consecutive)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - first_real_frame
    sample_indices = sorted(set(np.linspace(0, total_frames - 1, N_BG_SAMPLES, dtype=int) + first_real_frame))

    bg_frames = []
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            bg_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32))

    if not bg_frames:
        cap.release()
        raise RuntimeError("No background frames could be read.")

    median_bg = np.median(np.stack(bg_frames), axis=0).astype(np.uint8)
    median_bg = cv2.GaussianBlur(median_bg, (5, 5), 0)

    output_fps = orig_fps / sample_interval
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height))

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=tile_grid_size)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    cap.set(cv2.CAP_PROP_POS_FRAMES, first_real_frame)
    frame_idx = 0
    frames_inferred = 0
    frames_skipped_by_gate = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_interval == 0:
            processed_bgr = preprocess(frame, clahe)
            infer_frame = cv2.resize(processed_bgr, (INFER_SIZE, int(INFER_SIZE * height / width)))

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray, median_bg)
            _, fg_mask = cv2.threshold(diff, bg_thresh, 255, cv2.THRESH_BINARY)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, morph_kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, morph_kernel)

            _, _, stats, _ = cv2.connectedComponentsWithStats(fg_mask)
            has_activity = any(s[cv2.CC_STAT_AREA] >= MIN_BLOB_AREA for s in stats[1:])

            if has_activity:
                frames_inferred += 1
                results = model.track(infer_frame, persist=True, tracker="bytetrack_custom.yaml", verbose=False, conf=confidence, iou=iou)
                if results[0].boxes.id is not None:
                    annotated = results[0].plot()
                    frame_out = cv2.resize(annotated, (width, height))
                else:
                    frame_out = processed_bgr
            else:
                frames_skipped_by_gate += 1
                frame_out = processed_bgr

            out.write(frame_out)

        frame_idx += 1

    cap.release()
    out.release()
    print(f"Video saved as {output_path}.")

def process_video_with_model(input_path, output_path):
    """Main function called by Streamlit app"""
    perform_inference(input_path, clahe_L_median, output_path=output_path)