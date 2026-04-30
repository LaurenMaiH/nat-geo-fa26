import os
import sys
import subprocess
import threading
import tempfile
from pathlib import Path
from tqdm import tqdm
import streamlit as st
import ultralytics
from ultralytics import YOLOWorld, settings
import cv2
import numpy as np
import yaml
import shutil
import csv
import zipfile
import gdown

# # --- GLOBAL CACHE SETUP --- COMMENTED OUT FOR SIMPLICITY USE LOCAL MODEL SETUP INSTEAD 
# YOLO_CACHE_DIR = Path.home() / ".config" / "ultralytics"
# YOLO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
# settings.update({'weights_dir': str(YOLO_CACHE_DIR), 'runs_dir': str(YOLO_CACHE_DIR / "runs")})

# --- LOCAL MODEL SETUP ---
# Use models located in the repo directory (next to this file).
LOCAL_MODEL_DIR = Path(__file__).parent
YOLO_CACHE_DIR = LOCAL_MODEL_DIR
settings.update({'weights_dir': str(YOLO_CACHE_DIR), 'runs_dir': str(YOLO_CACHE_DIR / "runs")})

# # --- MODEL & CONFIG INITIALIZATION (GLOBAL CACHE SETUP: COMMENTED OUT FOR SIMPLICITY) ---
# @st.cache_resource(show_spinner=False)
# def load_model_and_config(model_choice):
#     if model_choice == "Default (YOLO-World)":
#         model_path = YOLO_CACHE_DIR / "yolov8x-worldv2.pt"
#         if not model_path.exists():
#             with st.spinner("Downloading YOLO-World Model... (This will only happen once)"):
#                 print(f"Default model not found in cache. Downloading to {model_path}...")
#                 url = 'https://drive.google.com/file/d/1hh576zOzpUqgWSjIdSkR4EwgYQ8kH406/view?usp=sharing'
#                 gdown.download(url, str(model_path), quiet=False)
#                 st.success("Model downloaded successfully!")
#         print(f"Loading Default Model from: {model_path}")
#         model = YOLOWorld(str(model_path))
#         model.model.eval()
#         custom_vocabulary = [
#             "fish", "eel", "ray", "shark", "jellyfish", "animal", "shrimp", 
#             "crab", "lobster", "isopod", "octopus", "squid", "mollusk", 
#             "crustacean", "animal cluster"
#         ]
#         model.set_classes(custom_vocabulary)
#     else:
#         model_path = YOLO_CACHE_DIR / "yolov8x-jamstec.pt"
#         if not model_path.exists():
#             with st.spinner("Downloading 150MB Deep Sea Model... (This will only happen once)"):
#                 print(f"Custom model not found in cache. Downloading to {model_path}...")
#                 url = 'https://drive.google.com/file/d/1rCyq4GZcG5UCqrl2SNNjmaHkLOZbWgyv/view?usp=sharing'
#                 gdown.download(url, str(model_path), quiet=False)
#                 st.success("Model downloaded successfully!")
#         print(f"Loading Fine-Tuned Model from: {model_path}")
#         model = YOLOWorld(str(model_path))
#         model.model.eval()

# --- MODEL & CONFIG INITIALIZATION (LOCAL MODEL SETUP) --- 
@st.cache_resource(show_spinner=False)
def load_model_and_config(model_choice):
    local_dir = Path(__file__).parent
    if model_choice == "Default (YOLO-World)":
        model_path = local_dir / "yolov8x-worldv2.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Default model not found at {model_path}. Place yolov8x-worldv2.pt next to model_pipeline.py")
        print(f"Loading Default Model from: {model_path}")
        model = YOLOWorld(str(model_path))
        model.model.eval()
        custom_vocabulary = [
            "fish", "eel", "ray", "shark", "jellyfish", "animal", "shrimp", 
            "crab", "lobster", "isopod", "octopus", "squid", "mollusk", 
            "crustacean", "animal cluster"
        ]
        model.set_classes(custom_vocabulary)
    else:
        model_path = local_dir / "yolov8x-jamstec.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Custom model not found at {model_path}. Place yolov8x-jamstec.pt next to model_pipeline.py")
        print(f"Loading Fine-Tuned Model from: {model_path}")
        model = YOLOWorld(str(model_path))
        model.model.eval()

    yaml_path = os.path.join(os.path.dirname(ultralytics.__file__), "cfg", "trackers", "bytetrack.yaml")
    custom_yaml_path = Path(__file__).parent / "bytetrack_custom.yaml"

    if not custom_yaml_path.exists():
        shutil.copy(yaml_path, custom_yaml_path)
        with open(custom_yaml_path, "r") as f:
            config = yaml.safe_load(f)
        config["track_buffer"] = 60 
        config["match_thresh"] = 0.85
        with open(custom_yaml_path, "w") as f:
            yaml.dump(config, f)
        print(f"Created custom tracker config at: {custom_yaml_path}")
            
    return model, str(custom_yaml_path)

# --- PREPROCESSING FUNCTIONS ---
def skip_black_frames(cap, threshold=10, require_consecutive=3):
    consecutive_bright = 0
    first_real_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        brightness = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()
        if brightness > threshold: consecutive_bright += 1
        else: consecutive_bright = 0
        if consecutive_bright >= require_consecutive:
            first_real_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - require_consecutive
            break
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_real_frame)
    return first_real_frame

def clahe_on_l_channel_LAB(frame, clahe):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b_ch = cv2.split(lab)
    lab = cv2.merge([clahe.apply(l), a, b_ch])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def clahe_L_median(frame, clahe):
    median_blur_frame = cv2.medianBlur(frame, 3)
    return clahe_on_l_channel_LAB(median_blur_frame, clahe)

# --- VIDEO RESIZING CAPABILITIES ---
def detect_encoder() -> tuple[str, list[str]]:
    probes = [('h264_nvenc', ['-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda']), ('h264_vaapi', ['-hwaccel', 'vaapi', '-vaapi_device', '/dev/dri/renderD128'])]
    null = subprocess.DEVNULL
    for enc, hw_flags in probes:
        probe_cmd = ['ffmpeg', '-y', '-f', 'lavfi', '-i', 'testsrc=duration=1:size=3840x2160', *hw_flags, '-vframes', '1', '-c:v', enc, '-f', 'null', '-']
        if subprocess.run(probe_cmd, stdout=null, stderr=null).returncode == 0: return enc, hw_flags
    return 'libx264', []

def _build_scale_filter(encoder: str, target_height: int) -> str:
    target_width = -2  
    if 'vaapi' in encoder: return f'scale_vaapi=w={target_width}:h={target_height}:force_original_aspect_ratio=decrease'
    elif 'nvenc' in encoder: return f'scale_cuda={target_width}:{target_height}:force_original_aspect_ratio=decrease,hwdownload,format=nv12'
    else: return f'scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,format=yuv420p'

def _get_frame_count(path: Path) -> int:
    r = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_packets', '-show_entries', 'stream=nb_read_packets', '-of', 'csv=p=0', str(path)], capture_output=True, text=True)
    val = r.stdout.strip()
    return int(val) if val.isdigit() else 0

def resize_video(input_path, output_path, crf=23, encoder=None, hw_flags=None, force_cpu=False, target_height=1280, progress_callback=None) -> bool:
    input_path, output_path = Path(input_path), Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if encoder is None:
        if force_cpu: encoder, hw_flags = 'libx264', []
        else: encoder, hw_flags = detect_encoder()
    hw_flags = hw_flags or []
    vf = _build_scale_filter(encoder, target_height)
    extra = []
    if encoder == 'libx264': extra = ['-crf', str(crf), '-preset', 'fast']
    elif encoder == 'h264_nvenc': extra = ['-cq',  str(crf), '-preset', 'p4']
    elif encoder == 'h264_vaapi': extra = ['-qp',  str(crf)]
    total_frames = _get_frame_count(input_path)
    cmd = ['ffmpeg', '-y', *hw_flags, '-i', str(input_path), '-vf', vf, '-c:v', encoder, *extra, '-an', '-movflags', '+faststart', '-progress', 'pipe:1', '-nostats', str(output_path)]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stderr_buf = []
    threading.Thread(target=lambda: stderr_buf.extend(process.stderr.readlines()), daemon=True).start()
    with tqdm(total=total_frames or None, desc=f'  Compressing {input_path.name}', unit='frame', leave=True) as pbar:
        last = 0
        for line in process.stdout:
            if line.startswith('frame='):
                val = line.split('=')[1].strip()
                if val.isdigit():
                    current = int(val)
                    pbar.update(current - last)
                    last = current
                    if progress_callback and total_frames > 0:
                        progress_callback(min(current / total_frames, 1.0), f"Compressing video... ({current}/{total_frames} frames)")
        process.wait()
    if process.returncode != 0: return False
    return True

# --- ANNOTATION FUNCTION ---
def draw_boxes_no_labels(frame, boxes, track_ids=None, thickness=6):
    annotated = frame.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        if track_ids is not None and len(track_ids) > i:
            np.random.seed(int(track_ids[i]))
            box_color = tuple(np.random.randint(100, 255, 3).tolist())
        else:
            box_color = (0, 255, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, thickness)
    return annotated

# --- MAIN INFERENCE PIPELINE ---
def perform_inference(video_path, model, tracker_config, preprocess, output_path="output_video.mp4", original_filename=None, skip_black=(10, 3), sample_interval=2, confidence=0.15, iou=0.7, bg_thresh=40, tile_grid_size=(32,32), preresize=None, progress_callback=None, apply_clahe=True, trim_black=True, frame_skip=2, csv_path=None, zip_path=None):
    N_BG_SAMPLES = 20
    MIN_BLOB_AREA = 500
    resized_path = None
    sample_interval = max(1, int(frame_skip if frame_skip is not None else sample_interval))

    if preresize is not None:
        temp_dir = tempfile.gettempdir()
        resized_filename = f"{Path(video_path).stem}_resized{preresize}.mp4"
        resized_path = os.path.join(temp_dir, resized_filename)
        success = resize_video(video_path, resized_path, target_height=preresize, progress_callback=progress_callback)
        if not success: raise RuntimeError("Pre-resize failed, aborting inference.")
        video_path = resized_path

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise RuntimeError(f"Could not open video: {video_path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)

    first_real_frame = 0
    if trim_black and skip_black:
        first_real_frame = skip_black_frames(cap, skip_black[0], skip_black[1])

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - first_real_frame
    if total_frames <= 0:
        cap.release()
        raise RuntimeError("No frames available for inference after preprocessing.")
    
    sample_indices = sorted(set(np.linspace(0, total_frames - 1, N_BG_SAMPLES, dtype=int) + first_real_frame))
    bg_frames = []
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret: bg_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32))

    if not bg_frames:
        cap.release()
        raise RuntimeError("No background frames could be read.")

    median_bg = np.median(np.stack(bg_frames), axis=0).astype(np.uint8)
    median_bg = cv2.GaussianBlur(median_bg, (5, 5), 0)

    output_fps = max((orig_fps if orig_fps and orig_fps > 0 else 30.0) / sample_interval, 1)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height))

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=tile_grid_size)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    cap.set(cv2.CAP_PROP_POS_FRAMES, first_real_frame)
    frame_idx = 0
    frames_inferred = 0
    frames_skipped_by_gate = 0

    # --- NEW: Setup CSV and ZIP Infrastructure ---
    max_animals_counted = 0
    csv_file = None
    csv_writer = None
    if csv_path:
        csv_file = open(csv_path, mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["media_id", "frame", "x", "y", "width", "height"])

    seen_track_ids = set()
    temp_crop_dir = None
    if zip_path: temp_crop_dir = tempfile.mkdtemp() 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        if progress_callback and total_frames > 0:
            progress_callback(min(frame_idx / total_frames, 1.0), f"Running YOLO Inference... ({frame_idx}/{total_frames} frames)")

        if frame_idx % sample_interval == 0:
            processed_bgr = preprocess(frame, clahe) if apply_clahe else frame
            infer_frame = processed_bgr

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray, median_bg)
            _, fg_mask = cv2.threshold(diff, bg_thresh, 255, cv2.THRESH_BINARY)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, morph_kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, morph_kernel)

            _, _, stats, _ = cv2.connectedComponentsWithStats(fg_mask)
            has_activity = any(s[cv2.CC_STAT_AREA] >= MIN_BLOB_AREA for s in stats[1:])

            if has_activity:
                frames_inferred += 1
                results = model.track(infer_frame, persist=True, tracker=tracker_config, verbose=False, conf=confidence, iou=iou)
                
                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    class_ids = results[0].boxes.cls.int().cpu().tolist()
                    
                    if len(track_ids) > max_animals_counted:
                        max_animals_counted = len(track_ids)
                        
                    for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
                        x1, y1, x2, y2 = map(int, box)
                        class_name = model.names[cls_id] 
                        
                        if csv_writer:
                            # --- USES THE ORIGINAL FILENAME HERE ---
                            media_id = original_filename if original_filename else Path(video_path).name
                            csv_writer.writerow([media_id, frame_idx, f"{x1 / width:.4f}", f"{y1 / height:.4f}", f"{(x2 - x1) / width:.4f}", f"{(y2 - y1) / height:.4f}"])
                            
                        if zip_path and temp_crop_dir and track_id not in seen_track_ids:
                            seen_track_ids.add(track_id)
                            crop_img = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
                            if crop_img.size > 0:
                                cv2.imwrite(os.path.join(temp_crop_dir, f"id{track_id}_{class_name}_f{frame_idx}.jpg"), crop_img)

                    annotated = draw_boxes_no_labels(processed_bgr, boxes, track_ids, thickness=6)
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
    
    if csv_file: csv_file.close()
        
    if zip_path and temp_crop_dir:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(temp_crop_dir):
                for file in files: zf.write(os.path.join(root, file), file)
        shutil.rmtree(temp_crop_dir)

    if resized_path and Path(resized_path).exists(): Path(resized_path).unlink()
    return max_animals_counted

# --- STREAMLIT WRAPPER ---
def process_video_with_model(input_path, output_path, model_choice, original_filename=None, progress_callback=None, confidence=0.15, iou=0.7, apply_clahe=True, trim_black=True, frame_skip=2, csv_path=None, zip_path=None):
    active_model, active_tracker_config = load_model_and_config(model_choice)
    return perform_inference(
        video_path=input_path, model=active_model, tracker_config=active_tracker_config, preprocess=clahe_L_median, output_path=output_path, original_filename=original_filename, preresize=1080, progress_callback=progress_callback, confidence=confidence, iou=iou, apply_clahe=apply_clahe, trim_black=trim_black, frame_skip=frame_skip, csv_path=csv_path, zip_path=zip_path
    )