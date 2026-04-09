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

# --- GLOBAL CACHE SETUP ---
# Force Ultralytics to use a global weight/config directory to prevent repo bloat
YOLO_CACHE_DIR = Path.home() / ".config" / "ultralytics"
YOLO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
settings.update({'weights_dir': str(YOLO_CACHE_DIR), 'runs_dir': str(YOLO_CACHE_DIR / "runs")})

# --- MODEL & CONFIG INITIALIZATION ---
@st.cache_resource
def load_model_and_config():
    """Load model once and cache it globally"""
    model_path = YOLO_CACHE_DIR / "yolov8x-worldv2.pt"
    
    # Initialize using the full path
    model = YOLOWorld(str(model_path)) 
    
    custom_vocabulary = [
        "fish", "eel", "ray", "shark", "jellyfish", "animal", "shrimp", 
        "crab", "lobster", "isopod", "octopus", "squid", "mollusk", 
        "crustacean", "animal cluster"
    ]
    model.set_classes(custom_vocabulary)
    
    yaml_path = os.path.join(os.path.dirname(ultralytics.__file__), "cfg", "trackers", "bytetrack.yaml")
    custom_yaml_path = YOLO_CACHE_DIR / "bytetrack_custom.yaml"

    shutil.copy(yaml_path, custom_yaml_path)
    with open(custom_yaml_path, "r") as f:
        config = yaml.safe_load(f)
        
    config["track_buffer"] = 60 
    config["match_thresh"] = 0.85
    
    with open(custom_yaml_path, "w") as f:
        yaml.dump(config, f)
    
    return model, str(custom_yaml_path)

model, custom_tracker_config = load_model_and_config()

# --- PREPROCESSING FUNCTIONS ---
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
            
    print(f"Skipping {first_real_frame} black frames, starting at frame {first_real_frame}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_real_frame)
    return first_real_frame

def clahe_on_l_channel_LAB(frame, clahe):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b_ch = cv2.split(lab)
    lab = cv2.merge([clahe.apply(l), a, b_ch])
    processed_bgr  = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return processed_bgr

def clahe_L_median(frame, clahe):
    median_blur_frame = cv2.medianBlur(frame, 3)
    final_frame = clahe_on_l_channel_LAB(median_blur_frame, clahe)
    return final_frame

# --- VIDEO RESIZING CAPABILITIES ---
def detect_encoder() -> tuple[str, list[str]]:
    probes = [
        ('h264_nvenc', ['-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda']),
        ('h264_vaapi', ['-hwaccel', 'vaapi', '-vaapi_device', '/dev/dri/renderD128']),
    ]
    null = subprocess.DEVNULL
    for enc, hw_flags in probes:
        probe_cmd = [
            'ffmpeg', '-y', '-f', 'lavfi', '-i', 'testsrc=duration=1:size=3840x2160',
            *hw_flags, '-vframes', '1', '-c:v', enc, '-f', 'null', '-',
        ]
        if subprocess.run(probe_cmd, stdout=null, stderr=null).returncode == 0:
            return enc, hw_flags
    return 'libx264', []

def _build_scale_filter(encoder: str, target_height: int) -> str:
    target_width = -2  
    if 'vaapi' in encoder:
        return f'scale_vaapi=w={target_width}:h={target_height}:force_original_aspect_ratio=decrease'
    elif 'nvenc' in encoder:
        return f'scale_cuda={target_width}:{target_height}:force_original_aspect_ratio=decrease,hwdownload,format=nv12'
    else:
        return f'scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,format=yuv420p'

def _get_frame_count(path: Path) -> int:
    r = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-count_packets', '-show_entries', 'stream=nb_read_packets',
         '-of', 'csv=p=0', str(path)],
        capture_output=True, text=True,
    )
    val = r.stdout.strip()
    return int(val) if val.isdigit() else 0

def resize_video(input_path, output_path, crf=23, encoder=None, hw_flags=None, force_cpu=False, target_height=1280, progress_callback=None) -> bool:
    input_path  = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if encoder is None:
        if force_cpu:
            encoder, hw_flags = 'libx264', []
        else:
            encoder, hw_flags = detect_encoder()

    hw_flags = hw_flags or []
    vf = _build_scale_filter(encoder, target_height)
    extra = []
    if encoder == 'libx264':      extra = ['-crf', str(crf), '-preset', 'fast']
    elif encoder == 'h264_nvenc': extra = ['-cq',  str(crf), '-preset', 'p4']
    elif encoder == 'h264_vaapi': extra = ['-qp',  str(crf)]

    total_frames = _get_frame_count(input_path)

    cmd = [
        'ffmpeg', '-y', *hw_flags,
        '-i', str(input_path),
        '-vf', vf, '-c:v', encoder, *extra,
        '-an', '-movflags', '+faststart',
        '-progress', 'pipe:1', '-nostats',
        str(output_path),
    ]

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
                        percent = min(current / total_frames, 1.0)
                        progress_callback(percent, f"Compressing video... ({current}/{total_frames} frames)")
        process.wait()

    if process.returncode != 0:
        print(f'  FFmpeg error ({input_path.name}):\n' + ''.join(stderr_buf[-20:]), file=sys.stderr)
        return False
    return True

# --- MAIN INFERENCE PIPELINE ---
def perform_inference(video_path, preprocess, output_path="output_video.mp4", skip_black=(10, 3), sample_interval=2, confidence=0.15, iou=0.7, bg_thresh=40, tile_grid_size=(32,32), preresize=None, progress_callback=None, apply_clahe=True, trim_black=True, frame_skip=2, csv_path=None, zip_path=None):
    N_BG_SAMPLES = 20
    MIN_BLOB_AREA = 500
    resized_path = None
    sample_interval = max(1, int(frame_skip if frame_skip is not None else sample_interval))

    if preresize is not None:
        temp_dir = tempfile.gettempdir()
        resized_filename = f"{Path(video_path).stem}_resized{preresize}.mp4"
        resized_path = os.path.join(temp_dir, resized_filename)
        
        print(f"Pre-resizing to {preresize}p...")
        success = resize_video(video_path, resized_path, target_height=preresize, progress_callback=progress_callback)
        if not success:
            raise RuntimeError("Pre-resize failed, aborting inference.")
        video_path = resized_path

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)

    first_real_frame = 0
    if trim_black and skip_black:
        black_thresh, require_consecutive = skip_black
        first_real_frame = skip_black_frames(cap, black_thresh, require_consecutive)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - first_real_frame
    if total_frames <= 0:
        cap.release()
        raise RuntimeError("No frames available for inference after preprocessing.")
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

    output_fps = max((orig_fps if orig_fps and orig_fps > 0 else 30.0) / sample_interval, 1)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height))

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=tile_grid_size)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    cap.set(cv2.CAP_PROP_POS_FRAMES, first_real_frame)
    frame_idx = 0
    frames_inferred = 0
    frames_skipped_by_gate = 0
    has_activity = False

    # --- NEW: Setup CSV and ZIP Infrastructure ---
    max_animals_counted = 0
    csv_file = None
    csv_writer = None
    if csv_path:
        csv_file = open(csv_path, mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        # Write the CSV header
        csv_writer.writerow(["video", "frame", "tracker_id", "class", "confidence", "x1", "y1", "x2", "y2"])

    seen_track_ids = set()
    temp_crop_dir = None
    if zip_path:
        temp_crop_dir = tempfile.mkdtemp() # Create a temporary folder for images

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if progress_callback and total_frames > 0:
            percent = min(frame_idx / total_frames, 1.0)
            progress_callback(percent, f"Running YOLO Inference... ({frame_idx}/{total_frames} frames)")

        if frame_idx % sample_interval == 0:
            processed_bgr = preprocess(frame, clahe) if apply_clahe else frame
            infer_frame = processed_bgr

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray, median_bg)
            _, fg_mask = cv2.threshold(diff, bg_thresh, 255, cv2.THRESH_BINARY)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,   morph_kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, morph_kernel)

            _, _, stats, _ = cv2.connectedComponentsWithStats(fg_mask)
            has_activity = any(s[cv2.CC_STAT_AREA] >= MIN_BLOB_AREA for s in stats[1:])

            if has_activity:
                frames_inferred += 1
                results = model.track(
                    infer_frame,
                    persist=True,
                    tracker=custom_tracker_config, 
                    verbose=False,
                    conf=confidence,
                    iou=iou,
                )
                
                # --- NEW: Data Extraction Logic ---
                if results[0].boxes.id is not None:
                    # Get all tracking data for this frame
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    class_ids = results[0].boxes.cls.int().cpu().tolist()
                    confs = results[0].boxes.conf.cpu().numpy()
                    
                    # 1. Update MaxN
                    current_animal_count = len(track_ids)
                    if current_animal_count > max_animals_counted:
                        max_animals_counted = current_animal_count
                        
                    # 2. Write CSV Data and Crop Images
                    for box, track_id, cls_id, conf in zip(boxes, track_ids, class_ids, confs):
                        x1, y1, x2, y2 = map(int, box)
                        class_name = model.names[cls_id] # Look up the string name (e.g., 'fish')
                        
                        # Write to CSV
                        if csv_writer:
                            csv_writer.writerow(["current_video", frame_idx, track_id, class_name, f"{conf:.4f}", x1, y1, x2, y2])
                            
                        # Save Cropped Image (Only the first time we see this ID)
                        if zip_path and temp_crop_dir and track_id not in seen_track_ids:
                            seen_track_ids.add(track_id)
                            
                            # Ensure crop stays within image bounds
                            y1_safe, y2_safe = max(0, y1), min(height, y2)
                            x1_safe, x2_safe = max(0, x1), min(width, x2)
                            crop_img = frame[y1_safe:y2_safe, x1_safe:x2_safe]
                            
                            if crop_img.size > 0:
                                crop_filename = os.path.join(temp_crop_dir, f"id{track_id}_{class_name}_f{frame_idx}.jpg")
                                cv2.imwrite(crop_filename, crop_img)

                    # Draw the boxes for the output video
                    annotated = results[0].plot()
                    frame_out = cv2.resize(annotated, (width, height))
                else:
                    frame_out = processed_bgr
            else:
                frames_skipped_by_gate += 1
                frame_out = processed_bgr

            out.write(frame_out)

        print(f"Processed frame {frame_idx} | activity={'YES' if has_activity else 'no '}", end="\r")
        frame_idx += 1

    cap.release()
    out.release()
    
    # --- NEW: Cleanup and Packaging ---
    if csv_file:
        csv_file.close()
        
    if zip_path and temp_crop_dir:
        # Zip all the cropped images
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(temp_crop_dir):
                for file in files:
                    zf.write(os.path.join(root, file), file)
        # Delete the temporary folder
        shutil.rmtree(temp_crop_dir)

    if resized_path and Path(resized_path).exists():
        print(f"\nCleaning up temporary file: {resized_path}")
        Path(resized_path).unlink()

    total_sampled = frames_inferred + frames_skipped_by_gate
    skip_rate = frames_skipped_by_gate / max(total_sampled, 1)
    print(f"\nDone. Video saved as {output_path}.")
    print(f"Frames inferred:         {frames_inferred}")
    print(f"Frames skipped by gate:  {frames_skipped_by_gate}")
    print(f"Gate skip rate:          {skip_rate:.1%}")
    print(f"Max Animals (MaxN):      {max_animals_counted}")

    return max_animals_counted

def process_video_with_model(input_path, output_path, progress_callback=None, confidence=0.15, iou=0.7, apply_clahe=True, trim_black=True, frame_skip=2, csv_path=None, zip_path=None):
    """Main wrapper function called by the Streamlit frontend"""
    
    # Pass all the caught variables directly into perform_inference
    max_animals = perform_inference(
        video_path=input_path, 
        preprocess=clahe_L_median, 
        output_path=output_path, 
        preresize=1080, 
        progress_callback=progress_callback,
        confidence=confidence,           # Passed from UI
        iou=iou,                         # Passed from UI
        apply_clahe=apply_clahe,         # Passed from UI
        trim_black=trim_black,           # Passed from UI
        frame_skip=frame_skip,           # Passed from UI
        csv_path=csv_path,               # Target for your Luis's CSV generator
        zip_path=zip_path                # Target for your Luis's Zip generator
    )
    
    return max_animals