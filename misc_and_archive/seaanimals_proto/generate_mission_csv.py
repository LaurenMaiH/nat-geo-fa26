import dropbox
import cv2
from ultralytics import YOLOWorld, settings
import os
import shutil
import yaml
from pathlib import Path
import gdown
import ultralytics
import subprocess
import threading
from tqdm import tqdm
import numpy as np
import csv
import zipfile
import tempfile
import torch
from dotenv import load_dotenv

from model_pipeline import process_video_with_model

load_dotenv()

# none processes all vids from a folder
SELECTED_VIDEOS = ["PLW_dscm_05_c058.mp4", "PLW_dscm_05_c117.mp4","PLW_dscm_05_c120.mp4", "PLW_dscm_05_c137.mp4", "PLW_dscm_05_c183.mp4", "PLW_dscm_05_c186.mp4"]
# SELECTED_VIDEOS = None


APP_KEY        = os.environ["DROPBOX_APP_KEY"]
APP_SECRET     = os.environ["DROPBOX_APP_SECRET"]
REFRESH_TOKEN  = os.environ["DROPBOX_REFRESH_TOKEN"]
MISSION_PATH = "/DOEX0096_Palau/PLW_dscm_05/Video"
OUTPUT_CSV_PATH = Path.cwd() / "test.csv"


YOLO_CACHE_DIR = Path.home() / ".config" / "ultralytics"
YOLO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
settings.update({'weights_dir': str(YOLO_CACHE_DIR), 'runs_dir': str(YOLO_CACHE_DIR / "runs")})
INFERENCE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using inference device: {INFERENCE_DEVICE}")



def connect_to_dropbox():
    dbx = dropbox.Dropbox(
        app_key=APP_KEY,
        app_secret=APP_SECRET,
        oauth2_refresh_token=REFRESH_TOKEN,
    )

    try:
        dbx.files_list_folder('')
        print('Connection Successful')
    except Exception as e:
        print(f'Error: {e}')

    for entry in dbx.files_list_folder(MISSION_PATH).entries:
        print(entry.name)

    return dbx
dbx = connect_to_dropbox()



def load_model_and_config():
    model_path = YOLO_CACHE_DIR / "yolov8x-worldv2.pt"
    if not model_path.exists():
        print(f"Default model not found in cache. Downloading to {model_path}...")
        url = 'https://drive.google.com/file/d/1hh576zOzpUqgWSjIdSkR4EwgYQ8kH406/view?usp=sharing'
        gdown.download(url, str(model_path), quiet=False)
    print(f"Loading Default Model from: {model_path}")
    model = YOLOWorld(str(model_path))
    model.model.eval()
    custom_vocabulary = [
            "fish", "eel", "ray", "shark", "jellyfish", "animal", "shrimp", 
            "crab", "lobster", "isopod", "octopus", "squid", "mollusk", 
            "crustacean", "animal cluster"
    ]
    model.set_classes(custom_vocabulary)


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
connect_to_dropbox()


def perform_inference(
    video_path, 
    model, 
    tracker_config, 
    preprocess, 
    original_filename=None, 
    skip_black=(10, 3), 
    confidence=0.15, 
    iou=0.7, 
    bg_thresh=40, 
    tile_grid_size=(32,32), 
    preresize=None, 
    progress_callback=None, 
    apply_clahe=True, 
    trim_black=True, 
    csv_path=None, 
    zip_path=None):

    N_BG_SAMPLES = 20
    MIN_BLOB_AREA = 500
    resized_path = None
    sample_interval = 1

    if preresize is not None:
        temp_dir = tempfile.gettempdir()
        resized_filename = f"{Path(video_path).stem}_resized{preresize}.mp4"
        resized_path = os.path.join(temp_dir, resized_filename)
        success = resize_video(video_path, resized_path, target_height=preresize, progress_callback=progress_callback)
        if not success: raise RuntimeError("Pre-resize failed, aborting inference.")
        video_path = resized_path
    
    cap = cv2.VideoCapture(str(video_path))
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

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=tile_grid_size)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    cap.set(cv2.CAP_PROP_POS_FRAMES, first_real_frame)
    frame_idx = 0
    frames_inferred = 0
    frames_skipped_by_gate = 0
    
    csv_file = None
    csv_writer = None
    if csv_path:
        csv_path = Path(csv_path)
        should_write_header = not csv_path.exists() or csv_path.stat().st_size == 0
        csv_file = open(csv_path, mode='a', newline='')
        csv_writer = csv.writer(csv_file)
        if should_write_header:
            csv_writer.writerow(["media_id", "frame", "track_id", "x", "y", "width", "height", "confidence"])

    progress_desc = f"YOLO {original_filename or Path(video_path).name}"
    with tqdm(total=total_frames, desc=progress_desc, unit="frame", leave=True) as pbar:
        while frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

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
                    results = model.track(infer_frame, persist=True, tracker=tracker_config, verbose=False, conf=confidence, iou=iou, device=INFERENCE_DEVICE)
                    
                    if results[0].boxes.id is not None:
                        boxes = results[0].boxes.xyxy.cpu().numpy()
                        track_ids = results[0].boxes.id.int().cpu().tolist()
                        class_ids = results[0].boxes.cls.int().cpu().tolist()

                        confs = results[0].boxes.conf.cpu().tolist()

                        for box, track_id, cls_id, conf in zip(boxes, track_ids, class_ids, confs):
                            x1, y1, x2, y2 = map(int, box)
                            if csv_writer:
                                csv_writer.writerow([original_filename, frame_idx, track_id, f"{x1 / width:.4f}", f"{y1 / height:.4f}", f"{(x2 - x1) / width:.4f}", f"{(y2 - y1) / height:.4f}", f"{conf:.4f}"])
                                csv_file.flush()
                else:
                    frames_skipped_by_gate += 1

            frame_idx += 1
            pbar.update(1)
            pbar.set_postfix(yolo_frames=frames_inferred, skipped=frames_skipped_by_gate)

    cap.release()
    
    if csv_file:
        csv_file.close()

def clean_csv(csv_path: Path):
    """Remove sentinel -1 rows written for videos with no detections."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return
    with open(csv_path, newline='') as f:
        rows = list(csv.reader(f))
    header, data = rows[0], rows[1:]
    cleaned = [r for r in data if r and r[1] != '-1']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(cleaned)
    removed = len(data) - len(cleaned)
    print(f"Removed {removed} sentinel row(s) from {csv_path}")


def get_last_processed(csv_path: Path) -> str | None:
    """Return the media_id of the last row written to the CSV, or None."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return None
    with open(csv_path, newline='') as f:
        rows = [r for r in csv.reader(f) if r]
    if len(rows) < 2:
        return None
    return rows[-1][0] 


def main():
    model, tracker_config = load_model_and_config()
    preprocess = clahe_L_median
    print(f"Appending detections to {OUTPUT_CSV_PATH}")

    last_processed = get_last_processed(OUTPUT_CSV_PATH)
    if last_processed:
        print(f"Resuming after: {last_processed}")
    else:
        print("Starting from the beginning")

    all_entries = sorted(
        [e for e in dbx.files_list_folder(MISSION_PATH).entries if e.name.endswith('.mp4')],
        key=lambda e: e.name
    )

    # Filter to selected videos if specified
    if SELECTED_VIDEOS is not None:
        selected_set = set(SELECTED_VIDEOS)
        entries = [e for e in all_entries if e.name in selected_set]
        missing = selected_set - {e.name for e in entries}
        if missing:
            print(f"Warning: these selected videos were not found in Dropbox: {missing}")
        print(f"Selected {len(entries)} video(s) from {len(all_entries)} in mission")
    else:
        entries = all_entries
        print(f"Processing all {len(entries)} video(s) in mission")

    # Resume: skip up to and including last_processed within the filtered list
    if last_processed and last_processed in {e.name for e in entries}:
        idx = next(i for i, e in enumerate(entries) if e.name == last_processed)
        remaining = entries[idx + 1:]
        print(f"Skipping {idx + 1} video(s), {len(remaining)} remaining\n")
    else:
        remaining = entries
        print(f"{len(remaining)} video(s) to process\n")

    counter = 0
    with tempfile.TemporaryDirectory(prefix="natgeo_videos_") as tmpdir:
        tmpdir = Path(tmpdir)
        for entry in remaining:
            dropbox_video_path = MISSION_PATH + "/" + entry.name
            local_video_path = tmpdir / entry.name

            print(f"Downloading {entry.name}")
            dbx.files_download_to_file(str(local_video_path), dropbox_video_path)

            print(f"Processing {entry.name}")
            try:
                perform_inference(
                    str(local_video_path), model, tracker_config, preprocess,
                    original_filename=entry.name,
                    skip_black=(10, 3), confidence=0.15, iou=0.7,
                    bg_thresh=40, tile_grid_size=(32, 32), preresize=1080,
                    progress_callback=None, apply_clahe=True, trim_black=True,
                    csv_path=OUTPUT_CSV_PATH, zip_path=None
                )
                # Write sentinel so videos with no detections are still recorded
                last = get_last_processed(OUTPUT_CSV_PATH)
                if last != entry.name:
                    with open(OUTPUT_CSV_PATH, 'a', newline='') as f:
                        csv.writer(f).writerow([entry.name, -1, None, None, None, None, None, None])
                counter += 1
                print(f"✓ {entry.name} done")
            except Exception as e:
                print(f"✗ {entry.name} failed: {e}")
            finally:
                local_video_path.unlink(missing_ok=True)

    print(f"\nProcessed {counter} videos this session")


main()
clean_csv(OUTPUT_CSV_PATH)