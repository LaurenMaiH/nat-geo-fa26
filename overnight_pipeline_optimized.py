#!/usr/bin/env python3
"""
Overnight Detection + Extraction + Embedding  [OPTIMIZED]

Key changes vs original:
  1. Download prefetching  — next video downloads in background while current is processing
  2. Single-pass crops     — frames are cached during YOLO inference; no second video seek pass
  3. Batched embeddings    — all crops for a video are embedded in one GPU call
  4. Minor: ffprobe removed from resize hot-path (uses cv2 instead)

Everything else (CSV format, crop folder layout, JSON schema, resume logic,
motion gating, CLAHE, ByteTrack config, MIN_TRACK_LEN filter) is unchanged.
"""

import os
import csv
import json
import queue
import shutil
import tempfile
import subprocess
import threading
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
import yaml
import gdown
import dropbox
import ultralytics
from ultralytics import YOLOWorld, settings
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────────────
APP_KEY        = os.environ["DROPBOX_APP_KEY"]
APP_SECRET     = os.environ["DROPBOX_APP_SECRET"]
REFRESH_TOKEN  = os.environ["DROPBOX_REFRESH_TOKEN"]
MISSION_PATH = "/DOEX0096_Palau/PLW_dscm_33/Video"

SELECTED_VIDEOS = None
# Start processing from this video onward (inclusive), ignoring everything before it.
# Useful when you want to skip ahead without touching the CSV resume logic.
# Set to None to disable (default behaviour: start from beginning or resume from CSV).
# Example: START_FROM_VIDEO = "PLW_dscm_11_c006.mp4"
START_FROM_VIDEO = "PLW_dscm_33_c013.mp4"

CHECKPOINT_PATH = Path(__file__).parent / "finetuning" / "checkpoints" / "best.pt"  # set to None to use pretrained weights

_MISSION_DIR      = Path(__file__).parent / "processed_missions" / "mission_33"
OUTPUT_CSV_PATH   = _MISSION_DIR / "overnight_detections.csv"
OUTPUT_CROPS_DIR  = _MISSION_DIR / "overnight_crops"
OUTPUT_EMBEDDINGS = _MISSION_DIR / "overnight_embeddings.json"

CONF_THRESHOLD  = 0.15
IOU_THRESHOLD   = 0.70
MIN_TRACK_LEN   = 3
APPLY_CLAHE     = True
TRIM_BLACK      = True
PRERESIZE       = 1080
CROPS_PER_TRACK = 3
CROP_PAD        = 10

CUSTOM_VOCABULARY = [
    "fish", "eel", "ray", "shark", "jellyfish", "animal", "shrimp",
    "crab", "lobster", "isopod", "octopus", "squid", "mollusk",
    "crustacean", "animal cluster",
]

YOLO_CACHE_DIR   = Path.home() / ".config" / "ultralytics"
INFERENCE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CROP_LABELS      = ["early", "mid", "late"]

# How many frames to keep in the rolling cache for crop extraction.
# Increase if your highest-confidence detections tend to cluster near the end
# of long tracks; decrease to reduce peak RAM usage.
FRAME_CACHE_MAX = 500

# ── MODEL SETUP ───────────────────────────────────────────────────────────────
YOLO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
settings.update({"weights_dir": str(YOLO_CACHE_DIR), "runs_dir": str(YOLO_CACHE_DIR / "runs")})

_model_cache: dict = {}


def load_model_and_config():
    if "model" in _model_cache:
        return _model_cache["model"], _model_cache["tracker"]

    model_path = YOLO_CACHE_DIR / "yolov8x-worldv2.pt"
    if not model_path.exists():
        print(f"Downloading YOLO-World model to {model_path}...")
        gdown.download(
            "https://drive.google.com/file/d/1hh576zOzpUqgWSjIdSkR4EwgYQ8kH406/view?usp=sharing",
            str(model_path), quiet=False,
        )

    model = YOLOWorld(str(model_path))
    model.model.eval()
    model.set_classes(CUSTOM_VOCABULARY)

    default_yaml = Path(os.path.dirname(ultralytics.__file__)) / "cfg" / "trackers" / "bytetrack.yaml"
    custom_yaml  = Path(__file__).parent / "bytetrack_custom.yaml"
    if not custom_yaml.exists():
        shutil.copy(default_yaml, custom_yaml)
        with open(custom_yaml) as f:
            cfg = yaml.safe_load(f)
        cfg.update({"track_buffer": 60, "match_thresh": 0.85})
        with open(custom_yaml, "w") as f:
            yaml.dump(cfg, f)

    _model_cache["model"]   = model
    _model_cache["tracker"] = str(custom_yaml)
    print(f"Model loaded. Device: {INFERENCE_DEVICE}")
    return model, str(custom_yaml)


# ──────── PREPROCESSING ──────────────────────────────────────────────────────
def skip_black_frames(cap, threshold=10, require_consecutive=3) -> int:
    consecutive = 0
    first_real  = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean() > threshold:
            consecutive += 1
        else:
            consecutive = 0
        if consecutive >= require_consecutive:
            first_real = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - require_consecutive
            break
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_real)
    return first_real


def clahe_L_median(frame: np.ndarray, clahe) -> np.ndarray:
    blurred = cv2.medianBlur(frame, 3)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)


# ──────── VIDEO RESIZE ───────────────────────────────────────────────────────
def resize_video(input_path: Path, output_path: Path, target_height: int = 1080) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use cv2 for frame count — avoids spawning an ffprobe subprocess
    _cap = cv2.VideoCapture(str(input_path))
    total = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    _cap.release()

    vf  = f"scale=-2:{target_height}:force_original_aspect_ratio=decrease,format=yuv420p"
    cmd = ["ffmpeg", "-y", "-i", str(input_path), "-vf", vf,
           "-c:v", "libx264", "-crf", "23", "-preset", "fast",
           "-an", "-movflags", "+faststart", str(output_path)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    threading.Thread(target=lambda: proc.stderr.read(), daemon=True).start()
    with tqdm(total=total, desc=f"  Resizing {input_path.name}", unit="frame", leave=False) as pbar:
        last = 0
        for line in proc.stdout:
            if line.startswith("frame="):
                val = line.split("=")[1].strip()
                if val.isdigit():
                    cur = int(val)
                    pbar.update(cur - last)
                    last = cur
        proc.wait()
    return proc.returncode == 0


# ──────── EMBEDDING MODEL ────────────────────────────────────────────────────
def load_embedding_model():
    import open_clip

    if not CHECKPOINT_PATH:
        raise RuntimeError(
            "CHECKPOINT_PATH is None — finetuned weights are required. "
            "Set CHECKPOINT_PATH to your best.pt file."
        )

    checkpoint = Path(CHECKPOINT_PATH)
    if not checkpoint.exists():
        print(f"Finetuned weights not found at {checkpoint}, downloading...")
        gdown.download(
            "https://drive.google.com/file/d/1Qec1Gzxl8zAazuLof8a2LUSrbOR7GmtT/view?usp=drive_link",
            str(checkpoint), quiet=False, fuzzy=True,
        )

    if not checkpoint.exists():
        raise RuntimeError(
            f"Finetuned weights still not found at {checkpoint} after download attempt. "
            "Check your CHECKPOINT_PATH and that the download succeeded."
        )

    model, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:imageomics/bioclip-2", device=INFERENCE_DEVICE
    )

    print(f"Loading finetuned weights from {checkpoint}...")
    ckpt = torch.load(checkpoint, map_location=INFERENCE_DEVICE)
    model.visual.load_state_dict(ckpt["model_state"])
    print("Finetuned weights loaded ✓")

    model.eval()
    return ("bioclip", model, preprocess)


# CHANGED: accepts a list of BGR images; returns array shape (N, D)
def get_embeddings_batch(embed_info, images_bgr: list) -> np.ndarray:
    """Embed a batch of BGR images in a single GPU forward pass."""
    kind, model, preprocess = embed_info
    if kind == "bioclip":
        from PIL import Image
        tensors = torch.stack([
            preprocess(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
            for img in images_bgr
        ]).to(INFERENCE_DEVICE)
        with torch.no_grad():
            vecs = model.encode_image(tensors).cpu().numpy()
        vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
        return vecs
    else:
        vecs = np.random.rand(len(images_bgr), 512).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs


# ──────── CSV HELPERS ────────────────────────────────────────────────────────
def get_last_processed(csv_path: Path):
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return None
    with open(csv_path, newline="") as f:
        rows = [r for r in csv.reader(f) if r]
    return rows[-1][0] if len(rows) >= 2 else None


def clean_csv(csv_path: Path):
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))
    header, data = rows[0], rows[1:]
    cleaned = [r for r in data if r and r[1] != "-1"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(cleaned)
    print(f"Removed {len(data) - len(cleaned)} sentinel row(s) from {csv_path}")


# ──────── PER-VIDEO PIPELINE ─────────────────────────────────────────────────
def process_video(
    video_path: Path,
    media_id: str,
    video_name: str,
    model,
    tracker_config: str,
    csv_path: Path,
    crops_dir: Path,
) -> dict:
    """
    Run YOLO+ByteTrack on one video.

    CHANGED vs original:
    - Single video pass: frames needed for crops are cached in memory during
      inference (bounded by FRAME_CACHE_MAX). Eliminates the second
      VideoCapture seek pass entirely.
    - Everything else (CSV schema, motion gate, CLAHE, track filtering,
      best-per-segment crop selection) is identical.

    Returns {track_id: [crop_path, ...]} for the embedding step.
    """
    resized_path = None
    work_path    = video_path

    if PRERESIZE:
        tmp = Path(tempfile.gettempdir()) / f"{video_path.stem}_rs{PRERESIZE}.mp4"
        if resize_video(video_path, tmp, PRERESIZE):
            work_path    = tmp
            resized_path = tmp
        else:
            print("  Resize failed, using original resolution.")

    cap = cv2.VideoCapture(str(work_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {work_path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    first_real = 0
    if TRIM_BLACK:
        first_real = skip_black_frames(cap, threshold=10, require_consecutive=3)
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - first_real
    if total_frames <= 0:
        cap.release()
        raise RuntimeError("No usable frames after black-frame trim.")

    # Background model
    N_BG = 20
    sample_indices = sorted(set(
        np.linspace(0, total_frames - 1, N_BG, dtype=int) + first_real
    ))
    bg_frames = []
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, fr = cap.read()
        if ret:
            bg_frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32))
    if not bg_frames:
        cap.release()
        raise RuntimeError("Cannot build background model.")

    median_bg    = cv2.GaussianBlur(np.median(np.stack(bg_frames), axis=0).astype(np.uint8), (5, 5), 0)
    clahe        = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(32, 32))
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    BG_THRESH    = 40
    MIN_BLOB     = 500

    # CSV
    should_write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    csv_file   = open(csv_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if should_write_header:
        csv_writer.writerow(["media_id", "frame", "track_id", "class",
                             "x", "y", "width", "height", "confidence"])

    # Per-track detection accumulator
    track_detections: dict = defaultdict(list)

    # ── SINGLE-PASS FRAME CACHE ───────────────────────────────────────────────
    # Maps frame_idx -> raw BGR frame.  We only store frames that have been
    # seen by YOLO (i.e. frames with activity), and we evict the oldest ones
    # once the cache exceeds FRAME_CACHE_MAX to bound memory.
    frame_cache: dict[int, np.ndarray] = {}
    cache_order: list[int] = []   # insertion order for eviction

    def _cache_frame(fi: int, frm: np.ndarray):
        if fi in frame_cache:
            return
        if len(cache_order) >= FRAME_CACHE_MAX:
            evict = cache_order.pop(0)
            frame_cache.pop(evict, None)
        frame_cache[fi] = frm.copy()
        cache_order.append(fi)

    cap.set(cv2.CAP_PROP_POS_FRAMES, first_real)
    frame_idx       = 0
    frames_inferred = 0

    with tqdm(total=total_frames, desc=f"  YOLO {video_name}", unit="frame", leave=True) as pbar:
        while frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray, median_bg)
            _, fg = cv2.threshold(diff, BG_THRESH, 255, cv2.THRESH_BINARY)
            fg    = cv2.morphologyEx(cv2.morphologyEx(fg, cv2.MORPH_OPEN, morph_kernel),
                                     cv2.MORPH_DILATE, morph_kernel)
            _, _, stats, _ = cv2.connectedComponentsWithStats(fg)
            has_activity   = any(s[cv2.CC_STAT_AREA] >= MIN_BLOB for s in stats[1:])

            if has_activity:
                infer   = clahe_L_median(frame, clahe) if APPLY_CLAHE else frame
                results = model.track(
                    infer, persist=True, tracker=tracker_config,
                    verbose=False, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
                    device=INFERENCE_DEVICE,
                )
                frames_inferred += 1

                if results[0].boxes.id is not None:
                    boxes     = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    class_ids = results[0].boxes.cls.int().cpu().tolist()
                    confs     = results[0].boxes.conf.cpu().tolist()

                    for box, tid, cid, conf in zip(boxes, track_ids, class_ids, confs):
                        x1, y1, x2, y2 = map(int, box)
                        cls_name        = model.names[cid]
                        csv_writer.writerow([
                            media_id, frame_idx, tid, cls_name,
                            f"{x1/width:.4f}", f"{y1/height:.4f}",
                            f"{(x2-x1)/width:.4f}", f"{(y2-y1)/height:.4f}",
                            f"{conf:.4f}",
                        ])
                        track_detections[tid].append((frame_idx, conf, x1, y1, x2, y2, cls_name))

                # Cache the *original* (un-CLAHEd) frame for clean crops
                _cache_frame(frame_idx, frame)

            frame_idx += 1
            pbar.update(1)
            pbar.set_postfix(inferred=frames_inferred)

    cap.release()
    csv_file.flush()
    csv_file.close()

    # ── Filter short tracks ───────────────────────────────────────────────────
    long_tracks = {tid: dets for tid, dets in track_detections.items()
                   if len(dets) >= MIN_TRACK_LEN}
    print(f"  Tracks: {len(track_detections)} total, "
          f"{len(long_tracks)} kept (>= {MIN_TRACK_LEN} detections)")

    video_crops_dir = crops_dir / video_name
    video_crops_dir.mkdir(parents=True, exist_ok=True)

    track_crop_paths: dict = {}

    if not long_tracks:
        if resized_path and resized_path.exists():
            resized_path.unlink()
        return track_crop_paths

    # ── CHANGED: crop from cache instead of a second video pass ──────────────
    # Identify which frame each (track, segment) wants, then pull from cache.
    # Frames that were evicted (rare for short videos; possible for very long
    # ones with many active frames) fall back to a targeted cap.read().
    k = CROPS_PER_TRACK
    needed: dict[int, list] = defaultdict(list)  # frame_idx -> [(tid,...)]
    for tid, dets in long_tracks.items():
        dets_sorted = sorted(dets, key=lambda d: d[0])
        n           = len(dets_sorted)
        for seg_i, label in enumerate(CROP_LABELS[:k]):
            segment = dets_sorted[seg_i * n // k : (seg_i + 1) * n // k]
            if not segment:
                continue
            best = max(segment, key=lambda d: d[1])
            fi, conf, x1, y1, x2, y2, cls_name = best
            needed[fi].append((tid, x1, y1, x2, y2, cls_name, label))

    # Collect frames not in cache so we can do a single fallback pass if needed
    missing_frames = sorted(fi for fi in needed if fi not in frame_cache)
    fallback_frames: dict[int, np.ndarray] = {}
    if missing_frames:
        print(f"  {len(missing_frames)} crop frames evicted from cache — doing targeted seeks")
        cap_fb = cv2.VideoCapture(str(work_path))
        for fi in tqdm(missing_frames, desc=f"  Fallback seek {video_name}", unit="frame", leave=False):
            cap_fb.set(cv2.CAP_PROP_POS_FRAMES, first_real + fi)
            ret, frm = cap_fb.read()
            if ret:
                fallback_frames[fi] = frm
        cap_fb.release()

    for fi, crops_info in tqdm(needed.items(), desc=f"  Cropping {video_name}", unit="frame", leave=False):
        frame = frame_cache.get(fi)
        if frame is None:
            frame = fallback_frames.get(fi)
        if frame is None:
            continue
        for tid, x1, y1, x2, y2, cls_name, label in crops_info:
            x1p = max(0, x1 - CROP_PAD)
            y1p = max(0, y1 - CROP_PAD)
            x2p = min(width,  x2 + CROP_PAD)
            y2p = min(height, y2 + CROP_PAD)
            crop = frame[y1p:y2p, x1p:x2p]
            if crop.size == 0:
                continue
            cls_slug = cls_name.replace(" ", "_")
            fname    = f"id{tid:04d}_{cls_slug}_{label}.jpg"
            fpath    = video_crops_dir / fname
            cv2.imwrite(str(fpath), crop)
            track_crop_paths.setdefault(tid, []).append(str(fpath))

    if resized_path and resized_path.exists():
        resized_path.unlink()

    return track_crop_paths


# ──────── DROPBOX ────────────────────────────────────────────────────────────
def connect_dropbox():
    dbx = dropbox.Dropbox(
        app_key=APP_KEY,
        app_secret=APP_SECRET,
        oauth2_refresh_token=REFRESH_TOKEN,
    )
    try:
        dbx.files_list_folder("")
        print("Dropbox connection OK")
    except Exception as e:
        raise RuntimeError(f"Dropbox connection failed: {e}")
    return dbx


# ──────── DOWNLOAD PREFETCH THREAD ──────────────────────────────────────────
# CHANGED: downloads run in a background thread so the next video arrives
# while the current one is being processed on the GPU.
def _download_worker(entries, tmpdir: Path, dbx, mission_path: str,
                     out_q: queue.Queue, stop_evt: threading.Event):
    """Feeds (entry, local_path) tuples into out_q; sends None as sentinel."""
    for entry in entries:
        if stop_evt.is_set():
            break
        local_path = tmpdir / entry.name
        try:
            print(f"\n[Prefetch] Downloading {entry.name} ...")
            dbx.files_download_to_file(str(local_path), f"{mission_path}/{entry.name}")
            out_q.put((entry, local_path))
        except Exception as e:
            print(f"[Prefetch] Download failed for {entry.name}: {e}")
            out_q.put((entry, None))  # signal failure; main loop will skip
    out_q.put(None)  # sentinel: no more videos


# ──────── MAIN ───────────────────────────────────────────────────────────────
def main():
    print(f"Using device: {INFERENCE_DEVICE}")

    dbx            = connect_dropbox()
    model, tracker = load_model_and_config()
    embed_info     = load_embedding_model()

    OUTPUT_CROPS_DIR.mkdir(parents=True, exist_ok=True)

    all_embeddings: dict = {}
    if OUTPUT_EMBEDDINGS.exists():
        with open(OUTPUT_EMBEDDINGS) as f:
            all_embeddings = json.load(f)
        print(f"Loaded {len(all_embeddings)} existing embeddings from {OUTPUT_EMBEDDINGS}")

    # Build video list
    all_entries = sorted(
        [e for e in dbx.files_list_folder(MISSION_PATH).entries if e.name.endswith(".mp4")],
        key=lambda e: e.name,
    )
    if SELECTED_VIDEOS is not None:
        sel     = set(SELECTED_VIDEOS)
        entries = [e for e in all_entries if e.name in sel]
        missing = sel - {e.name for e in entries}
        if missing:
            print(f"Warning: not found in Dropbox: {missing}")
        print(f"Selected {len(entries)}/{len(all_entries)} videos")
    else:
        entries = all_entries
        print(f"Processing all {len(entries)} videos")

    # Manual start-from override (takes priority over CSV resume)
    if START_FROM_VIDEO is not None:
        names = [e.name for e in entries]
        if START_FROM_VIDEO in names:
            idx     = names.index(START_FROM_VIDEO)
            entries = entries[idx:]
            print(f"START_FROM_VIDEO: starting at '{START_FROM_VIDEO}' ({len(entries)} videos)")
        else:
            print(f"Warning: START_FROM_VIDEO '{START_FROM_VIDEO}' not found in video list — ignoring")

    # CSV resume logic (only runs if START_FROM_VIDEO is not set)
    else:
        last_processed = get_last_processed(OUTPUT_CSV_PATH)
        if last_processed and last_processed in {e.name for e in entries}:
            idx     = next(i for i, e in enumerate(entries) if e.name == last_processed)
            entries = entries[idx + 1:]
            print(f"Resuming after '{last_processed}': {len(entries)} videos remaining")
        else:
            print("Starting from the beginning")

    counter   = 0
    stop_evt  = threading.Event()

    with tempfile.TemporaryDirectory(prefix="natgeo_") as tmpdir:
        tmpdir = Path(tmpdir)

        # CHANGED: spin up the prefetch thread and buffer up to 2 videos ahead
        download_q: queue.Queue = queue.Queue(maxsize=2)
        dl_thread = threading.Thread(
            target=_download_worker,
            args=(entries, tmpdir, dbx, MISSION_PATH, download_q, stop_evt),
            daemon=True,
        )
        dl_thread.start()

        while True:
            item = download_q.get()
            if item is None:
                break  # no more videos

            entry, local_path = item
            if local_path is None:
                print(f"✗  {entry.name} — download failed, skipping")
                continue

            video_name = Path(entry.name).stem
            media_id   = entry.name

            print(f"\n{'='*60}")
            print(f"Processing: {entry.name}")

            try:
                track_crops = process_video(
                    local_path, media_id, video_name, model, tracker,
                    OUTPUT_CSV_PATH, OUTPUT_CROPS_DIR,
                )

                # CHANGED: batch-embed all crops for this video in one GPU call
                all_crop_images = []
                all_crop_keys   = []
                for tid, crop_paths in track_crops.items():
                    images = [cv2.imread(p) for p in crop_paths]
                    images = [img for img in images if img is not None]
                    if not images:
                        continue
                    for img in images:
                        all_crop_images.append(img)
                        all_crop_keys.append(tid)

                total_crops = len(all_crop_images)
                print(f"  Embedding {total_crops} crops across {len(track_crops)} tracks (batched)...")

                if all_crop_images:
                    vecs = get_embeddings_batch(embed_info, all_crop_images)  # (N, D)
                    # Group by track_id and mean-pool
                    tid_vecs: dict[int, list] = defaultdict(list)
                    for tid, vec in zip(all_crop_keys, vecs):
                        tid_vecs[tid].append(vec)
                    for tid, tvecs in tid_vecs.items():
                        mean_v = np.mean(tvecs, axis=0)
                        mean_v /= (np.linalg.norm(mean_v) + 1e-8)
                        all_embeddings[f"{video_name}__{tid}"] = mean_v.tolist()

                with open(OUTPUT_EMBEDDINGS, "w") as f:
                    json.dump(all_embeddings, f)

                # Sentinel row so resume works
                last = get_last_processed(OUTPUT_CSV_PATH)
                if last != entry.name:
                    with open(OUTPUT_CSV_PATH, "a", newline="") as f:
                        csv.writer(f).writerow([entry.name, -1] + [None] * 7)

                counter += 1
                print(f"✓  {entry.name}  ({len(track_crops)} tracks, "
                      f"{len(all_embeddings)} total embeddings so far)")

            except Exception as e:
                print(f"✗  {entry.name} failed: {e}")

            finally:
                local_path.unlink(missing_ok=True)

        stop_evt.set()
        dl_thread.join()

    clean_csv(OUTPUT_CSV_PATH)
    print(f"\nDone! Processed {counter} videos this session.")
    print(f"  Detections CSV : {OUTPUT_CSV_PATH}")
    print(f"  Crops folder   : {OUTPUT_CROPS_DIR}/")
    print(f"  Embeddings JSON: {OUTPUT_EMBEDDINGS}  ({len(all_embeddings)} entries)")


if __name__ == "__main__":
    main()