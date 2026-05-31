#!/usr/bin/env python3
"""
Overnight Detection + Extraction + Embedding

For each video in a Dropbox mission folder:
  1. YOLO-World + ByteTrack detection (conf threshold configurable)
  2. Append ALL detections to one shared CSV
  3. Filter out short tracks (noise reduction, configurable)
  4. Extract top-3 crops per animal (best confidence from each third of detection span)
  5. Store crops in per-video subfolders under overnight_crops/
  6. Embed all crops via BioClip -> store in one JSON  {video_name__animal_ID: embedding}

Optimizations over previous version:
  - Crop extraction uses grab()-based forward seek instead of random seek
  - Resized video written to same tmpdir as download (predictable cleanup)
  - BioCLIP embeddings are batched per track instead of one image at a time
"""

import os
import csv
import json
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
from dotenv import load_dotenv

load_dotenv()

# ── CONFIG ────────────────────────────────────────────────────────────────────
APP_KEY        = os.environ["DROPBOX_APP_KEY"]
APP_SECRET     = os.environ["DROPBOX_APP_SECRET"]
REFRESH_TOKEN  = os.environ["DROPBOX_REFRESH_TOKEN"]

MISSION_ID   = "PLW_dscm_03"          # ← only line to change between missions
MISSION_PATH = f"/DOEX0096_Palau/{MISSION_ID}/Video"   # ← and this one

SELECTED_VIDEOS = None  # set to a list of filenames, or None for all
START_FROM      = "PLW_dscm_03_c011.mp4"  # set to a filename to start from that video (inclusive)

CHECKPOINT_PATH = Path(__file__).parent / "checkpoints" / "best.pt"

_MISSION_DIR      = Path(__file__).parent / "processed_missions" / MISSION_ID
_MISSION_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV_PATH   = _MISSION_DIR / f"{MISSION_ID}_detections.csv"
OUTPUT_CROPS_DIR  = _MISSION_DIR / f"{MISSION_ID}_crops"
OUTPUT_EMBEDDINGS = _MISSION_DIR / f"{MISSION_ID}_embeddings.json"

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
def _get_frame_count(path: Path) -> int:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    val = r.stdout.strip()
    return int(val) if val.isdigit() else 0


def resize_video(input_path: Path, output_path: Path, target_height: int = 1080) -> bool:
    # ── output_path is now in the same tmpdir as the download ──
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = _get_frame_count(input_path)
    vf    = f"scale=-2:{target_height}:force_original_aspect_ratio=decrease,format=yuv420p"
    cmd   = ["ffmpeg", "-y", "-i", str(input_path), "-vf", vf,
              "-c:v", "libx264", "-crf", "23", "-preset", "fast",
              "-an", "-movflags", "+faststart", str(output_path)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    threading.Thread(target=lambda: proc.stderr.read(), daemon=True).start()
    with tqdm(total=total or None, desc=f"  Resizing {input_path.name}", unit="frame", leave=False) as pbar:
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
    try:
        import open_clip

        if CHECKPOINT_PATH and not Path(CHECKPOINT_PATH).exists():
            print(f"Finetuned weights not found at {CHECKPOINT_PATH}, downloading...")
            Path(CHECKPOINT_PATH).parent.mkdir(parents=True, exist_ok=True)
            gdown.download(
                "https://drive.google.com/file/d/1Qec1Gzxl8zAazuLof8a2LUSrbOR7GmtT/view?usp=drive_link",
                str(CHECKPOINT_PATH), quiet=False, fuzzy=True,
            )

        model, _, preprocess = open_clip.create_model_and_transforms(
            "hf-hub:imageomics/bioclip-2", device=INFERENCE_DEVICE
        )

        if CHECKPOINT_PATH and Path(CHECKPOINT_PATH).exists():
            print(f"Loading finetuned weights from {CHECKPOINT_PATH}...")
            ckpt = torch.load(CHECKPOINT_PATH, map_location=INFERENCE_DEVICE)
            model.visual.load_state_dict(ckpt["model_state"])
            print("Finetuned weights loaded ✓")
        else:
            print("Using pretrained BioCLIP-2 weights")

        model.eval()
        return ("bioclip", model, preprocess)
    except Exception as e:
        print(f"BioCLIP not available ({e}) -- using placeholder embeddings.")
        return ("placeholder", None, None)


def get_embeddings_batch(embed_info, images_bgr: list[np.ndarray]) -> np.ndarray:
    """
    Embed a batch of BGR images in one forward pass.
    Returns (N, D) float32 array of L2-normalised embeddings.
    """
    kind, model, preprocess = embed_info
    if kind == "bioclip":
        from PIL import Image
        tensors = []
        for img in images_bgr:
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            tensors.append(preprocess(pil))
        inp = torch.stack(tensors).to(INFERENCE_DEVICE)
        with torch.no_grad():
            vecs = model.encode_image(inp).cpu().float().numpy()
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(norms == 0, 1, norms)
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
    tmpdir: Path,
    reporter=None,
) -> dict:
    resized_path = None
    work_path    = video_path

    # ── Resize into the same tmpdir as the download ──
    if PRERESIZE:
        tmp = tmpdir / f"{video_path.stem}_rs{PRERESIZE}.mp4"
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

    # Build median background for motion gating
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

    should_write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    csv_file   = open(csv_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if should_write_header:
        csv_writer.writerow(["media_id", "frame", "track_id", "class",
                             "x", "y", "width", "height", "confidence"])

    track_detections: dict = defaultdict(list)

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

            frame_idx += 1
            pbar.update(1)
            pbar.set_postfix(inferred=frames_inferred)
            if reporter and frame_idx % 50 == 0:
                reporter.frame_progress(frame_idx, total_frames)

    if reporter:
        reporter.frame_progress(total_frames, total_frames)
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

    # ── Select best frame per segment per track ───────────────────────────────
    # needed maps frame_idx -> [(tid, x1,y1,x2,y2,cls,label)]
    needed: dict = defaultdict(list)
    k = CROPS_PER_TRACK
    for tid, dets in long_tracks.items():
        dets_sorted = sorted(dets, key=lambda d: d[0])
        n = len(dets_sorted)
        for seg_i, label in enumerate(CROP_LABELS[:k]):
            segment = dets_sorted[seg_i * n // k : (seg_i + 1) * n // k]
            if not segment:
                continue
            best = max(segment, key=lambda d: d[1])
            fi, conf, x1, y1, x2, y2, cls_name = best
            needed[fi].append((tid, x1, y1, x2, y2, cls_name, label))

    # ── Crop extraction with grab()-based forward seek ────────────────────────
    cap2 = cv2.VideoCapture(str(work_path))
    current_frame = -1
    sorted_frames = sorted(needed.keys())
    n_crop_frames = len(sorted_frames)

    for crop_fi, fi in enumerate(tqdm(sorted_frames, desc=f"  Cropping {video_name}", unit="frame", leave=False)):
        abs_fi = first_real + fi

        # Forward-seek with grab() when close; random seek when far or going back
        if current_frame < 0 or abs_fi < current_frame or (abs_fi - current_frame) > 300:
            cap2.set(cv2.CAP_PROP_POS_FRAMES, abs_fi)
            ret, frame = cap2.read()
        else:
            while current_frame < abs_fi - 1:
                cap2.grab()
                current_frame += 1
            ret, frame = cap2.read()

        if not ret:
            continue
        current_frame = abs_fi

        for tid, x1, y1, x2, y2, cls_name, label in needed[fi]:
            x1p = max(0, x1 - CROP_PAD)
            y1p = max(0, y1 - CROP_PAD)
            x2p = min(width,  x2 + CROP_PAD)
            y2p = min(height, y2 + CROP_PAD)
            crop = frame[y1p:y2p, x1p:x2p]
            if crop.size == 0:
                continue
            cls_slug = cls_name.replace(" ", "_")
            fname    = f"id{tid:04d}_{cls_slug}_f{abs_fi:05d}_{label}.jpg"
            fpath    = video_crops_dir / fname
            cv2.imwrite(str(fpath), crop)
            track_crop_paths.setdefault(tid, []).append(str(fpath))

        if reporter and n_crop_frames > 0:
            reporter.set_video_pct(1/3 + (crop_fi + 1) / n_crop_frames * (1/3))

    cap2.release()

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


# ──────── MAIN ───────────────────────────────────────────────────────────────
def main(reporter=None):
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

    last_processed = get_last_processed(OUTPUT_CSV_PATH)
    if last_processed and last_processed in {e.name for e in entries}:
        idx     = next(i for i, e in enumerate(entries) if e.name == last_processed)
        entries = entries[idx + 1:]
        print(f"Resuming after '{last_processed}': {len(entries)} videos remaining")
    elif START_FROM and START_FROM in {e.name for e in entries}:
        idx     = next(i for i, e in enumerate(entries) if e.name == START_FROM)
        entries = entries[idx:]
        print(f"Starting from '{START_FROM}': {len(entries)} videos remaining")
    else:
        print("Starting from the beginning")

    total   = len(entries)
    counter = 0
    with tempfile.TemporaryDirectory(prefix="natgeo_") as tmpdir:
        tmpdir = Path(tmpdir)
        for idx, entry in enumerate(entries):
            if reporter and reporter.should_stop():
                print("Stop signal received — exiting after current video.")
                break

            video_name = Path(entry.name).stem
            media_id   = entry.name
            local_path = tmpdir / entry.name

            if reporter:
                reporter.begin_video(video_name, idx + 1, total)
                reporter.stage("downloading")

            print(f"\n{'='*60}")
            print(f"Downloading: {entry.name}")
            dbx.files_download_to_file(str(local_path), f"{MISSION_PATH}/{entry.name}")

            try:
                if reporter:
                    reporter.stage("detecting")
                track_crops = process_video(
                    local_path, media_id, video_name, model, tracker,
                    OUTPUT_CSV_PATH, OUTPUT_CROPS_DIR,
                    tmpdir=tmpdir,
                    reporter=reporter,
                )

                # ── Batch embedding per track ─────────────────────────────────
                total_crops = sum(len(v) for v in track_crops.values())
                print(f"  Embedding {total_crops} crops across {len(track_crops)} tracks...")
                if reporter:
                    reporter.stage("embedding")
                track_items  = list(track_crops.items())
                n_emb_tracks = len(track_items)
                for emb_i, (tid, crop_paths) in enumerate(track_items):
                    images = [cv2.imread(p) for p in crop_paths]
                    images = [img for img in images if img is not None]
                    if not images:
                        continue
                    # All crops for this track embedded in one forward pass
                    vecs   = get_embeddings_batch(embed_info, images)
                    mean_v = vecs.mean(axis=0)
                    mean_v /= (np.linalg.norm(mean_v) + 1e-8)
                    all_embeddings[f"{video_name}__{tid}"] = mean_v.tolist()
                    if reporter and n_emb_tracks > 0:
                        reporter.set_video_pct(2/3 + (emb_i + 1) / n_emb_tracks * (1/3))

                with open(OUTPUT_EMBEDDINGS, "w") as f:
                    json.dump(all_embeddings, f)

                last = get_last_processed(OUTPUT_CSV_PATH)
                if last != entry.name:
                    with open(OUTPUT_CSV_PATH, "a", newline="") as f:
                        csv.writer(f).writerow([entry.name, -1] + [None] * 7)

                counter += 1
                print(f"✓  {entry.name}  ({len(track_crops)} tracks, "
                      f"{len(all_embeddings)} total embeddings so far)")
                if reporter:
                    reporter.video_done(video_name, len(track_crops), len(all_embeddings))

            except Exception as e:
                print(f"✗  {entry.name} failed: {e}")
                if reporter:
                    reporter.video_error(video_name, str(e))
            finally:
                local_path.unlink(missing_ok=True)

    clean_csv(OUTPUT_CSV_PATH)
    print(f"\nDone! Processed {counter} videos this session.")
    print(f"  Detections CSV : {OUTPUT_CSV_PATH}")
    print(f"  Crops folder   : {OUTPUT_CROPS_DIR}/")
    print(f"  Embeddings JSON: {OUTPUT_EMBEDDINGS}  ({len(all_embeddings)} entries)")
    if reporter:
        reporter.pipeline_done(counter, len(all_embeddings))


if __name__ == "__main__":
    main()