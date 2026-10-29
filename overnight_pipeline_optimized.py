#!/usr/bin/env python3
"""
Overnight Detection + Extraction + Embedding

For each video in a mission folder (Dropbox or a local drive — see application/video_sources.py):
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
import re
import sys
import csv
import json
import time
import shutil
import platform
import tempfile
import subprocess
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from collections import Counter, defaultdict

# Apple Silicon (mps): run ops Metal doesn't implement on the CPU instead of failing.
# Read by PyTorch, so it must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
import torch
import yaml
import gdown
import ultralytics
from ultralytics import YOLOWorld, settings
from tqdm import tqdm
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent / "application"))
from video_sources import (
    DropboxSource, check_stem_collisions, has_completion_records, is_complete,
    load_manifest, manifest_entry, replace_file, video_stem, write_json_atomic, write_manifest,
)

load_dotenv()

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Read when main() runs, not at import, so the app can set any of these on the
# module before calling main(). Importing this file creates no files or folders.
MISSION_ID   = "PLW_dscm_03"          # ← only line to change between missions
DEPLOYMENT_ROOT = os.environ.get("DROPBOX_DEPLOYMENT_ROOT", "/DOEX0096_Palau")
MISSION_PATH = f"{DEPLOYMENT_ROOT}/{MISSION_ID}/Video"   # ← and this one

# Video source (video_sources.LocalSource / DropboxSource). The app sets this;
# None means Dropbox at DEPLOYMENT_ROOT, with MISSION_PATH as the Dropbox folder.
SOURCE = None

# Mission output folder. The app sets processed_missions/<deployment>/<mission>;
# None means processed_missions/<MISSION_ID> next to this script.
OUTPUT_DIR = None

SELECTED_VIDEOS = None  # set to a list of filenames, or None for all
START_FROM      = "PLW_dscm_03_c011.mp4"  # set to a filename to start from that video (inclusive)

CHECKPOINT_PATH = Path(__file__).parent / "checkpoints" / "best.pt"
# Google Drive file IDs (download by ID works on gdown 4.x–6.x; 6.0 dropped fuzzy URLs)
BIOCLIP_CHECKPOINT_DRIVE_ID = "1Qec1Gzxl8zAazuLof8a2LUSrbOR7GmtT"   # fine-tuned BioCLIP best.pt (~3.6 GB)
YOLO_WORLD_DRIVE_ID         = "1hh576zOzpUqgWSjIdSkR4EwgYQ8kH406"   # yolov8x-worldv2.pt

_MISSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")

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
# None = the INFERENCE_DEVICE environment variable (cpu | cuda | mps) if set, else the
# best available: cuda > mps > cpu. Resolved when main() runs.
INFERENCE_DEVICE = None
CROP_LABELS      = ["early", "mid", "late"]
DEFAULT_YOLO_WEIGHTS = "yolov8x-worldv2.pt"


@dataclass
class SpeedOptions:
    """Speed/accuracy knobs, passed to main(options=...). The defaults are the
    pipeline's normal behavior; benchmark.py varies them."""
    detect_every: int = 1            # run YOLO-World + ByteTrack on every Nth frame (motion gate still applies)
    yolo_weights: str = DEFAULT_YOLO_WEIGHTS   # e.g. yolov8l-worldv2.pt / yolov8m-worldv2.pt (Ultralytics downloads these by name)
    imgsz: int | None = None         # YOLO inference size; None = the checkpoint's own (640 for yolov8x-worldv2)
    crop_from_original: bool = False # cut crops from the original-resolution video instead of the resized copy


def resolve_device(requested=None) -> str:
    """The device to run on: `requested` or the INFERENCE_DEVICE env var if either is
    set (error if it isn't available), else cuda > mps > cpu."""
    req = (requested or os.environ.get("INFERENCE_DEVICE") or "").strip().lower()
    mps_ok = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    if req:
        if req not in ("cpu", "cuda", "mps"):
            raise ValueError(f"INFERENCE_DEVICE={req!r} isn't recognised: use cpu, cuda or mps")
        if req == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("INFERENCE_DEVICE=cuda, but PyTorch can't see a CUDA GPU (no NVIDIA "
                               "GPU or driver, or a CPU-only PyTorch build)")
        if req == "mps" and not mps_ok:
            raise RuntimeError("INFERENCE_DEVICE=mps, but Apple's Metal (MPS) backend isn't available "
                               "(it only exists on Apple Silicon Macs, running natively, not in Docker)")
        return req
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if mps_ok else "cpu"

# ── OUTPUT PATHS ──────────────────────────────────────────────────────────────
def output_paths() -> SimpleNamespace:
    """This run's output files, from MISSION_ID / OUTPUT_DIR as they are when called."""
    if not _MISSION_ID_RE.match(MISSION_ID or ""):
        raise ValueError(f"Invalid MISSION_ID {MISSION_ID!r} (letters, digits, _ and - only)")
    mission_dir = Path(OUTPUT_DIR) if OUTPUT_DIR else Path(__file__).parent / "processed_missions" / MISSION_ID
    return SimpleNamespace(
        dir        = mission_dir,
        csv        = mission_dir / f"{MISSION_ID}_detections.csv",
        crops      = mission_dir / f"{MISSION_ID}_crops",
        embeddings = mission_dir / f"{MISSION_ID}_embeddings.json",
        manifest   = mission_dir / f"{MISSION_ID}_manifest.json",
        run_log    = mission_dir / f"{MISSION_ID}_run_log.json",
    )


# ── MODEL SETUP ───────────────────────────────────────────────────────────────
_model_cache: dict = {}   # yolo weights name → (model, tracker config)
_downloads: list = []     # weight downloads made this run (for the run log's setup costs)


def _record_download(what: str, path: Path, started: float):
    if path.exists():
        _downloads.append({"file": what, "seconds": round(time.perf_counter() - started, 1),
                           "bytes": path.stat().st_size})


def load_model_and_config(weights: str = DEFAULT_YOLO_WEIGHTS):
    if weights in _model_cache:
        return _model_cache[weights]

    YOLO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    settings.update({"weights_dir": str(YOLO_CACHE_DIR), "runs_dir": str(YOLO_CACHE_DIR / "runs")})

    model_path = YOLO_CACHE_DIR / weights
    if not model_path.exists():
        started = time.perf_counter()
        if weights == DEFAULT_YOLO_WEIGHTS:
            print(f"Downloading YOLO-World model to {model_path}...")
            gdown.download(id=YOLO_WORLD_DRIVE_ID, output=str(model_path), quiet=False)
        else:
            from ultralytics.utils.downloads import attempt_download_asset
            print(f"Downloading {weights} from the Ultralytics release assets to {model_path}...")
            attempt_download_asset(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"YOLO-World weights {weights!r} could not be downloaded by name")
        _record_download(weights, model_path, started)

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

    _model_cache[weights] = (model, str(custom_yaml))
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


def _get_video_height(path: Path) -> int:
    """Height of the first video stream in pixels, or 0 if it can't be read."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=height", "-of", "csv=p=0", str(path)],
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
def _download_errors() -> tuple:
    """Exceptions that mean the Drive download itself failed (network, Drive refused the
    file, disk). requests' errors subclass OSError. Anything else, e.g. a TypeError
    from a gdown API change, is a bug and must not be mistaken for a download failure."""
    exc = getattr(gdown, "exceptions", None)
    names = ("DownloadError", "FileURLRetrievalError")   # gdown 6.x / 4.7+
    return (OSError,) + tuple(getattr(exc, n) for n in names if exc is not None and hasattr(exc, n))


def _download_checkpoint() -> bool:
    """Fetch the fine-tuned checkpoint to CHECKPOINT_PATH. False on a download failure."""
    print(f"Finetuned weights not found at {CHECKPOINT_PATH}, downloading (~3.6 GB)...")
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        gdown.download(id=BIOCLIP_CHECKPOINT_DRIVE_ID, output=str(CHECKPOINT_PATH), quiet=False)
        _record_download(CHECKPOINT_PATH.name, CHECKPOINT_PATH, started)
    except _download_errors() as dl_err:
        print(f"Finetuned checkpoint download failed ({dl_err}).")
        # gdown downloads to <name>XXXX.part and renames on success; drop the leftover
        for part in CHECKPOINT_PATH.parent.glob(f"{CHECKPOINT_PATH.name}*.part"):
            part.unlink(missing_ok=True)
        return False
    return CHECKPOINT_PATH.exists()


def load_embedding_model():
    """Returns (weights, model, preprocess); weights is "finetuned", "pretrained"
    (checkpoint download failed) or "placeholder" (no BioCLIP model at all; main()
    refuses to run with it)."""
    if CHECKPOINT_PATH and not Path(CHECKPOINT_PATH).exists():
        if not _download_checkpoint():
            print("WARNING: using pretrained BioCLIP-2 weights instead of the fine-tuned "
                  "checkpoint. Embeddings and clusters will differ from the fine-tuned model's.")

    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            "hf-hub:imageomics/bioclip-2", device=INFERENCE_DEVICE
        )
    except Exception as e:
        print(f"BioCLIP not available ({e}) -- only placeholder embeddings are possible.")
        return ("placeholder", None, None)

    weights = "pretrained"
    if CHECKPOINT_PATH and Path(CHECKPOINT_PATH).exists():
        print(f"Loading finetuned weights from {CHECKPOINT_PATH}...")
        try:
            ckpt = torch.load(CHECKPOINT_PATH, map_location=INFERENCE_DEVICE)
            model.visual.load_state_dict(ckpt["model_state"])
        except Exception as e:
            raise RuntimeError(
                f"Could not load the fine-tuned checkpoint {CHECKPOINT_PATH} ({type(e).__name__}: {e}). "
                "The file is probably corrupted or incomplete. Delete it and start the run "
                "again to download a fresh copy."
            ) from e
        print("Finetuned weights loaded ✓")
        weights = "finetuned"
    else:
        print("Using pretrained BioCLIP-2 weights")

    model.eval()
    return (weights, model, preprocess)


def get_embeddings_batch(embed_info, images_bgr: list[np.ndarray]) -> np.ndarray:
    """
    Embed a batch of BGR images in one forward pass.
    Returns (N, D) float32 array of L2-normalised embeddings.
    """
    kind, model, preprocess = embed_info
    if kind != "placeholder":
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


# ──────── RUN LOG (timings) ──────────────────────────────────────────────────
# <mission>_run_log.json: {"runs": [...]}, one entry per pipeline run that processed
# videos: machine, settings, embedding weights, one-time setup costs, and per-video
# stage timings. Separate from the manifest; nothing else reads it.

def _cpu_name() -> str:
    try:
        if sys.platform == "win32":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        if sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True).stdout.strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def machine_info() -> dict:
    import psutil   # installed with ultralytics
    gpu = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu = {"name": props.name, "memory_gb": round(props.total_memory / 2**30, 1),
               "cuda": torch.version.cuda}
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        gpu = {"name": "Apple Silicon GPU (Metal / MPS)"}
    return {
        "os":            platform.platform(),
        "cpu":           _cpu_name(),
        "cores_physical": psutil.cpu_count(logical=False),
        "cores_logical": psutil.cpu_count(logical=True),
        "ram_gb":        round(psutil.virtual_memory().total / 2**30, 1),   # in Docker: what the VM allows
        "gpu":           gpu,
        "device":        INFERENCE_DEVICE,
        "torch":         torch.__version__,
        "ultralytics":   ultralytics.__version__,
        "python":        platform.python_version(),
    }


def run_settings(options: "SpeedOptions", model=None) -> dict:
    overrides = getattr(model, "overrides", None) or {}
    return {
        "conf_threshold": CONF_THRESHOLD, "iou_threshold": IOU_THRESHOLD,
        "min_track_len": MIN_TRACK_LEN, "crops_per_track": CROPS_PER_TRACK, "crop_pad": CROP_PAD,
        "apply_clahe": APPLY_CLAHE, "trim_black": TRIM_BLACK, "preresize": PRERESIZE,
        **asdict(options),
        "imgsz_effective": options.imgsz or overrides.get("imgsz"),
        "device": INFERENCE_DEVICE,
    }


def _load_run_log(path: Path) -> dict:
    if not path.exists():
        return {"runs": []}
    try:
        log = json.loads(path.read_text())
        if isinstance(log, dict) and isinstance(log.get("runs"), list):
            return log
    except ValueError:
        pass
    backup = path.with_name(path.name + ".unreadable")
    replace_file(path, backup)
    print(f"Warning: {path.name} was unreadable; moved it to {backup.name} and started a new one")
    return {"runs": []}


def _video_record(entry, status: str, stats: dict, stages: dict, started: float, **extra) -> dict:
    total   = stats.get("frames_total") or 0
    active  = stats.get("frames_motion_active")
    return {
        "video": entry.stem, "filename": entry.name, "status": status, **extra,
        **stats,
        "motion_gate_hit_rate": round(active / total, 4) if total and active is not None else None,
        "stages_s": {k: round(v, 2) for k, v in stages.items()},
        "total_s": round(time.perf_counter() - started, 2),
    }


# ──────── RESUME / COMPLETION RECORDS ─────────────────────────────────────────
# Each finished video (including ones with zero tracks) is marked "complete" in the
# mission manifest; a rerun skips those and redoes everything else from scratch.

def plan_videos(all_entries, manifest: dict, selected=None, start_from=None) -> SimpleNamespace:
    """Which videos this run processes: SELECTED_VIDEOS / START_FROM narrow the
    listing, then videos already marked complete are skipped."""
    entries = all_entries
    missing = set()
    if selected is not None:
        sel     = set(selected)
        entries = [e for e in entries if e.name in sel]
        missing = sel - {e.name for e in entries}
    start_found = None
    if start_from:
        names = [e.name for e in entries]
        start_found = start_from in names
        if start_found:
            entries = entries[names.index(start_from):]
    done = [e for e in entries if is_complete(manifest.get(e.stem))]
    return SimpleNamespace(
        todo=[e for e in entries if not is_complete(manifest.get(e.stem))],
        done=done, missing=missing, start_found=start_found,
    )


def _last_csv_video(csv_path: Path):
    """media_id of the CSV's last data row (the old resume point), or None."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return None
    last = None
    with open(csv_path, newline="") as f:
        for i, row in enumerate(csv.reader(f)):
            if i > 0 and row:
                last = row[0]
    return last


def legacy_completion(source, all_entries, manifest: dict, csv_path: Path, embeddings: dict) -> dict:
    """Completion records for a mission processed before they existed (no manifest,
    or one without "status"), using the old resume rule: every video up to and
    including the one in the CSV's last row is done. {} if nothing applies."""
    if has_completion_records(manifest):
        return {}
    last = _last_csv_video(csv_path)
    names = [e.name for e in all_entries]
    if last not in names:
        return {}
    n_tracks = Counter(k.rsplit("__", 1)[0] for k in embeddings)
    return {
        e.stem: {**manifest_entry(source, e), "status": "complete", "n_tracks": n_tracks[e.stem],
                 "finished_at": None, "inferred_from_csv": True}
        for e in all_entries[:names.index(last) + 1]
    }


def purge_video_outputs(stems: set, paths: SimpleNamespace, embeddings: dict) -> bool:
    """Delete what an unfinished run left behind for these videos (detection rows,
    crops folder, embeddings) so they're redone cleanly. Also drops the sentinel rows
    older versions wrote ("<video>,-1,,,..."). Edits `embeddings` in place; returns True
    if it changed (the caller saves it)."""
    removed = 0
    if paths.csv.exists() and paths.csv.stat().st_size > 0:
        fd, tmp = tempfile.mkstemp(prefix=f".{paths.csv.stem}_", suffix=".tmp", dir=paths.csv.parent)
        try:
            with open(paths.csv, newline="") as src, os.fdopen(fd, "w", newline="") as dst:
                writer = csv.writer(dst)
                for i, row in enumerate(csv.reader(src)):
                    if i > 0 and (not row or video_stem(row[0]) in stems
                                  or (len(row) > 1 and row[1] == "-1")):
                        removed += 1
                        continue
                    writer.writerow(row)
            if removed:
                replace_file(tmp, paths.csv)
        finally:
            Path(tmp).unlink(missing_ok=True)

    for stem in stems:
        crop_dir = paths.crops / stem
        if crop_dir.is_dir():
            shutil.rmtree(crop_dir)

    stale = [k for k in embeddings if k.rsplit("__", 1)[0] in stems]
    for k in stale:
        del embeddings[k]
    if removed or stale:
        print(f"Removed {removed} leftover CSV row(s) and {len(stale)} embedding(s) "
              f"from unfinished videos")
    return bool(stale)


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
    options: SpeedOptions = None,
    stats: dict = None,
    stages: dict = None,
) -> dict:
    """Detect, track and crop one video. Fills `stats` (frame counts, resolutions,
    crop sizes) and `stages` (seconds for resize / detect_track / crop) for the run log."""
    opts   = options or SpeedOptions()
    stats  = {} if stats is None else stats
    stages = {} if stages is None else stages
    resized_path = None
    work_path    = video_path

    src = cv2.VideoCapture(str(video_path))
    src_w, src_h = int(src.get(cv2.CAP_PROP_FRAME_WIDTH)), int(src.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps, src_frames = src.get(cv2.CAP_PROP_FPS) or 0, int(src.get(cv2.CAP_PROP_FRAME_COUNT))
    src.release()
    stats.update({"source_resolution": f"{src_w}x{src_h}", "fps": round(src_fps, 3),
                  "duration_s": round(src_frames / src_fps, 2) if src_fps else None})

    # ── Resize into the same tmpdir as the download (downscale only) ──
    t0 = time.perf_counter()
    src_height = _get_video_height(video_path) if PRERESIZE else 0
    if not PRERESIZE:
        stats["resize"] = "off"
    elif 0 < src_height <= PRERESIZE:
        print(f"  Source is {src_height}p (<= {PRERESIZE}p), skipping resize.")
        stats["resize"] = "skipped (source already small enough)"
    else:
        tmp = tmpdir / f"{video_path.stem}_rs{PRERESIZE}.mp4"
        if resize_video(video_path, tmp, PRERESIZE):
            work_path    = tmp
            resized_path = tmp
            stats["resize"] = "done"
        else:
            print("  Resize failed, using original resolution.")
            stats["resize"] = "failed"
    stages["resize"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    cap = cv2.VideoCapture(str(work_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {work_path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stats["processed_resolution"] = f"{width}x{height}"

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
    frames_active   = 0
    n_detections    = 0
    track_kwargs    = {"imgsz": opts.imgsz} if opts.imgsz else {}

    with tqdm(total=total_frames, desc=f"  YOLO {video_name}", unit="frame", leave=True) as pbar:
        while frame_idx < total_frames:
            if opts.detect_every > 1 and frame_idx % opts.detect_every:
                # Not a detection frame: advance without decoding the image
                if not cap.grab():
                    break
            else:
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(gray, median_bg)
                _, fg = cv2.threshold(diff, BG_THRESH, 255, cv2.THRESH_BINARY)
                fg    = cv2.morphologyEx(cv2.morphologyEx(fg, cv2.MORPH_OPEN, morph_kernel),
                                         cv2.MORPH_DILATE, morph_kernel)
                _, _, blob_stats, _ = cv2.connectedComponentsWithStats(fg)
                has_activity   = any(s[cv2.CC_STAT_AREA] >= MIN_BLOB for s in blob_stats[1:])

                if has_activity:
                    frames_active += 1
                    infer   = clahe_L_median(frame, clahe) if APPLY_CLAHE else frame
                    results = model.track(
                        infer, persist=True, tracker=tracker_config,
                        verbose=False, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
                        device=INFERENCE_DEVICE, **track_kwargs,
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
                            n_detections += 1
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
    stats.update({
        "frames_total": total_frames, "black_frames_skipped": first_real,
        "frames_motion_active": frames_active, "frames_inferred": frames_inferred,
        "n_detections": n_detections, "n_tracks_raw": len(track_detections),
        "n_tracks_kept": len(long_tracks),
    })
    stages["detect_track"] = time.perf_counter() - t0
    t0 = time.perf_counter()

    video_crops_dir = crops_dir / video_name
    video_crops_dir.mkdir(parents=True, exist_ok=True)
    track_crop_paths: dict = {}

    if not long_tracks:
        if resized_path and resized_path.exists():
            resized_path.unlink()
        stages["crop"] = time.perf_counter() - t0
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
    # Crops come from the video detection ran on, or (crop_from_original) from the
    # original file with the boxes scaled back up (frame numbers must line up, so the
    # frame counts are compared).
    crop_source, crop_w, crop_h = work_path, width, height
    if opts.crop_from_original and resized_path:
        crop_source, crop_w, crop_h = video_path, src_w, src_h
        if src_frames != total_frames + first_real:
            stats["frame_count_mismatch"] = [src_frames, total_frames + first_real]
            print(f"  WARNING: original has {src_frames} frames, resized copy {total_frames + first_real}; "
                  "crops from the original may be off by a few frames")
    sx, sy = crop_w / width, crop_h / height
    crop_sizes = []

    cap2 = cv2.VideoCapture(str(crop_source))
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
            if crop_source is not work_path:
                x1p, x2p = int(x1p * sx), min(crop_w, round(x2p * sx))
                y1p, y2p = int(y1p * sy), min(crop_h, round(y2p * sy))
            crop = frame[y1p:y2p, x1p:x2p]
            if crop.size == 0:
                continue
            crop_sizes.append(crop.shape[:2])
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

    stats["crops_from"] = "original" if crop_source is not work_path else "processed"
    if crop_sizes:
        stats["n_crops"] = len(crop_sizes)
        stats["crop_px_mean"] = [round(float(np.mean([s[1] for s in crop_sizes])), 1),   # width, height
                                 round(float(np.mean([s[0] for s in crop_sizes])), 1)]
    stages["crop"] = time.perf_counter() - t0
    return track_crop_paths


# ──────── MAIN ───────────────────────────────────────────────────────────────
def main(reporter=None, options: SpeedOptions = None):
    global INFERENCE_DEVICE
    options = options or SpeedOptions()
    chosen_by = "INFERENCE_DEVICE" if (INFERENCE_DEVICE or os.environ.get("INFERENCE_DEVICE")) else "auto-detected"
    INFERENCE_DEVICE = resolve_device(INFERENCE_DEVICE)
    print(f"Using device: {INFERENCE_DEVICE} ({chosen_by})")
    paths = output_paths()

    # List videos first so a bad path / missing credentials fails before model load
    source      = SOURCE or DropboxSource(DEPLOYMENT_ROOT)
    all_entries = source.list_videos(MISSION_PATH)
    print(f"Found {len(all_entries)} videos in {source.describe(MISSION_PATH)}")
    check_stem_collisions(all_entries)

    all_embeddings: dict = {}
    if paths.embeddings.exists():
        with open(paths.embeddings) as f:
            all_embeddings = json.load(f)
        print(f"Loaded {len(all_embeddings)} existing embeddings from {paths.embeddings}")

    manifest = load_manifest(paths.manifest)
    legacy   = legacy_completion(source, all_entries, manifest, paths.csv, all_embeddings)
    if legacy:
        manifest.update(legacy)
        print(f"No completion records yet: treating {len(legacy)} videos up to "
              f"'{max(e['filename'] for e in legacy.values())}' (last in the detections CSV) as done")

    plan = plan_videos(all_entries, manifest, SELECTED_VIDEOS, START_FROM)
    if SELECTED_VIDEOS is not None:
        if plan.missing:
            print(f"Warning: not found in source folder: {plan.missing}")
        print(f"Selected {len(plan.todo) + len(plan.done)}/{len(all_entries)} videos")
    if plan.start_found:
        print(f"Starting from '{START_FROM}'")
    elif START_FROM:
        print(f"Start video '{START_FROM}' not in the list, starting from the beginning")
    if plan.done:
        print(f"Skipping {len(plan.done)} video(s) already marked complete")
    entries = plan.todo
    print(f"{len(entries)} video(s) to process")

    counter = 0
    if entries:
        # One-time setup, timed apart from the per-video numbers (downloads separately)
        _downloads.clear()
        t0 = time.perf_counter()
        model, tracker = load_model_and_config(options.yolo_weights)
        yolo_load_s    = time.perf_counter() - t0
        t0 = time.perf_counter()
        embed_info     = load_embedding_model()
        embed_load_s   = time.perf_counter() - t0
        weights        = embed_info[0]
        if reporter:
            reporter.embedding_weights(weights)
        if weights == "placeholder":
            raise RuntimeError(
                "No BioCLIP embedding model could be loaded (see the log above), so embeddings "
                "would be random placeholders. Stopping instead of producing meaningless clusters."
            )

        paths.crops.mkdir(parents=True, exist_ok=True)
        if legacy:
            write_manifest(paths.manifest, manifest)
        # Redo unfinished videos from scratch (a run killed mid-video leaves partial rows)
        if purge_video_outputs({e.stem for e in entries}, paths, all_embeddings):
            write_json_atomic(paths.embeddings, all_embeddings)

        downloaded = {d["file"]: d["seconds"] for d in _downloads}
        run_log = _load_run_log(paths.run_log)
        run = {
            "started_at":  datetime.now().isoformat(timespec="seconds"),
            "finished_at": None,
            "mission_id":  MISSION_ID,
            "source":      source.describe(MISSION_PATH),
            "machine":     machine_info(),
            "settings":    run_settings(options, model),
            "embedding_weights": weights,
            "setup": {
                # load times exclude download time; the first BioCLIP-2 load also
                # fetches its base weights from Hugging Face if they aren't cached
                "yolo_model_load_s":      round(yolo_load_s - downloaded.get(options.yolo_weights, 0), 1),
                "embedding_model_load_s": round(embed_load_s - downloaded.get(CHECKPOINT_PATH.name, 0), 1),
                "downloads":              list(_downloads),
            },
            "videos": [],
        }
        run_log["runs"].append(run)
        write_json_atomic(paths.run_log, run_log, indent=1)

        total = len(entries)
        with tempfile.TemporaryDirectory(prefix="natgeo_") as tmpdir:
            tmpdir = Path(tmpdir)
            for idx, entry in enumerate(entries):
                if reporter and reporter.should_stop():
                    print("Stop signal received — exiting after current video.")
                    break

                video_name = entry.stem
                media_id   = entry.name

                if reporter:
                    reporter.begin_video(video_name, idx + 1, total)

                print(f"\n{'='*60}")
                video_started = time.perf_counter()
                stats, stages = {}, {}
                try:
                    manifest[video_name] = {**manifest_entry(source, entry), "status": "in_progress"}
                    write_manifest(paths.manifest, manifest)

                    if source.type == "dropbox":
                        print(f"Downloading: {entry.name}")
                        if reporter:
                            reporter.stage("downloading")
                    else:
                        print(f"Reading: {entry.rel_path}")

                    t0 = time.perf_counter()
                    with source.local_copy(entry) as local_path:
                        stages["open"] = time.perf_counter() - t0   # the download, for Dropbox
                        if reporter:
                            reporter.stage("detecting")
                        track_crops = process_video(
                            local_path, media_id, video_name, model, tracker,
                            paths.csv, paths.crops,
                            tmpdir=tmpdir,
                            reporter=reporter,
                            options=options, stats=stats, stages=stages,
                        )

                    # ── Batch embedding per track ─────────────────────────────
                    t0 = time.perf_counter()
                    total_crops = sum(len(v) for v in track_crops.values())
                    print(f"  Embedding {total_crops} crops across {len(track_crops)} tracks...")
                    if reporter:
                        reporter.stage("embedding")
                    track_items  = list(track_crops.items())
                    n_emb_tracks = len(track_items)
                    n_embedded   = 0
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
                        n_embedded += 1
                        if reporter and n_emb_tracks > 0:
                            reporter.set_video_pct(2/3 + (emb_i + 1) / n_emb_tracks * (1/3))

                    write_json_atomic(paths.embeddings, all_embeddings)
                    stages["embed"] = time.perf_counter() - t0

                    manifest[video_name].update({
                        "status":      "complete",
                        "n_tracks":    n_embedded,
                        "finished_at": datetime.now().isoformat(timespec="seconds"),
                        "weights":     weights,
                    })
                    write_manifest(paths.manifest, manifest)

                    run["videos"].append(_video_record(entry, "complete", stats, stages, video_started,
                                                       n_tracks=n_embedded))
                    write_json_atomic(paths.run_log, run_log, indent=1)

                    counter += 1
                    print(f"✓  {entry.name}  ({n_embedded} tracks, "
                          f"{len(all_embeddings)} total embeddings so far)")
                    if reporter:
                        reporter.video_done(video_name, n_embedded, len(all_embeddings))

                except Exception as e:
                    print(f"✗  {entry.name} failed: {e}")
                    if reporter:
                        reporter.video_error(video_name, str(e))
                    try:
                        manifest[video_name] = {**manifest_entry(source, entry),
                                                "status": "failed", "error": str(e)}
                        write_manifest(paths.manifest, manifest)
                        if purge_video_outputs({video_name}, paths, all_embeddings):
                            write_json_atomic(paths.embeddings, all_embeddings)
                        run["videos"].append(_video_record(entry, "failed", stats, stages, video_started,
                                                           error=str(e)))
                        write_json_atomic(paths.run_log, run_log, indent=1)
                    except Exception as cleanup_err:
                        print(f"  (could not clean up after the failure: {cleanup_err})")

        run["finished_at"] = datetime.now().isoformat(timespec="seconds")
        write_json_atomic(paths.run_log, run_log, indent=1)

    print(f"\nDone! Processed {counter} videos this session.")
    print(f"  Detections CSV : {paths.csv}")
    print(f"  Crops folder   : {paths.crops}/")
    print(f"  Embeddings JSON: {paths.embeddings}  ({len(all_embeddings)} entries)")
    if entries:
        print(f"  Run log        : {paths.run_log}")
    if reporter:
        reporter.pipeline_done(counter, len(all_embeddings))


if __name__ == "__main__":
    main()
