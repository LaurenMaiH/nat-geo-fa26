"""
Extract crops from Dropbox videos using bounding box detections from a CSV.

OPTIMIZATIONS:
  1. select_three_crops applied BEFORE video access — only 3 frames sought per track
  2. Forward-seek with grab() instead of CAP_PROP_POS_FRAMES random seek
  3. Prefetch pipeline: download next video while processing current one
  4. Parallel JPEG writes via ThreadPoolExecutor

Two modes (auto-detected from CSV headers):
  Annotated   — CSV has 'Scientific Name' column → subdirs by species
  Unannotated — no species labels               → subdirs by video stem

Output structure (matches overnight pipeline convention):
    Annotated:   crops/Coryphaenoides_yaquinae/id0007_coryphaenoides_yaquinae_f00269_mid.jpg
    Unannotated: crops/PLW_dscm_57_c002/id0007_fish_f00269_mid.jpg

CSV columns used: media / media_id, frame, track_id / tracker_id, class (optional),
                  x, y, width, height, confidence, Scientific Name (optional)
Coordinates are normalized (x, y, width, height relative to frame size).
"""

import os
import csv
import cv2
import tempfile
import dropbox
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CONFIG — update these to match your setup
# ============================================================
APP_KEY        = os.environ["DROPBOX_APP_KEY"]
APP_SECRET     = os.environ["DROPBOX_APP_SECRET"]
REFRESH_TOKEN  = os.environ["DROPBOX_REFRESH_TOKEN"]

MISSION_ROOT   = "/DOEX0096_Palau/PLW_dscm_34/Video"
_MISSION_DIR   = Path(__file__).parent / "processed_missions" / "mission_34"
CSV_PATH       = _MISSION_DIR / "mission_34.csv"
CROPS_DIR      = _MISSION_DIR / "mission_34_crops"

FORCE_SPECIES  = None   # set to e.g. "Chaceon micronesicus" to override CSV labels
CONF_THRESHOLD = 0.15   # minimum confidence to consider a detection at all
PADDING        = 0.05   # fractional padding around each crop (0.05 = 5% of frame)
MIN_CROP_PX    = 32     # discard crops smaller than this in either dimension
WINDOW_PCT     = 0.10   # select_three_crops window size (fraction of track duration)

# Parallelism
WRITE_WORKERS  = 4      # threads for parallel JPEG writes
PREFETCH       = True   # download next video while processing current one


# ============================================================
# SELECT THREE CROPS (pure CSV logic — no video access)
# ============================================================

def select_three_crops(detections: list[dict], window_pct: float = WINDOW_PCT) -> list[dict]:
    """
    Given all detections for one track (list of dicts with 'frame' and 'conf'),
    return at most 3 representative detections: best, early, late.
    Works entirely on CSV data — no video access needed.
    """
    if not detections:
        return []

    detections = sorted(detections, key=lambda d: d["frame"])
    frames = [d["frame"] for d in detections]
    start, end = frames[0], frames[-1]
    duration = max(end - start, 1)
    window = max(int(duration * window_pct), 1)

    best = max(detections, key=lambda d: d["conf"])

    def best_in_window(target):
        lo, hi = target - window, target + window
        candidates = [d for d in detections if lo <= d["frame"] <= hi]
        if candidates:
            return max(candidates, key=lambda d: d["conf"])
        return min(detections, key=lambda d: abs(d["frame"] - target))

    early = best_in_window(start + int(duration * 0.25))
    late  = best_in_window(end   - int(duration * 0.25))

    # Deduplicate by frame (short tracks may collapse to fewer than 3)
    seen = {}
    for tag, det in [("mid", best), ("early", early), ("late", late)]:
        if det["frame"] not in seen:
            seen[det["frame"]] = {**det, "tag": tag}

    return list(seen.values())


# ============================================================
# LOAD CSV — group by video, apply select_three_crops per track
# ============================================================

def load_detections(csv_path: Path) -> tuple[dict[str, list[dict]], bool]:
    """
    Return ({media_filename: [selected_detection, ...]}, is_annotated).
    select_three_crops is applied per track before returning, so only
    the frames we actually need to seek to are included.
    """
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    is_annotated = any(r.get("Scientific Name", "").strip() for r in rows)

    # Group by (media, track_id) first
    by_video_track: dict[str, dict] = defaultdict(lambda: defaultdict(list))
    skipped_conf = 0

    for row in rows:
        conf = float(row.get("confidence") or row.get("conf") or row.get("score") or 1.0)
        if conf < CONF_THRESHOLD:
            skipped_conf += 1
            continue

        media = (row.get("media") or row.get("media_id") or "").strip()
        if not media:
            continue

        track_id = row.get("track_id") or row.get("tracker_id")
        track_id = int(track_id) if track_id not in (None, "", "None") else None

        species = None
        if is_annotated:
            species = FORCE_SPECIES or row.get("Scientific Name", "").strip()

        by_video_track[media][track_id].append({
            "frame":      int(row["frame"]),
            "x":          float(row["x"]),
            "y":          float(row["y"]),
            "w":          float(row["width"]),
            "h":          float(row["height"]),
            "species":    species,
            "class_name": row.get("class", "").strip(),
            "conf":       conf,
            "track_id":   track_id,
        })

    if skipped_conf:
        print(f"Filtered out {skipped_conf} detection(s) below confidence threshold ({CONF_THRESHOLD})")

    # Apply select_three_crops per track, then flatten back to per-video list
    by_video: dict[str, list[dict]] = {}
    total_tracks = total_selected = 0

    for media, tracks in by_video_track.items():
        selected = []
        for track_id, dets in tracks.items():
            if track_id is None:
                # No track ID — fall back to highest-confidence single detection
                chosen = [max(dets, key=lambda d: d["conf"])]
                chosen[0]["tag"] = "best"
            else:
                chosen = select_three_crops(dets)
            selected.extend(chosen)
            total_tracks += 1
            total_selected += len(chosen)

        selected.sort(key=lambda d: d["frame"])
        by_video[media] = selected

    print(f"Selected {total_selected} crops from {total_tracks} tracks "
          f"(avg {total_selected/max(total_tracks,1):.1f} per track)")

    return by_video, is_annotated


# ============================================================
# PARALLEL JPEG WRITER
# ============================================================

def write_crop(args):
    path, crop = args
    cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])


# ============================================================
# CROP EXTRACTION (single video — only seeks selected frames)
# ============================================================

def extract_crops_from_video(
    video_path: Path,
    detections: list[dict],
    crops_dir: Path,
    fallback_label: str | None = None,
    padding: float = PADDING,
    min_px: int = MIN_CROP_PX,
    write_executor: ThreadPoolExecutor = None,
) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Could not open {video_path.name}, skipping")
        return 0, len(detections)

    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    saved = skipped = 0
    current_frame = -1
    write_futures = []

    by_frame = defaultdict(list)
    for d in detections:
        by_frame[d["frame"]].append(d)

    for frame_idx in sorted(by_frame.keys()):
        # ── Fast seek ──────────────────────────────────────────
        if current_frame < 0 or frame_idx < current_frame or \
                (frame_idx - current_frame) > 300:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
        else:
            while current_frame < frame_idx - 1:
                cap.grab()
                current_frame += 1
            ret, frame = cap.read()

        if not ret:
            skipped += len(by_frame[frame_idx])
            continue
        current_frame = frame_idx

        for det in by_frame[frame_idx]:
            label = det["species"] or fallback_label
            if not label:
                skipped += 1
                continue

            cx = det["x"] * W
            cy = det["y"] * H
            bw = det["w"] * W
            bh = det["h"] * H
            pad_x = bw * padding
            pad_y = bh * padding

            x1 = max(int(cx - pad_x), 0)
            y1 = max(int(cy - pad_y), 0)
            x2 = min(int(cx + bw + pad_x), W)
            y2 = min(int(cy + bh + pad_y), H)

            if (x2 - x1) < min_px or (y2 - y1) < min_px:
                skipped += 1
                continue

            crop = frame[y1:y2, x1:x2].copy()

            out_dir = crops_dir / label
            out_dir.mkdir(parents=True, exist_ok=True)

            tag = det.get("tag", "det")
            tid = det.get("track_id")
            if det.get("species"):
                cls_slug = det["species"].lower().replace(" ", "_")
            elif det.get("class_name"):
                cls_slug = det["class_name"].replace(" ", "_")
            else:
                cls_slug = "unknown"
            if isinstance(tid, int):
                fname = f"id{tid:04d}_{cls_slug}_f{frame_idx:05d}_{tag}.jpg"
            else:
                fname = f"id{tid or 'notrack'}_{cls_slug}_f{frame_idx:05d}_{tag}.jpg"
            out_path = out_dir / fname

            if write_executor:
                write_futures.append(write_executor.submit(write_crop, (out_path, crop)))
            else:
                write_crop((out_path, crop))

            saved += 1

    cap.release()

    for f in write_futures:
        f.result()

    return saved, skipped


# ============================================================
# PREFETCH DOWNLOADER
# ============================================================

def prefetch_downloader(dbx, items, tmpdir: Path, q: queue.Queue, pbar):
    for media_name, detections in items:
        dropbox_path = MISSION_ROOT + "/" + media_name
        local_path   = tmpdir / media_name
        try:
            dbx.files_download_to_file(str(local_path), dropbox_path)
            q.put((media_name, local_path, detections, None))
        except Exception as e:
            q.put((media_name, None, detections, str(e)))
    q.put(None)


# ============================================================
# MAIN
# ============================================================

def main():
    CROPS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading detections from {CSV_PATH}...")
    by_video, is_annotated = load_detections(CSV_PATH)
    mode = "annotated (subdirs by species)" if is_annotated else "unannotated (subdirs by video)"
    print(f"Mode: {mode}")
    print(f"Found detections across {len(by_video)} video(s)\n")

    dbx = dropbox.Dropbox(
        app_key=APP_KEY,
        app_secret=APP_SECRET,
        oauth2_refresh_token=REFRESH_TOKEN,
    )

    total_saved = total_skipped = 0
    items = list(by_video.items())

    with tempfile.TemporaryDirectory(prefix="crop_extraction_") as tmpdir:
        tmpdir = Path(tmpdir)

        with ThreadPoolExecutor(max_workers=WRITE_WORKERS) as write_executor:

            if PREFETCH and len(items) > 1:
                dl_queue = queue.Queue(maxsize=2)
                dl_thread = threading.Thread(
                    target=prefetch_downloader,
                    args=(dbx, items, tmpdir, dl_queue, None),
                    daemon=True,
                )
                dl_thread.start()

                with tqdm(total=len(items), desc="Videos", unit="video") as pbar:
                    while True:
                        item = dl_queue.get()
                        if item is None:
                            break
                        media_name, local_path, detections, err = item

                        if err:
                            print(f"\n  ✗ Download failed for {media_name}: {err}")
                            total_skipped += len(detections)
                        else:
                            print(f"\nProcessing {media_name} ({len(detections)} crops to extract)...")
                            fallback_label = None if is_annotated else Path(media_name).stem
                            saved, skipped = extract_crops_from_video(
                                local_path, detections, CROPS_DIR,
                                fallback_label=fallback_label,
                                write_executor=write_executor,
                            )
                            total_saved   += saved
                            total_skipped += skipped
                            print(f"  ✓ {saved} crops saved, {skipped} skipped")
                            local_path.unlink(missing_ok=True)

                        pbar.update(1)

            else:
                for media_name, detections in tqdm(items, desc="Videos", unit="video"):
                    dropbox_path = MISSION_ROOT + "/" + media_name
                    local_path   = tmpdir / media_name

                    print(f"\nDownloading {media_name} ({len(detections)} crops to extract)...")
                    try:
                        dbx.files_download_to_file(str(local_path), dropbox_path)
                    except Exception as e:
                        print(f"  ✗ Download failed: {e}")
                        total_skipped += len(detections)
                        continue

                    fallback_label = None if is_annotated else Path(media_name).stem
                    saved, skipped = extract_crops_from_video(
                        local_path, detections, CROPS_DIR,
                        fallback_label=fallback_label,
                        write_executor=write_executor,
                    )
                    total_saved   += saved
                    total_skipped += skipped
                    print(f"  ✓ {saved} crops saved, {skipped} skipped")
                    local_path.unlink(missing_ok=True)

    print(f"\nDone. {total_saved} crops saved to {CROPS_DIR}/")
    print(f"      {total_skipped} detections skipped (bad frame / too small / no label / low confidence)")

    print("\nCrop counts by label:")
    for label_dir in sorted(CROPS_DIR.iterdir()):
        if label_dir.is_dir():
            n = len(list(label_dir.iterdir()))
            print(f"  {label_dir.name}: {n}")


main()