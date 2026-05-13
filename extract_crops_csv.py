"""
Extract crops from Dropbox videos using bounding box detections from a CSV.

Output structure (compatible with augment_images() and CropDataset):
    crops/
        Coryphaenoides/
            PLW_dscm_57_c001_frame0042_track3.jpg
        Synaphobranchus/
            ...

CSV columns used: media, frame, x, y, width, height, Scientific Name
Coordinates are normalized (x, y, width, height relative to frame size).
"""

import csv
import cv2
import tempfile
import dropbox
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


# ============================================================
# CONFIG — update these to match your setup
# ============================================================

ACCESS_TOKEN  = "sl.u.AGeE-YE2b2Pai822S1aqhAceLXRdWbSw4q_QVC5sXSnlWcBn7tRv_ST5A2NSLgenwebXna5kvomwbQT_GLwu-ET_9JYKAzsbd8-sikZGhXtLl3jY63xORWdWxsW9oS22utXy_7eXXqGMd2SjLvEjtl7o_6jpze5CM7tLbabvi-YLaAaaNhPk6k_ADSoHb4NkNsS2znV9V24AjVW6nUgyozHHVFj6twmbRoneCKwRy6t7V-a8xhrNBDaQHFPMASbwWJvxmW9yoQM_waF_r4mSr-lPY2C4U-S5pLXkAdkIaSOmq4SJg6eIW463_LThJ8f8hUSjoKJTCAo76vm9qusp_Oi6_Kk_NqTqW_FdEvhU7RiS3js7-htuKFtTf0HmE2FjRUVobNXjvMXl0111W5fFpUAdxLZjRWiDPhvcJp6SjxbpAxmLO3WzPFT--pmWpKw1rlMkfbfoQ5Lq9FbC2Tbz_gJHa7XVeIfzdYo3xq1x20AaxSQKfweX3JHPGZV1kcacKqaszp0Urv8Uhb2lGc_hQMcpxQf-JMV_ABwX9fuQESchqyzrPc4rFe8x80wyxZK85Ezw-1kxgMfVGNdt0bn0Mp58ZfYYiwmSatVFnQolRB41GwLN9tnDqjGN1l65hRPpLOwCakO8unAGEY7J6SG2J5BafFmhsXCX3osT3ZJ_JRPB8sn2svKAtc-ZR0FLlky_x24VYV3vm6fMi7gK9_znM0wulBjEXQ6CcNK_xmR9wDg_4616KyyVtbPIP9MEi2w9s-sL_b9ul5GZhihgdn9FZtg0MdFHDMTdyylKa48m0yG4afbAESU0fsTuLFqRZvokY5BncwpNO3fomDBhj4Dhh15IIgyDuF1-jELrN5qtpVw7fDs4N7zXEsjf89XpXiRFE4ApOPwe0dE8W8Osh-QtYHqU6RfSjur7sGLo5_nc5D2mOi7FooF-oAkNjp3yohLK6i6sogkHZeWtwffDb-MWJYN3_vVMSTA06Xp2XRH_cF5ByC9mlVMwxgeE3UDdv22RUIAl3pKfsAFLsKtv31VuzgQ5PaldIFfjOBwQvNdzWwkBLQSkdAUlhm3mVjpNdZUQNKgn81rRx8ImvEch364FkenUjNSvCE9hOiICvWbM1dwjt9raeXcZT37CXtHFHhjvDUh2vA5ZQJHjmOile8hdoaHdS1a4ZkwFVUb3PG3uazyO7SSdFMiBnmGLITb2j8Chwv8EfpfePB081LPcNmLXl2mfy00GRdU1zenGoxJhhp3G-gda6-afza_w8_-HrbrWogv18N3KP9O8mSlV5GZq6hLxsjw_zaWoM0PtZ1SzzwqWmzPFpsYGfFitgb-6uKiX_I1WO3Z_XXwPaMI_T_XkrVHdqTiA-0tK01djYHvJcag32HGrFKbemzXHngvbEC4hThk2H_yOc4CsWRH30tLGnZOsT4qwNwjjwetuZk4JBfQ31gs3mpwmGtvqwTMC7IwPBM4"
MISSION_ROOT  = "/DOEX0096_Palau/PLW_dscm_09/Video"   # Dropbox folder containing the .mp4s
CSV_PATH      = Path("annotated_detections.csv")
CROPS_DIR     = Path("crops")

PADDING       = 0.05    # fractional padding added around each crop (0.05 = 5% of frame)
MIN_CROP_PX   = 32      # discard crops smaller than this in either dimension


# ============================================================
# LOAD CSV — group detections by video filename
# ============================================================

def load_detections(csv_path: Path) -> dict[str, list[dict]]:
    """Return {media_filename: [detection, ...]} sorted by frame."""
    by_video = defaultdict(list)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            media = row["media"].strip()
            by_video[media].append({
                "frame":   int(row["frame"]),
                "x":       float(row["x"]),
                "y":       float(row["y"]),
                "w":       float(row["width"]),
                "h":       float(row["height"]),
                "species": row["Scientific Name"].strip(),
            })
    for dets in by_video.values():
        dets.sort(key=lambda d: d["frame"])
    return by_video


# ============================================================
# CROP EXTRACTION
# ============================================================

def extract_crops_from_video(
    video_path: Path,
    detections: list[dict],
    crops_dir: Path,
    padding: float = PADDING,
    min_px: int = MIN_CROP_PX,
) -> tuple[int, int]:
    """
    Seek to each detected frame and save the crop.
    Returns (saved, skipped).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Could not open {video_path.name}, skipping")
        return 0, len(detections)

    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    saved = skipped = 0

    # group by frame so we only seek once per frame
    by_frame = defaultdict(list)
    for d in detections:
        by_frame[d["frame"]].append(d)

    for frame_idx in sorted(by_frame.keys()):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            skipped += len(by_frame[frame_idx])
            continue

        for i, det in enumerate(by_frame[frame_idx]):
            species = det["species"]
            if not species:
                skipped += 1
                continue

            # denormalize + add padding
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

            crop = frame[y1:y2, x1:x2]

            out_dir = crops_dir / species
            out_dir.mkdir(parents=True, exist_ok=True)

            stem = video_path.stem
            fname = f"{stem}_frame{frame_idx:05d}_det{i:02d}.jpg"
            cv2.imwrite(str(out_dir / fname), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1

    cap.release()
    return saved, skipped


# ============================================================
# MAIN
# ============================================================

def main():
    CROPS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading detections from {CSV_PATH}...")
    by_video = load_detections(CSV_PATH)
    print(f"Found detections across {len(by_video)} video(s)\n")

    dbx = dropbox.Dropbox(ACCESS_TOKEN)

    total_saved = total_skipped = 0

    with tempfile.TemporaryDirectory(prefix="crop_extraction_") as tmpdir:
        tmpdir = Path(tmpdir)

        for media_name, detections in tqdm(by_video.items(), desc="Videos", unit="video"):
            dropbox_path = MISSION_ROOT + "/" + media_name
            local_path   = tmpdir / media_name

            print(f"\nDownloading {media_name} ({len(detections)} detections)...")
            try:
                dbx.files_download_to_file(str(local_path), dropbox_path)
            except Exception as e:
                print(f"  ✗ Download failed: {e}")
                total_skipped += len(detections)
                continue

            saved, skipped = extract_crops_from_video(local_path, detections, CROPS_DIR)
            total_saved   += saved
            total_skipped += skipped
            print(f"  ✓ {saved} crops saved, {skipped} skipped")

            local_path.unlink(missing_ok=True)

    print(f"\nDone. {total_saved} crops saved to {CROPS_DIR}/")
    print(f"      {total_skipped} detections skipped (bad frame / too small / no label)")

    # summary of what's in each species folder
    print("\nCrop counts by species:")
    for species_dir in sorted(CROPS_DIR.iterdir()):
        if species_dir.is_dir():
            n = len(list(species_dir.iterdir()))
            print(f"  {species_dir.name}: {n}")


main()