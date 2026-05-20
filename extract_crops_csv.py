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

ACCESS_TOKEN  = "sl.u.AGe4QPkAbD7xHcc5VQ4qHVl28JzcRGTTMpdgGMRSG02iBZWxdqEO110HIh_AzVnL18bQROZkKqYMTeU5yDGGuRxTt-6Sh9ApGI9XgjojvdBtyvjRMxbRF6K3uRab9-W1RPSLhb-Vw7p7-Dji15MOjE-WsMVmNAukFqMfAv7_Evy71pA4AQPKFMb68KV-NNNRzYE6m4eR2KOqPWPzoz96dIl0lFJbKW5_Mc9j6-hkyz0_Kh5uvSlwtpCJE5FC9mj8lTUvd-oKKf89O7ZLFT5JUy3BYJ-IsvXlXok13qUOVdNk_7zG77WM7eb313eEyMytu2-GHBy21X159NhiKZC8rrsOE_MeTud5Bzk2U0_pKE65owD3D6CHKMSZT1ATjlq8cGnwBTKpPnq62E1sQgvjrYAzPsQsURbvSZElt27pje7UrUIR0GgnWfGR-SDyKlBTo9zwld0NyEZHGphzqs8y-BGEUS_Xl_NALt9Qt1wHhkWqWZdlA8PL1wE1T2i8-a8-ExX_nkJTnYk82YzOcXR3mrh7v8TRYG1I0fcDYzpBB3X643ebjZDFiK6gwXHN7ZaHJH3f5QWPxZAlMtkG7HI1wmy54lwi02xr2g-VDxKGIMFbv0PtWmbbe2JYQ0OVwVA8JgLo8Y4oMz1ammK6GGzqqHVccCjO4_MmpiQQpSqRFMIUymTQmLD2iV_OVcyzcmIFTXpEpxR5FfFofsemRhJOf-EpdiIA2HcdYKiva2Tuo9mmUHqsELK000ym6RlM38y5QTqhaKtQ_Upvur4iin_N-7E1SvgR34Gh-7eH_ie0GxpJWSjijy9GKcDo0GA21VQE2HZe0dm178ka1GJGns66i5wJieUwnO89dnLf2UTlEstVBLAZZNTgt1QAUhqfbII3H6ChY54g9Mol9HXMHNYLPqxwP2S6dp4P2GGtXlquKs3TV7GHIayHFJV6olUr8j2S6jQkuhlFo9WGiWcdLF4GRIwprd7zCsNhb1OPjaMTW6RCo6d6ngCJdiIwOTKvYB__qKozWju7laf01gMoLGkIMHU5vetMsRashVEuBj6H-QMga-FVU_8M2S29ElHkHAp7MBJsQSuX9UiBl7n0Cyk12voPGcd0bdvpEpOGyq8KBQKC66DlOp-MyD5yP_4Mt17FaYGkCLMM62BnGCs5FUS1jP72Hi2s8BF-YENaLT4tA0MCzCg8kbGFoidsDbJQ086GqxT6CpeoPEXw75EdD19f2fz9WLyOhDG7Lb7NogQTCZ7JTuz_ko1ck0LYYVldg3BKgt7qqyxVVDzQg8AmRF3pR8v1KeQjII_tJofpx6-63oF2HGlhj0zdSAC6sG1Wuw23CIMqwLrJmljq5YquLrxN23HokjBB9gALOc-tv2EJ3eD5tJ4ftYG34hFx-BYR3KCO3izNZ32T3CXFPPzlGF9JVEUtm_J8NkOaSDYlzKI0NkzIa_39VTL21Cwzdjcn0GGjxoiVTiB8-C5XCm_rjbyMAlcU"
MISSION_ROOT  = "/DOEX0096_Palau/PLW_dscm_05/Video"   # Dropbox folder containing the .mp4s
CSV_PATH      = Path("mission05_crabs.csv")
CROPS_DIR     = Path("crops")


FORCE_SPECIES = "Chaceon micronesicus" # set to None to use scientific name if provided in csv

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
            # support both column naming conventions
            media = (row.get("media") or row.get("media_id") or "").strip()
            by_video[media].append({
                "frame":   int(row["frame"]),
                "x":       float(row["x"]),
                "y":       float(row["y"]),
                "w":       float(row["width"]),
                "h":       float(row["height"]),
                "species": FORCE_SPECIES or row.get("Scientific Name", "").strip(),
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