"""One-time fix for missions processed before frame numbers counted from the start of
the video. Older pipeline runs wrote each detection's `frame` counted from the first
non-black frame; this shifts them so frame 0 is the first frame of the video file,
matching new runs, the crop file names and Tator.

How many black frames were skipped, per clip, in order of preference:
  1. <mission>_run_log.json  ("black_frames_skipped", recorded by the pipeline)
  2. the crop file names (they always used the true frame number): the shift that
     lines every crop up with a detection of its own track
  3. --video-root: re-run the pipeline's black-frame check on the original video
Clips already marked "frame_origin": "video_start" in the manifest are left alone, so
running this twice is safe. Nothing is written without --apply; the original CSV is
kept next to it as <name>.before_frame_fix.csv.

  python scripts/convert_frame_numbers.py processed_missions/<deployment>/<mission>
  python scripts/convert_frame_numbers.py processed_missions/<deployment>/<mission> --apply
"""
import argparse
import csv
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ORIGIN = "video_start"
CROP_RE = re.compile(r"^id(\d+)_.*_f(\d+)_[^_]+\.jpg$")


def video_stem(media_id: str) -> str:
    return Path(media_id).stem


def offsets_from_run_log(mission_dir: Path, mission: str) -> dict:
    path = mission_dir / f"{mission}_run_log.json"
    if not path.exists():
        return {}
    found = {}
    for run in json.loads(path.read_text()).get("runs", []):          # later runs win
        for v in run.get("videos", []):
            if v.get("status") == "complete" and v.get("black_frames_skipped") is not None:
                found[v["video"]] = int(v["black_frames_skipped"])
    return found


CROP_LABELS = ["early", "mid", "late"]   # same as the pipeline: crop i = best detection of segment i


def offset_from_crops(crop_dir: Path, dets_by_track: dict):
    """Shift implied by the crop file names, which always used the true frame number.

    The pipeline cut each track's detections (in frame order) into len(CROP_LABELS)
    equal segments and cropped the highest-confidence detection of each (the first one
    on a tie). Re-doing that choice from the CSV gives the old-style frame of every crop;
    the crop's file name gives the true frame. All crops must agree on the difference."""
    shifts = set()
    for p in crop_dir.glob("*.jpg") if crop_dir.is_dir() else []:
        m = CROP_RE.match(p.name)
        label = p.stem.rsplit("_", 1)[-1]
        dets = dets_by_track.get(int(m.group(1))) if m else None
        if not dets or label not in CROP_LABELS:
            continue
        dets = sorted(dets)                                   # (frame, conf), frame order
        n, k, i = len(dets), len(CROP_LABELS), CROP_LABELS.index(label)
        segment = dets[i * n // k:(i + 1) * n // k]
        if not segment:
            continue
        best_frame = max(segment, key=lambda d: d[1])[0]
        shifts.add(int(m.group(2)) - best_frame)
    return shifts.pop() if len(shifts) == 1 and min(shifts | {0}) >= 0 else None


def offset_from_video(video_path: Path, threshold=10, require_consecutive=3):
    """Same black-frame check as the pipeline's skip_black_frames (kept in sync by hand so
    this script doesn't need torch / ultralytics installed)."""
    import cv2                                    # only needed for this fallback
    cap = cv2.VideoCapture(str(video_path))
    try:
        consecutive = 0
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                return None
            consecutive = consecutive + 1 if cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean() > threshold else 0
            if consecutive >= require_consecutive:
                return int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - require_consecutive
        return None
    finally:
        cap.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mission_dir", type=Path)
    ap.add_argument("--video-root", type=Path, help="folder with the original videos (last-resort lookup)")
    ap.add_argument("--apply", action="store_true", help="write the changes (default: report only)")
    args = ap.parse_args()

    mdir = args.mission_dir.resolve()
    mission = mdir.name
    csv_path = mdir / f"{mission}_detections.csv"
    manifest_path = mdir / f"{mission}_manifest.json"
    if not csv_path.exists():
        sys.exit(f"No detections CSV at {csv_path}")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        header, rows = reader.fieldnames, list(reader)

    by_video = defaultdict(lambda: defaultdict(list))
    filenames = {}
    for r in rows:
        v = video_stem(r["media_id"])
        filenames.setdefault(v, r["media_id"])
        try:
            by_video[v][int(r["track_id"])].append((int(r["frame"]), float(r["confidence"])))
        except (ValueError, TypeError):
            pass

    from_log = offsets_from_run_log(mdir, mission)
    plan, problems = {}, []
    for v in sorted(by_video):
        if manifest.get(v, {}).get("frame_origin") == ORIGIN:
            continue
        shift, how = from_log.get(v), "run log"
        if shift is None:
            shift, how = offset_from_crops(mdir / f"{mission}_crops" / v, by_video[v]), "crop file names"
        if shift is None and args.video_root:
            path = next(args.video_root.rglob(filenames[v]), None)
            shift, how = (offset_from_video(path) if path else None), "original video"
        if shift is None:
            problems.append(v)
        else:
            plan[v] = (shift, how)

    for v, (shift, how) in plan.items():
        print(f"  {v}: +{shift} frames  (from {how})")
    for v in problems:
        print(f"  {v}: UNKNOWN shift, left unchanged (pass --video-root to look at the original)")
    already = [v for v in by_video if manifest.get(v, {}).get("frame_origin") == ORIGIN]
    print(f"{len(plan)} clip(s) to convert, {len(already)} already converted, {len(problems)} unknown")

    if not args.apply or not plan:
        if plan:
            print("Report only. Re-run with --apply to write.")
        return

    backup = csv_path.with_name(csv_path.stem + ".before_frame_fix.csv")
    if not backup.exists():
        shutil.copy2(csv_path, backup)
    tmp = csv_path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            v = video_stem(r["media_id"])
            if v in plan:
                r = {**r, "frame": str(int(r["frame"]) + plan[v][0])}
            w.writerow(r)
    os.replace(tmp, csv_path)
    for v in plan:
        manifest.setdefault(v, {})["frame_origin"] = ORIGIN
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    os.replace(tmp, manifest_path)
    print(f"Converted. Original CSV kept as {backup.name}. Restart the review app to reload detections.")


if __name__ == "__main__":
    main()
