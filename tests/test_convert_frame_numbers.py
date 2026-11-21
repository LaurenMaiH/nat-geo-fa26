import csv
import json
import subprocess
import sys

from conftest import DETECTIONS_HEADER, REPO

SCRIPT = REPO / "scripts" / "convert_frame_numbers.py"


def _mission(tmp_path, run_log_videos):
    m = tmp_path / "M1"
    (m / "M1_crops" / "b").mkdir(parents=True)
    # a: 6 black frames (in the run log) · b: 10 black frames (only crop names know) · c: done already
    (m / "M1_detections.csv").write_text(DETECTIONS_HEADER + "".join(
        f"{v}.mp4,{f},1,fish,0,0,0.1,0.1,{conf}\n"
        for v in ("a", "b", "c") for f, conf in [(0, 0.5), (1, 0.9), (2, 0.6), (3, 0.7), (4, 0.95), (5, 0.4)]))
    for name in ("id0001_fish_f00011_early.jpg", "id0001_fish_f00013_mid.jpg", "id0001_fish_f00014_late.jpg"):
        (m / "M1_crops" / "b" / name).write_bytes(b"jpg")
    (m / "M1_manifest.json").write_text(json.dumps({
        "a": {"status": "complete"}, "b": {"status": "complete"},
        "c": {"status": "complete", "frame_origin": "video_start"}}))
    (m / "M1_run_log.json").write_text(json.dumps({"runs": [{"videos": run_log_videos}]}))
    return m


def _frames(m, video):
    with open(m / "M1_detections.csv", newline="") as f:
        return [int(r["frame"]) for r in csv.DictReader(f) if r["media_id"] == f"{video}.mp4"]


def test_report_only_changes_nothing(tmp_path):
    m = _mission(tmp_path, [{"video": "a", "status": "complete", "black_frames_skipped": 6}])
    before = (m / "M1_detections.csv").read_text()
    out = subprocess.run([sys.executable, str(SCRIPT), str(m)], capture_output=True, text=True).stdout
    assert "a: +6" in out and "b: +10" in out and "--apply" in out
    assert (m / "M1_detections.csv").read_text() == before


def test_apply_shifts_old_clips_once(tmp_path):
    m = _mission(tmp_path, [{"video": "a", "status": "complete", "black_frames_skipped": 6}])
    for _ in range(2):                                   # second run must be a no-op
        subprocess.run([sys.executable, str(SCRIPT), str(m), "--apply"], check=True, capture_output=True)
    assert _frames(m, "a") == [6, 7, 8, 9, 10, 11]
    assert _frames(m, "b") == [10, 11, 12, 13, 14, 15]   # crops at 11/13/14 = best of each third
    assert _frames(m, "c") == [0, 1, 2, 3, 4, 5]         # already counted from the start
    assert (m / "M1_detections.before_frame_fix.csv").exists()
    marks = {v: e.get("frame_origin") for v, e in json.loads((m / "M1_manifest.json").read_text()).items()}
    assert marks == {"a": "video_start", "b": "video_start", "c": "video_start"}


def test_unknown_shift_is_left_alone(tmp_path):
    m = _mission(tmp_path, [])
    for p in (m / "M1_crops" / "b").iterdir():
        p.unlink()
    out = subprocess.run([sys.executable, str(SCRIPT), str(m), "--apply"], capture_output=True, text=True).stdout
    assert "UNKNOWN" in out
    assert _frames(m, "a") == [0, 1, 2, 3, 4, 5]
