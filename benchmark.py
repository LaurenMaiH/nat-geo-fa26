#!/usr/bin/env python3
"""
Speed / accuracy benchmark for overnight_pipeline_optimized.py.

Runs the pipeline on a local folder of videos under several settings and writes a
report (report.md + report.csv) to benchmark_results/<run>/:

    python benchmark.py D:\\PristineSeas\\videos\\bench_clips --footage-hours 100

Each config writes to its own fresh folder, benchmark_results/<run>/<config>/
(detections CSV, crops, embeddings, manifest, run log, pipeline_log.txt), so no
config can skip videos because another one already finished them. Nothing is
written to processed_missions/. Model weights are shared with normal runs
(~/.config/ultralytics and checkpoints/), so nothing is downloaded per config,
except the smaller YOLO-World variants the first time they're used.

Metrics, per config:
  - processing seconds per minute of footage: summed per-video wall time (open,
    resize, detect+track, crop, embed) / footage minutes. One-time setup (model
    loading, downloads) is reported separately and not included.
  - projected hours for --footage-hours of footage, from that rate.
  - tracks: tracks that survive the MIN_TRACK_LEN filter and get an embedding
    (what Cluster Review shows).
  - agreement with baseline (see match_fraction): "baseline matched" = share of
    baseline tracks that some variant track overlaps (IoU > 0.5) in at least one
    frame both detected; "variant matched" = the same the other way round.
"""

import argparse
import csv
import gc
import importlib.util
import io
import json
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

REPO          = Path(__file__).resolve().parent
PIPELINE_FILE = REPO / "overnight_pipeline_optimized.py"
RESULTS_DIR   = REPO / "benchmark_results"
MISSION_ID    = "bench"
IOU_MATCH     = 0.5

sys.path.insert(0, str(REPO / "application"))
from video_sources import LocalSource, video_stem  # noqa: E402

# name → (description, SpeedOptions fields, pipeline settings)
CONFIGS = {
    "baseline":       ("Current defaults", {}, {}),
    "every2":         ("YOLO + tracking on every 2nd frame", {"detect_every": 2}, {}),
    "every3":         ("YOLO + tracking on every 3rd frame", {"detect_every": 3}, {}),
    "yolo_l":         ("YOLO-World large (yolov8l-worldv2)", {"yolo_weights": "yolov8l-worldv2.pt"}, {}),
    "yolo_m":         ("YOLO-World medium (yolov8m-worldv2)", {"yolo_weights": "yolov8m-worldv2.pt"}, {}),
    "no_resize":      ("No pre-resize (work at source resolution)", {}, {"PRERESIZE": 0}),
    "resize_720":     ("Pre-resize to 720p", {}, {"PRERESIZE": 720}),
    "imgsz_640":      ("Inference size 640 (explicit)", {"imgsz": 640}, {}),
    "imgsz_960":      ("Inference size 960", {"imgsz": 960}, {}),
    "imgsz_1280":     ("Inference size 1280", {"imgsz": 1280}, {}),
    "orig_crops":     ("Crops cut from original-resolution frames", {"crop_from_original": True}, {}),
    "baseline_again": ("Current defaults, run again (run-to-run noise)", {}, {}),
}
DETECTION_UNCHANGED = {"orig_crops"}   # same detections as baseline: report runtime + crop size only


def load_pipeline():
    """A freshly imported pipeline module: no cached models, no tracker state."""
    spec = importlib.util.spec_from_file_location(f"pipeline_bench_{time.monotonic_ns()}", PIPELINE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def free_memory():
    """Call after dropping the last reference to a pipeline module and its models."""
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


# ── Preflight ─────────────────────────────────────────────────────────────────

def source_sizes(video_dir, videos):
    import cv2
    sizes = set()
    for v in videos:
        cap = cv2.VideoCapture(str(video_dir / v.rel_path))
        sizes.add((int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        cap.release()
    return sizes


def processed_sizes(sizes, preresize):
    """Frame sizes detection will see: sources taller than PRERESIZE are scaled down to it."""
    out = set()
    for w, h in sizes:
        if preresize and h > preresize:
            w, h = round(w * preresize / h / 2) * 2, preresize
        out.add((w, h))
    return out


def warm_detector(model, device, sizes, imgsz=None, track=False):
    """Run the detector on blank frames of each size, so first-time GPU kernel setup for
    those input shapes isn't charged to the timed run."""
    import numpy as np
    kwargs = {"imgsz": imgsz} if imgsz else {}
    for w, h in sizes:
        frame = np.full((h, w, 3), 90, dtype=np.uint8)
        for _ in range(3):
            if track:
                model.track(frame, persist=True, device=device, verbose=False, **kwargs)
            else:
                model.predict(frame, device=device, verbose=False, **kwargs)


def preflight(device, sizes):
    """Abort unless the fine-tuned BioCLIP weights load; warm the GPU up (tracker,
    embedding batch sizes) so the first config isn't charged for start-up costs."""
    import numpy as np
    mod = load_pipeline()
    mod.INFERENCE_DEVICE = mod.resolve_device(device)
    weights, emb_model, _ = info = mod.load_embedding_model()
    if weights != "finetuned":
        sys.exit(f"ABORTING: the embedding model loaded '{weights}' weights, not the fine-tuned "
                 f"checkpoint ({mod.CHECKPOINT_PATH}). Benchmark numbers would not reflect real runs. "
                 "See the messages above (download failed? corrupt best.pt?).")
    model, _ = mod.load_model_and_config()
    warm_detector(model, mod.INFERENCE_DEVICE, sizes, track=True)   # this model is thrown away
    crop = np.full((200, 200, 3), 90, dtype=np.uint8)
    for n in range(1, mod.CROPS_PER_TRACK + 1):
        mod.get_embeddings_batch(info, [crop] * n)
    resolved = mod.INFERENCE_DEVICE
    del model, emb_model, info, mod
    free_memory()
    return resolved


# ── One config ────────────────────────────────────────────────────────────────

def run_config(name, video_dir, out_dir, device, sizes):
    _, opt_fields, settings = CONFIGS[name]
    out_dir.mkdir(parents=True)
    mod = load_pipeline()
    mod.MISSION_ID, mod.OUTPUT_DIR = MISSION_ID, out_dir
    mod.SOURCE, mod.MISSION_PATH = LocalSource(str(video_dir)), "."
    mod.START_FROM, mod.SELECTED_VIDEOS = None, None
    mod.INFERENCE_DEVICE = device
    for k, v in settings.items():
        setattr(mod, k, v)
    options = mod.SpeedOptions(**opt_fields)
    error, yolo_load_s, downloads = None, None, []
    with open(out_dir / "pipeline_log.txt", "w", encoding="utf-8") as log:
        orig, sys.stdout = sys.stdout, Tee(sys.stdout, log)
        started = time.perf_counter()
        try:
            # Load + warm this config's detector before timing starts, so a short sample
            # isn't dominated by its one-off predictor setup. predict() (not track())
            # leaves no tracker state; main() then reuses the cached model.
            t0 = time.perf_counter()
            model, _ = mod.load_model_and_config(options.yolo_weights)
            downloads = list(mod._downloads)          # main() starts a fresh list
            yolo_load_s = time.perf_counter() - t0 - sum(d["seconds"] for d in downloads)
            warm_detector(model, device, processed_sizes(sizes, mod.PRERESIZE), options.imgsz)
            del model
            mod.main(options=options)
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            print(f"CONFIG FAILED: {error}")
        finally:
            sys.stdout = orig
    wall = time.perf_counter() - started
    downloads += mod._downloads                   # e.g. best.pt, if it was missing
    del mod
    free_memory()
    return error, wall, yolo_load_s, downloads


# ── Analysis ──────────────────────────────────────────────────────────────────

def read_outputs(out_dir):
    """(last run-log entry, {(video, track_id): {frame: (x, y, w, h)}} for kept tracks)."""
    log_path = out_dir / f"{MISSION_ID}_run_log.json"
    run = json.loads(log_path.read_text())["runs"][-1] if log_path.exists() else None
    emb_path = out_dir / f"{MISSION_ID}_embeddings.json"
    kept = {tuple(k.rsplit("__", 1)) for k in json.loads(emb_path.read_text())} if emb_path.exists() else set()
    tracks = defaultdict(dict)
    csv_path = out_dir / f"{MISSION_ID}_detections.csv"
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                key = (video_stem(row["media_id"]), row["track_id"])
                if key in kept:
                    tracks[key][int(row["frame"])] = tuple(float(row[c]) for c in ("x", "y", "width", "height"))
    return run, dict(tracks)


def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def match_fraction(ref, other, threshold=IOU_MATCH):
    """Share of `ref` tracks matched by some `other` track in the same video: at least
    one frame where both have a box with IoU > threshold. Boxes are fractions of the
    frame, so configs with different resolutions compare directly. Returns (matched, total)."""
    by_video = defaultdict(list)
    for (video, _), frames in other.items():
        by_video[video].append(frames)
    matched = 0
    for (video, _), frames in ref.items():
        if any(iou(box, cand[f]) > threshold
               for cand in by_video[video] for f, box in frames.items() if f in cand):
            matched += 1
    return matched, len(ref)


def summarize(run):
    videos = [v for v in run["videos"] if v["status"] == "complete"]
    footage = sum(v.get("duration_s") or 0 for v in videos)
    proc = sum(v["total_s"] for v in videos)
    stages = defaultdict(float)
    for v in videos:
        for k, s in v["stages_s"].items():
            stages[k] += s
    n_crops = sum(v.get("n_crops", 0) for v in videos)
    crop_px = None
    if n_crops:
        crop_px = [sum(v["crop_px_mean"][i] * v["n_crops"] for v in videos if v.get("n_crops")) / n_crops
                   for i in (0, 1)]
    frames_total = sum(v.get("frames_total", 0) for v in videos)
    return {
        "videos": len(videos), "failed": len(run["videos"]) - len(videos),
        "footage_s": footage, "processing_s": proc,
        "s_per_min": proc / (footage / 60) if footage else None,
        "stages_per_min": {k: s / (footage / 60) for k, s in stages.items()} if footage else {},
        "frames_total": frames_total,
        "gate_rate": sum(v.get("frames_motion_active", 0) for v in videos) / frames_total if frames_total else None,
        "inferred_rate": sum(v.get("frames_inferred", 0) for v in videos) / frames_total if frames_total else None,
        "tracks": sum(v.get("n_tracks", 0) for v in videos),
        "crop_px": crop_px,
        "resize": sorted({v.get("resize", "?") for v in videos}),
        "frame_mismatch": [v["video"] for v in videos if v.get("frame_count_mismatch")],
        "setup": run["setup"], "settings": run["settings"],
        "resolutions": sorted({v.get("source_resolution") for v in videos}),
        "processed_res": sorted({v.get("processed_resolution") for v in videos}),
    }


def fmt(x, nd=1, pct=False):
    if x is None:
        return "–"
    return f"{x * 100:.{nd}f}%" if pct else f"{x:.{nd}f}"


def write_report(run_dir, results, footage_hours, illustrative, video_dir, wall_total):
    base = results.get("baseline")
    if not base or base.get("error"):
        sys.exit("Baseline failed; no report possible. See benchmark_results/.../baseline/pipeline_log.txt")
    b, b_tracks = base["summary"], base["tracks"]
    run0 = base["run"]
    m = run0["machine"]
    gpu = (m.get("gpu") or {}).get("name", "none")

    rows = []
    for name, r in results.items():
        desc = CONFIGS[name][0]
        if r.get("error"):
            rows.append({"config": name, "description": desc, "error": r["error"]})
            continue
        s = r["summary"]
        row = {
            "config": name, "description": desc, "videos": s["videos"], "failed_videos": s["failed"],
            "s_per_footage_min": s["s_per_min"],
            "projected_hours": footage_hours * s["s_per_min"] / 60 if s["s_per_min"] else None,
            "speedup_vs_baseline": b["s_per_min"] / s["s_per_min"] if s["s_per_min"] else None,
            "tracks": s["tracks"],
            "motion_gate_hit_rate": s["gate_rate"], "frames_inferred_rate": s["inferred_rate"],
            "crop_px_w": s["crop_px"][0] if s["crop_px"] else None,
            "crop_px_h": s["crop_px"][1] if s["crop_px"] else None,
            "processed_resolution": " / ".join(s["processed_res"]),
            "resize": " / ".join(s["resize"]),
            "imgsz": s["settings"].get("imgsz_effective"),
            **{f"stage_{k}_s_per_min": v for k, v in s["stages_per_min"].items()},
            "yolo_load_s": round(r["yolo_load_s"], 1) if r["yolo_load_s"] is not None else None,
            "embedding_load_s": s["setup"]["embedding_model_load_s"],
            "downloads": "; ".join(f"{d['file']} {d['bytes'] / 1e6:.0f} MB in {d['seconds']} s"
                                   for d in r["downloads"]),
            "embedding_weights": r["run"]["embedding_weights"],
            "error": "",
        }
        if name not in DETECTION_UNCHANGED and name != "baseline":
            mb, nb = match_fraction(b_tracks, r["tracks"])
            mv, nv = match_fraction(r["tracks"], b_tracks)
            row["baseline_tracks_matched"] = mb / nb if nb else None
            row["variant_tracks_matched"] = mv / nv if nv else None
        rows.append(row)

    # CSV
    fields = []
    for row in rows:
        fields += [k for k in row if k not in fields]
    with open(run_dir / "report.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})

    # Markdown
    L = []
    title = "Pipeline benchmark" + (" (ILLUSTRATIVE: short sample clips, not real footage)" if illustrative else "")
    L += [f"# {title}", "",
          f"- **Date:** {datetime.now():%Y-%m-%d %H:%M}",
          f"- **Embedding weights: {run0['embedding_weights'].replace('finetuned', 'fine-tuned')}** "
          "(checked before any config ran, and in every config's run log)",
          f"- **Machine:** {m['cpu']}, {m['cores_physical']} cores / {m['cores_logical']} threads, "
          f"{m['ram_gb']} GB RAM, GPU: {gpu}",
          f"- **Device:** {m['device']} · torch {m['torch']} · ultralytics {m['ultralytics']} · {m['os']}",
          f"- **Footage:** {b['videos']} video(s) from `{video_dir}`, {b['footage_s'] / 60:.1f} min total, "
          f"source resolution {', '.join(b['resolutions'])}",
          "- **Videos:** " + "; ".join(
              f"{v['filename']} {v.get('source_resolution')} {v.get('duration_s')} s, "
              f"baseline resize {v.get('resize')}" for v in run0["videos"]),
          f"- **Projection:** {footage_hours:g} hours of footage · benchmark wall time {wall_total / 60:.0f} min",
          ""]
    L += ["## Results", "",
          f"Processing time excludes one-time setup (below). Projected hours = time to process "
          f"{footage_hours:g} h of footage like this sample.", "",
          "| Config | What changes | s / footage min | Projected h | Speed-up | Tracks | Baseline tracks matched | Variant tracks matched |",
          "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        if row.get("error"):
            L.append(f"| {row['config']} | {row['description']} | FAILED: {row['error']} | | | | | |")
            continue
        if row["config"] in DETECTION_UNCHANGED:
            agree = "same detections", "same detections"
        elif row["config"] == "baseline":
            agree = "–", "–"
        else:
            agree = fmt(row["baseline_tracks_matched"], 0, True), fmt(row["variant_tracks_matched"], 0, True)
        L.append(f"| {row['config']} | {row['description']} | {fmt(row['s_per_footage_min'])} | "
                 f"{fmt(row['projected_hours'])} | {fmt(row['speedup_vs_baseline'], 2)}× | {row['tracks']} | "
                 f"{agree[0]} | {agree[1]} |")

    stage_names = ["open", "resize", "detect_track", "crop", "embed"]
    L += ["", "## Where the time goes", "",
          "Seconds per minute of footage, by stage. *detect_track* includes decoding, black-frame trim, "
          "the motion gate and YOLO-World + ByteTrack.", "",
          "| Config | " + " | ".join(stage_names) + " | Motion gate hit rate | Frames sent to YOLO | Processed at | Resize | YOLO imgsz |",
          "|---|" + "---:|" * len(stage_names) + "---:|---:|---|---|---:|"]
    for row in rows:
        if row.get("error"):
            continue
        L.append(f"| {row['config']} | " + " | ".join(fmt(row.get(f"stage_{k}_s_per_min"), 2) for k in stage_names)
                 + f" | {fmt(row['motion_gate_hit_rate'], 0, True)} | {fmt(row['frames_inferred_rate'], 0, True)}"
                 f" | {row['processed_resolution']} | {row['resize']} | {row['imgsz']} |")

    L += ["", "## One-time setup costs (not included above)", "",
          "| Config | YOLO-World load (s) | BioCLIP load (s) | Downloads |", "|---|---:|---:|---|"]
    for row in rows:
        if not row.get("error"):
            L.append(f"| {row['config']} | {row['yolo_load_s']} | {row['embedding_load_s']} | {row['downloads'] or '–'} |")

    notes = []
    for name, r in results.items():
        if r.get("error"):
            if name in ("yolo_l", "yolo_m"):
                notes.append(f"**{name}**: the weights could not be used ({r['error']}). "
                             "They may not download by name from the Ultralytics release assets.")
            else:
                notes.append(f"**{name}** failed: {r['error']}")
            continue
        s = r["summary"]
        if "PRERESIZE" in CONFIGS[name][2]:
            skipped = [v["filename"] for v in r["run"]["videos"] if str(v.get("resize", "")).startswith("skipped")]
            if skipped:
                notes.append(f"**{name}**: resize skipped for {', '.join(skipped)} because the source was "
                             "already at or below the target height, so those videos ran exactly as in baseline.")
        if s["frame_mismatch"]:
            notes.append(f"**{name}**: resized copy and original had different frame counts for "
                         f"{', '.join(s['frame_mismatch'])}; original-resolution crops may be a few frames off.")
        if s["failed"]:
            notes.append(f"**{name}**: {s['failed']} video(s) failed; see its pipeline_log.txt.")
        for d in r["downloads"]:
            notes.append(f"**{name}**: downloaded {d['file']} ({d['bytes'] / 1e6:.0f} MB, {d['seconds']} s) on first use.")
    if "orig_crops" in results and not results["orig_crops"].get("error") and b["crop_px"]:
        o = results["orig_crops"]["summary"]
        if o["crop_px"]:
            notes.append(f"**orig_crops**: mean crop {o['crop_px'][0]:.0f}×{o['crop_px'][1]:.0f} px vs "
                         f"{b['crop_px'][0]:.0f}×{b['crop_px'][1]:.0f} px for baseline. Detections are identical "
                         "to baseline; its effect on embedding quality needs a separate check.")
    if "baseline_again" in results and not results["baseline_again"].get("error"):
        a = results["baseline_again"]["summary"]
        notes.append(f"**baseline_again** differs from baseline by "
                     f"{abs(a['s_per_min'] / b['s_per_min'] - 1) * 100:.0f}% in time; differences smaller than "
                     "that are within run-to-run noise.")
    if notes:
        L += ["", "## Notes", ""] + [f"- {n}" for n in notes]

    L += ["", "## Method", "",
          "- Each config ran in a freshly imported pipeline (no cached models or tracker state) with its own "
          "empty output folder, so resume never skipped a video. Model weights were shared.",
          "- Times are wall-clock per video from the pipeline's run log (`bench_run_log.json` in each config "
          "folder). GPU warm-up ran once before the first config, and each config's detector was loaded "
          "and warmed (inference only, no tracking) before its timing started.",
          f"- Tracks = tracks kept after the MIN_TRACK_LEN filter that got an embedding. A baseline track counts "
          f"as matched if some variant track in the same video has a box with IoU > {IOU_MATCH} on at least one "
          "frame where both have a box (and the reverse for variant tracks). Boxes are fractions of the frame, "
          "so resolutions compare directly. This is deliberately simple: one overlapping frame is enough, and "
          "it says nothing about species/embedding quality.",
          "- Detecting on every Nth frame gives tracks fewer detections, so more of them fall below "
          "MIN_TRACK_LEN; the motion gate's MIN_BLOB is in pixels, so resolution changes also change which "
          "frames are sent to YOLO."]
    (run_dir / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("video_dir", type=Path, help="local folder with the videos to benchmark on")
    p.add_argument("--footage-hours", type=float, default=100,
                   help="hours of footage to project processing time for (default 100)")
    p.add_argument("--configs", default=",".join(CONFIGS),
                   help=f"comma-separated subset of: {', '.join(CONFIGS)} (baseline is always run first)")
    p.add_argument("--name", help="results folder name (default: a timestamp)")
    p.add_argument("--device", help="cpu | cuda | mps (default: INFERENCE_DEVICE or auto)")
    p.add_argument("--illustrative", action="store_true",
                   help="label the report as illustrative (sample clips, not real footage)")
    args = p.parse_args()
    for stream in (sys.stdout, sys.stderr):   # the pipeline prints ✓ / ✗; don't crash on cp1252 consoles
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    video_dir = args.video_dir.resolve()
    videos = LocalSource(str(video_dir)).list_videos(".")
    if not videos:
        sys.exit(f"No videos (.mp4 / .mov) found in {video_dir}")
    names = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = [c for c in names if c not in CONFIGS]
    if unknown:
        sys.exit(f"Unknown config(s): {', '.join(unknown)}")
    names = ["baseline"] + [c for c in names if c != "baseline"]

    run_dir = RESULTS_DIR / (args.name or datetime.now().strftime("%Y%m%d_%H%M%S"))
    if run_dir.exists():
        sys.exit(f"{run_dir} already exists; pick another --name (configs must start from empty folders)")
    run_dir.mkdir(parents=True)
    print(f"Benchmarking {len(videos)} video(s) from {video_dir}\nConfigs: {', '.join(names)}\nResults: {run_dir}\n")

    started = time.perf_counter()
    sizes = source_sizes(video_dir, videos)
    device = preflight(args.device, sizes | processed_sizes(sizes, 1080))
    print(f"\nPreflight OK: fine-tuned weights load; device {device}\n")

    results = {}
    for name in names:
        print(f"\n{'#' * 70}\n# {name}: {CONFIGS[name][0]}\n{'#' * 70}")
        out = run_dir / name
        error, wall, yolo_load_s, downloads = run_config(name, video_dir, out, device, sizes)
        result = {"error": error, "wall_s": wall, "yolo_load_s": yolo_load_s, "downloads": downloads}
        if not error:
            run, tracks = read_outputs(out)
            if run is None:
                result["error"] = "no run log written (no videos processed?)"
            elif run["embedding_weights"] != "finetuned":
                shutil.rmtree(out, ignore_errors=True)
                sys.exit(f"ABORTING: config {name} ran with '{run['embedding_weights']}' embedding weights.")
            else:
                result.update(run=run, tracks=tracks, summary=summarize(run))
        if name == "baseline" and result.get("error"):
            sys.exit(f"Baseline failed ({result['error']}); see {out / 'pipeline_log.txt'}")
        results[name] = result

    write_report(run_dir, results, args.footage_hours, args.illustrative, video_dir,
                 time.perf_counter() - started)
    print(f"\nReport: {run_dir / 'report.md'}\n        {run_dir / 'report.csv'}")


if __name__ == "__main__":
    main()
