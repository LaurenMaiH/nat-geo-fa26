#!/usr/bin/env python3
"""
cluster_review.py — Video-by-video incremental clustering with human-in-the-loop review.

Given:
  - overnight_crops/          (directory of per-video crop folders)
  - overnight_embeddings.json (keyed as "{video_name}__{track_id}")

Workflow:
  1. Load embeddings JSON
  2. For each video (in order), run HDBSCAN on that video's track embeddings
  3. Match resulting clusters → existing global clusters by centroid distance
  4. Propose: "assign to existing cluster X" or "create new cluster"
  5. Serve a web UI so a human can review, drag-drop, accept/reject
  6. On confirmation, update global cluster state and move to the next video
  7. Save final cluster assignments to cluster_results.json

Install:
    pip install flask hdbscan scikit-learn numpy
"""

import json
import base64
import os
from pathlib import Path
from collections import defaultdict
from copy import deepcopy

import numpy as np
from flask import Flask, jsonify, request, send_from_directory, send_file
from sklearn.metrics.pairwise import cosine_distances

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    print("WARNING: hdbscan not installed. Run: pip install hdbscan")
    HDBSCAN_AVAILABLE = False

# ============================================================
# CONFIG
# ============================================================

_MISSION_DIR     = Path(__file__).parent.parent / "processed_missions" / "mission_03"
CROPS_DIR        = _MISSION_DIR
EMBEDDINGS_PATH  = _MISSION_DIR / "mission_03_embeddings.json"
RESULTS_PATH     = _MISSION_DIR / "cluster_results.json"

# HDBSCAN params
MIN_CLUSTER_SIZE = 2   # minimum tracks per cluster (small since tracks are already pooled)
MIN_SAMPLES      = 1   # controls noise sensitivity

# Matching threshold — cosine distance below this → assign to existing cluster
# Above this → propose as new cluster
MATCH_THRESHOLD  = 0.35

HOST = "127.0.0.1"
PORT = 5050

# ============================================================
# GLOBALS (in-memory state, persisted to RESULTS_PATH on confirm)
# ============================================================

# global_clusters: dict[cluster_id: str, {
#     "label": str,           # human-assigned label, defaults to cluster_id
#     "centroid": list[float],
#     "members": list[str],   # "{video_name}__{tid}" keys
# }]
global_clusters: dict = {}
next_cluster_id: int  = 0

# video queue
video_queue:     list = []   # ordered list of video names with crops
current_video_idx: int = 0

# pending proposal for the current video, set by /api/propose
# proposal: {
#   "video": str,
#   "assignments": [
#     {
#       "track_key": str,              # "{video}__{tid}"
#       "proposed_cluster": str|None,  # existing cluster_id, or None = new
#       "is_new": bool,
#       "distance": float|None,
#       "crop_paths": list[str],
#     }, ...
#   ]
# }
current_proposal: dict = {}

embeddings: dict = {}   # full embeddings dict loaded at startup


# ============================================================
# HELPERS
# ============================================================

def load_embeddings() -> dict:
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(f"Embeddings not found at {EMBEDDINGS_PATH}")
    with open(EMBEDDINGS_PATH) as f:
        return json.load(f)


def get_video_order() -> list:
    """Return sorted list of video names that have both crops and embeddings."""
    if not CROPS_DIR.exists():
        return []
    videos = sorted(d.name for d in CROPS_DIR.iterdir() if d.is_dir())
    # only keep videos that have at least one embedding
    emb_videos = set(k.split("__")[0] for k in embeddings)
    return [v for v in videos if v in emb_videos]


def get_video_embeddings(video_name: str) -> dict:
    """Return {track_key: np.array} for a single video."""
    prefix = f"{video_name}__"
    return {
        k: np.array(v)
        for k, v in embeddings.items()
        if k.startswith(prefix)
    }


def get_crop_paths(track_key: str) -> list:
    """Return list of crop image paths for a track key."""
    video_name, tid = track_key.split("__")
    video_dir = CROPS_DIR / video_name
    if not video_dir.exists():
        return []
    tid_int = int(tid)
    pattern = f"id{tid_int:04d}_"
    return sorted(str(p) for p in video_dir.iterdir() if p.name.startswith(pattern))


def cluster_video(video_embs: dict) -> dict:
    """
    Run HDBSCAN on a video's track embeddings.
    Returns {local_label: [track_key, ...]}
    Noise points (label=-1) get their own singleton pseudo-clusters.
    """
    if not video_embs:
        return {}

    keys = list(video_embs.keys())
    vecs = np.stack([video_embs[k] for k in keys])

    if len(keys) == 1:
        return {"0": keys}

    if not HDBSCAN_AVAILABLE:
        # fallback: treat each track as its own cluster
        return {str(i): [k] for i, k in enumerate(keys)}

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(vecs)

    clusters = defaultdict(list)
    noise_idx = 0
    for key, label in zip(keys, labels):
        if label == -1:
            clusters[f"noise_{noise_idx}"].append(key)
            noise_idx += 1
        else:
            clusters[str(label)].append(key)

    return dict(clusters)


def centroid_of(track_keys: list) -> np.ndarray:
    vecs = [np.array(embeddings[k]) for k in track_keys if k in embeddings]
    if not vecs:
        return np.zeros(768)
    c = np.mean(vecs, axis=0)
    c /= (np.linalg.norm(c) + 1e-8)
    return c


def match_to_global(new_centroid: np.ndarray) -> tuple:
    """
    Find closest existing global cluster by cosine distance.
    Returns (cluster_id, distance) or (None, None) if no global clusters exist.
    """
    if not global_clusters:
        return None, None

    best_id   = None
    best_dist = float("inf")
    for cid, info in global_clusters.items():
        existing = np.array(info["centroid"])
        dist = float(cosine_distances([new_centroid], [existing])[0][0])
        if dist < best_dist:
            best_dist = dist
            best_id   = cid

    return best_id, best_dist


def new_cluster_id() -> str:
    global next_cluster_id
    cid = f"cluster_{next_cluster_id:03d}"
    next_cluster_id += 1
    return cid


def save_results():
    out = {
        "clusters": global_clusters,
        "videos_processed": video_queue[:current_video_idx],
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(out, f, indent=2)


def load_results():
    """Resume from a previous session if results file exists."""
    global global_clusters, next_cluster_id, current_video_idx
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            data = json.load(f)
        global_clusters = data.get("clusters", {})
        processed = data.get("videos_processed", [])
        # advance queue past already-processed videos
        for i, v in enumerate(video_queue):
            if v not in processed:
                current_video_idx = i
                break
        else:
            current_video_idx = len(video_queue)
        # recompute next_cluster_id
        ids = [int(k.split("_")[1]) for k in global_clusters if k.startswith("cluster_")]
        next_cluster_id = max(ids) + 1 if ids else 0
        print(f"Resumed: {len(global_clusters)} existing clusters, "
              f"{current_video_idx}/{len(video_queue)} videos processed")


def image_to_b64(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return ""


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__, static_folder="cluster_ui", static_url_path="")


@app.route("/")
def index():
    return send_file("cluster_ui/index.html")


@app.route("/api/status")
def api_status():
    return jsonify({
        "total_videos": len(video_queue),
        "current_video_idx": current_video_idx,
        "current_video": video_queue[current_video_idx] if current_video_idx < len(video_queue) else None,
        "done": current_video_idx >= len(video_queue),
        "n_clusters": len(global_clusters),
        "cluster_labels": {cid: info["label"] for cid, info in global_clusters.items()},
    })


@app.route("/api/clusters")
def api_clusters():
    """Return existing global clusters with representative crop images."""
    out = {}
    for cid, info in global_clusters.items():
        # pick up to 3 representative crop images
        rep_crops = []
        for tk in info["members"][:6]:
            paths = get_crop_paths(tk)
            if paths:
                rep_crops.append({
                    "track_key": tk,
                    "b64": image_to_b64(paths[0]),
                    "path": paths[0],
                })
            if len(rep_crops) >= 3:
                break
        out[cid] = {
            "label": info["label"],
            "n_members": len(info["members"]),
            "rep_crops": rep_crops,
        }
    return jsonify(out)


@app.route("/api/propose")
def api_propose():
    """
    Cluster the current video and propose assignments.
    Returns the proposal without modifying global state.
    """
    global current_proposal

    if current_video_idx >= len(video_queue):
        return jsonify({"done": True})

    video_name = video_queue[current_video_idx]
    video_embs = get_video_embeddings(video_name)

    if not video_embs:
        current_proposal = {"video": video_name, "assignments": []}
        return jsonify(current_proposal)

    local_clusters = cluster_video(video_embs)

    assignments = []
    for local_label, track_keys in local_clusters.items():
        new_centroid = centroid_of(track_keys)
        best_id, best_dist = match_to_global(new_centroid)

        is_new = (best_id is None) or (best_dist > MATCH_THRESHOLD)

        # collect crop images for all tracks in this local cluster
        crops = []
        for tk in track_keys:
            paths = get_crop_paths(tk)
            for p in paths[:2]:  # max 2 crops per track to keep payload light
                crops.append({
                    "track_key": tk,
                    "b64": image_to_b64(p),
                    "path": p,
                })

        assignments.append({
            "local_label":        local_label,
            "track_keys":         track_keys,
            "proposed_cluster":   None if is_new else best_id,
            "proposed_label":     None if is_new else global_clusters[best_id]["label"],
            "is_new":             is_new,
            "distance":           round(best_dist, 4) if best_dist is not None else None,
            "crops":              crops,
            "new_centroid":       new_centroid.tolist(),
        })

    current_proposal = {"video": video_name, "assignments": assignments}
    return jsonify(current_proposal)


@app.route("/api/confirm", methods=["POST"])
def api_confirm():
    """
    Accept the (possibly edited) proposal and apply it to global state.
    Body: { assignments: [ { track_keys, proposed_cluster, is_new, new_label? }, ... ] }
    """
    global current_video_idx

    data = request.get_json()
    confirmed = data.get("assignments", [])

    for item in confirmed:
        track_keys = item["track_keys"]
        is_new     = item["is_new"]
        cid        = item.get("proposed_cluster")
        new_label  = item.get("new_label", "")

        if is_new or cid is None:
            cid = new_cluster_id()
            label = new_label or cid
            global_clusters[cid] = {
                "label":    label,
                "centroid": item["new_centroid"],
                "members":  [],
            }
        else:
            label = global_clusters[cid]["label"]

        global_clusters[cid]["members"].extend(track_keys)
        # update centroid as running mean
        global_clusters[cid]["centroid"] = centroid_of(
            global_clusters[cid]["members"]
        ).tolist()

    current_video_idx += 1
    save_results()

    return jsonify({
        "ok": True,
        "n_clusters": len(global_clusters),
        "videos_remaining": len(video_queue) - current_video_idx,
    })


@app.route("/api/skip", methods=["POST"])
def api_skip():
    """Skip the current video without making any cluster assignments."""
    global current_video_idx
    current_video_idx += 1
    save_results()
    return jsonify({"ok": True, "skipped": True})


@app.route("/api/relabel", methods=["POST"])
def api_relabel():
    """Rename an existing cluster."""
    data = request.get_json()
    cid  = data.get("cluster_id")
    label = data.get("label", "")
    if cid in global_clusters:
        global_clusters[cid]["label"] = label
        save_results()
        return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "cluster not found"}), 404


@app.route("/api/merge", methods=["POST"])
def api_merge():
    """Merge two existing clusters."""
    data = request.get_json()
    src  = data.get("src")
    dst  = data.get("dst")
    if src not in global_clusters or dst not in global_clusters:
        return jsonify({"ok": False, "error": "cluster not found"}), 404
    global_clusters[dst]["members"].extend(global_clusters[src]["members"])
    global_clusters[dst]["centroid"] = centroid_of(global_clusters[dst]["members"]).tolist()
    del global_clusters[src]
    save_results()
    return jsonify({"ok": True})


# ============================================================
# STARTUP
# ============================================================

def main():
    global embeddings, video_queue

    print("Loading embeddings...")
    embeddings = load_embeddings()
    print(f"  {len(embeddings)} track embeddings loaded")

    video_queue = get_video_order()
    print(f"  {len(video_queue)} videos with crops found")

    load_results()

    # make sure UI dir exists
    (Path(__file__).parent / "cluster_ui").mkdir(exist_ok=True)

    print(f"\nStarting UI at http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=False)


if __name__ == "__main__":
    main()