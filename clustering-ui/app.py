from flask import Flask, jsonify, request, send_file
from pathlib import Path
from collections import defaultdict
import json
import numpy as np
import hdbscan as hdbscan_lib

app = Flask(__name__)

BASE_DIR = Path(__file__).parent.parent  # nat-geo/ (clustering-ui lives inside nat-geo/)
CROPS_DIR = BASE_DIR / "processed_missions" / "mission_11" / "three_crop_output"
EMBEDDINGS_PATH = BASE_DIR / "processed_missions" / "mission_11" / "embeddings.json"
STATE_PATH = Path(__file__).parent / "state.json"


COLORS = [
    "#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6",
    "#EC4899", "#14B8A6", "#F97316", "#84CC16", "#06B6D4",
    "#A855F7", "#F43F5E", "#0EA5E9", "#22C55E", "#EAB308",
]


# ── State ──────────────────────────────────────────────────────────────────────

def load_state():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {
        "initialized": False,
        "seed_video": None,
        "hdbscan_params": {"min_cluster_size": 2, "min_samples": 1},
        "cosine_threshold": 0.25,
        "videos": {},
        "clusters": {
            "noise": {
                "id": "noise", "name": "Unassigned", "color": "#6B7280",
                "track_keys": [], "centroid": None, "representative": None,
            }
        },
        "track_assignments": {},
        "current_video": None,
    }


def save_state(state):
    STATE_PATH.write_text(json.dumps(state, indent=2))


def load_embeddings():
    if not EMBEDDINGS_PATH.exists():
        return {}
    return json.loads(EMBEDDINGS_PATH.read_text())


# ── Helpers ────────────────────────────────────────────────────────────────────

def l2_norm(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return (v / n).tolist() if n > 0 else v.tolist()


def compute_centroid(track_keys, emb_dict):
    vecs = [emb_dict[k] for k in track_keys if k in emb_dict]
    if not vecs:
        return None
    return l2_norm(np.mean(vecs, axis=0))


def find_representative(track_keys, centroid, emb_dict):
    if not centroid or not track_keys:
        return None
    c = np.array(centroid, dtype=np.float32)
    best, best_dist = None, float("inf")
    for k in track_keys:
        if k in emb_dict:
            d = 1.0 - float(np.dot(np.array(l2_norm(emb_dict[k])), c))
            if d < best_dist:
                best, best_dist = k, d
    return best


def prune_empty_clusters(state):
    empty = [cid for cid, c in state["clusters"].items()
             if cid != "noise" and not c.get("track_keys")]
    for cid in empty:
        del state["clusters"][cid]
        for k, v in state["track_assignments"].items():
            if v == cid:
                state["track_assignments"][k] = "noise"
                if k not in state["clusters"]["noise"]["track_keys"]:
                    state["clusters"]["noise"]["track_keys"].append(k)


def sorted_videos(emb_dict):
    return sorted({k.split("/")[0] for k in emb_dict})


def best_crop_path(video, track_id):
    video_dir = CROPS_DIR / video
    if not video_dir.exists():
        return None
    matches = sorted(video_dir.glob(f"{track_id}_*_best.jpg"))
    if not matches:
        matches = sorted(video_dir.glob(f"{track_id}_*.jpg"))
    return matches[0] if matches else None


def track_sort_key(tid):
    try:
        return int(tid.lstrip("id") or 0)
    except ValueError:
        return 0


def next_color(state):
    """Pick the first color from COLORS not already in use."""
    used = {c["color"] for c in state["clusters"].values()}
    return next((c for c in COLORS if c not in used), COLORS[len(state["clusters"]) % len(COLORS)])


def next_cluster_num(state):
    """Return the next unused cluster integer."""
    nums = [
        int(k.split("_")[1])
        for k in state["clusters"]
        if k.startswith("cluster_") and k.split("_")[1].isdigit()
    ]
    return max(nums) + 1 if nums else 0


# ── Sequential HDBSCAN helpers ────────────────────────────────────────────────

def hdbscan_video(track_keys, emb_dict, min_cluster_size, min_samples):
    """
    Run HDBSCAN on a single video's tracks.
    Returns {local_label: [track_key, ...]}
    Noise points (label == -1) get individual singleton entries.
    """
    if not track_keys:
        return {}

    embs = np.array([emb_dict[k] for k in track_keys], dtype=np.float32)

    if len(track_keys) == 1:
        return {"0": track_keys}

    labels = hdbscan_lib.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        core_dist_n_jobs=-1,
    ).fit_predict(embs)

    result = defaultdict(list)
    noise_idx = 0
    for key, label in zip(track_keys, labels):
        if label == -1:
            result[f"noise_{noise_idx}"].append(key)
            noise_idx += 1
        else:
            result[str(label)].append(key)

    return dict(result)


def match_local_to_global(local_centroid, clusters, threshold):
    """
    Find the closest existing global cluster by cosine distance.
    Returns (cluster_id, distance).
    If nothing is close enough (or no clusters exist), returns (None, inf).
    """
    best_cid, best_dist = None, float("inf")
    for cid, cluster in clusters.items():
        if cid == "noise" or not cluster.get("centroid"):
            continue
        c = np.array(cluster["centroid"], dtype=np.float32)
        lc = np.array(local_centroid, dtype=np.float32)
        dist = 1.0 - float(np.dot(lc, c))
        if dist < best_dist:
            best_dist, best_cid = dist, cid

    if best_cid is None or best_dist > threshold:
        return None, best_dist

    return best_cid, best_dist


def _send_to_noise(state, track_key, affected_cids):
    """Move a track to Unassigned, cleaning up its old cluster."""
    old_cid = state["track_assignments"].get(track_key)
    if old_cid and old_cid != "noise":
        old_keys = state["clusters"].get(old_cid, {}).get("track_keys", [])
        if track_key in old_keys:
            old_keys.remove(track_key)
        affected_cids.add(old_cid)
    if track_key not in state["clusters"]["noise"]["track_keys"]:
        state["clusters"]["noise"]["track_keys"].append(track_key)
    state["track_assignments"][track_key] = "noise"


def _assign_track(state, track_key, target_cid, affected_cids):
    """Move a track to target_cid, cleaning up its old cluster."""
    old_cid = state["track_assignments"].get(track_key)
    if old_cid and old_cid != target_cid:
        old_keys = state["clusters"].get(old_cid, {}).get("track_keys", [])
        if track_key in old_keys:
            old_keys.remove(track_key)
        affected_cids.add(old_cid)
    if track_key not in state["clusters"][target_cid]["track_keys"]:
        state["clusters"][target_cid]["track_keys"].append(track_key)
    state["track_assignments"][track_key] = target_cid
    affected_cids.add(target_cid)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_file(Path(__file__).parent / "templates" / "index.html")


@app.route("/api/state")
def api_state():
    state = load_state()
    emb = load_embeddings()
    track_counts = defaultdict(int)
    for k in emb:
        track_counts[k.split("/")[0]] += 1
    for v in sorted(track_counts):
        entry = state["videos"].get(v, {"name": v, "status": "pending"})
        entry["track_count"] = track_counts[v]
        state["videos"][v] = entry
    return jsonify(state)


@app.route("/api/videos")
def api_videos():
    state = load_state()
    emb = load_embeddings()
    track_counts = defaultdict(int)
    for k in emb:
        track_counts[k.split("/")[0]] += 1
    return jsonify([
        {
            "name": v,
            "status": state["videos"].get(v, {}).get("status", "pending"),
            "track_count": track_counts[v],
        }
        for v in sorted(track_counts)
    ])


@app.route("/api/video/<video_name>/tracks")
def api_video_tracks(video_name):
    state = load_state()
    emb = load_embeddings()
    track_ids = sorted(
        [k.split("/")[1] for k in emb if k.split("/")[0] == video_name],
        key=track_sort_key,
    )
    return jsonify([
        {
            "track_key": f"{video_name}/{tid}",
            "track_id": tid,
            "cluster_id": state["track_assignments"].get(f"{video_name}/{tid}", "noise"),
        }
        for tid in track_ids
    ])


@app.route("/api/initialize", methods=["POST"])
def api_initialize():
    """
    Seed the clustering state by running HDBSCAN on the seed video only.
    Each cluster found becomes a global cluster with a centroid.
    Noise tracks go to Unassigned.
    All subsequent videos are queued as pending.
    """
    body = request.get_json()
    seed_video       = body["seed_video"]
    min_cluster_size = int(body.get("min_cluster_size", 2))
    min_samples      = int(body.get("min_samples", 1))
    cosine_threshold = float(body.get("cosine_threshold", 0.25))

    emb_dict = load_embeddings()
    all_vids = sorted_videos(emb_dict)
    if seed_video not in all_vids:
        return jsonify({"error": "Unknown video"}), 400

    # ── HDBSCAN on seed video only ────────────────────────────────────────────
    track_keys    = [k for k in emb_dict if k.split("/")[0] == seed_video]
    local_clusters = hdbscan_video(track_keys, emb_dict, min_cluster_size, min_samples)

    clusters    = {
        "noise": {
            "id": "noise", "name": "Unassigned", "color": "#6B7280",
            "track_keys": [], "centroid": None, "representative": None,
        }
    }
    assignments = {}
    color_idx   = 0

    # First pass — build real clusters from HDBSCAN non-noise labels
    noise_keys = []
    for local_label, keys in sorted(local_clusters.items()):
        if local_label.startswith("noise_"):
            noise_keys.extend(keys)
            continue
        cid      = f"cluster_{color_idx}"
        centroid = compute_centroid(keys, emb_dict)
        clusters[cid] = {
            "id":             cid,
            "name":           f"Cluster {color_idx}",
            "color":          COLORS[color_idx % len(COLORS)],
            "track_keys":     keys,
            "centroid":       centroid,
            "representative": find_representative(keys, centroid, emb_dict),
        }
        for k in keys:
            assignments[k] = cid
        color_idx += 1

    # Second pass — give noise points a chance to join an existing cluster
    # or become their own singleton cluster rather than going straight to Unassigned
    for k in noise_keys:
        centroid = l2_norm(emb_dict[k]) if k in emb_dict else None
        if centroid is None:
            clusters["noise"]["track_keys"].append(k)
            assignments[k] = "noise"
            continue

        matched_cid, _ = match_local_to_global(centroid, clusters, cosine_threshold)

        if matched_cid is not None:
            # Close enough to an existing cluster — merge in
            clusters[matched_cid]["track_keys"].append(k)
            assignments[k] = matched_cid
        else:
            # Genuinely isolated — give it its own singleton cluster
            new_cid = f"cluster_{color_idx}"
            clusters[new_cid] = {
                "id":             new_cid,
                "name":           f"Cluster {color_idx}",
                "color":          COLORS[color_idx % len(COLORS)],
                "track_keys":     [k],
                "centroid":       centroid,
                "representative": k,
            }
            assignments[k] = new_cid
            color_idx += 1

    # Recompute centroids for any clusters that absorbed noise points
    for cid, cluster in clusters.items():
        if cid != "noise" and cluster.get("track_keys"):
            cluster["centroid"] = compute_centroid(cluster["track_keys"], emb_dict)
            cluster["representative"] = find_representative(
                cluster["track_keys"], cluster["centroid"], emb_dict
            )

    track_counts = defaultdict(int)
    for k in emb_dict:
        track_counts[k.split("/")[0]] += 1

    videos = {
        v: {
            "name":        v,
            "status":      "approved" if v == seed_video else "pending",
            "track_count": track_counts[v],
        }
        for v in all_vids
    }
    current_video = next((v for v in all_vids if videos[v]["status"] == "pending"), None)

    state = {
        "initialized":    True,
        "seed_video":     seed_video,
        "hdbscan_params": {"min_cluster_size": min_cluster_size, "min_samples": min_samples},
        "cosine_threshold": cosine_threshold,
        "videos":         videos,
        "clusters":       clusters,
        "track_assignments": assignments,
        "current_video":  current_video,
    }
    save_state(state)
    return jsonify(state)


@app.route("/api/process-video", methods=["POST"])
def api_process_video():
    """
    Process a new video using sequential HDBSCAN:
      1. Run HDBSCAN on this video's tracks independently.
      2. Compute a centroid per local cluster.
      3. Match each local cluster to the nearest global cluster by cosine distance.
         - Close enough  → merge into existing cluster, update centroid.
         - Too far away  → create a new global cluster (new species candidate).
      4. HDBSCAN noise singletons → Unassigned.
    """
    body       = request.get_json()
    video_name = body["video_name"]
    state      = load_state()
    emb_dict   = load_embeddings()

    min_cluster_size = state["hdbscan_params"]["min_cluster_size"]
    min_samples      = state["hdbscan_params"]["min_samples"]
    threshold        = float(body.get("threshold", state.get("cosine_threshold", 0.25)))

    track_keys    = [k for k in emb_dict if k.split("/")[0] == video_name]
    local_clusters = hdbscan_video(track_keys, emb_dict, min_cluster_size, min_samples)

    affected_cids = set()

    for local_label, keys in local_clusters.items():
        is_noise = local_label.startswith("noise_")

        if is_noise:
            # Don't immediately send to Unassigned — try centroid matching first
            for k in keys:
                noise_centroid = l2_norm(emb_dict[k]) if k in emb_dict else None
                if noise_centroid is None:
                    _send_to_noise(state, k, affected_cids)
                    continue

                matched_cid, _ = match_local_to_global(
                    noise_centroid, state["clusters"], threshold
                )

                if matched_cid is not None:
                    # Close enough to an existing cluster — merge in
                    _assign_track(state, k, matched_cid, affected_cids)
                else:
                    # Genuinely isolated — new singleton cluster
                    new_num = next_cluster_num(state)
                    new_cid = f"cluster_{new_num}"
                    state["clusters"][new_cid] = {
                        "id":             new_cid,
                        "name":           f"Cluster {new_num}",
                        "color":          next_color(state),
                        "track_keys":     [k],
                        "centroid":       noise_centroid,
                        "representative": k,
                    }
                    state["track_assignments"][k] = new_cid
                    affected_cids.add(new_cid)
            continue

        local_centroid = compute_centroid(keys, emb_dict)
        if local_centroid is None:
            continue

        matched_cid, dist = match_local_to_global(
            local_centroid, state["clusters"], threshold
        )

        if matched_cid is None:
            # ── New species candidate — create a new global cluster ───────────
            new_num = next_cluster_num(state)
            new_cid = f"cluster_{new_num}"
            state["clusters"][new_cid] = {
                "id":           new_cid,
                "name":         f"Cluster {new_num}",
                "color":        next_color(state),
                "track_keys":   list(keys),
                "centroid":     local_centroid,
                "representative": find_representative(keys, local_centroid, emb_dict),
            }
            for k in keys:
                state["track_assignments"][k] = new_cid
            affected_cids.add(new_cid)
        else:
            # ── Merge into existing global cluster ────────────────────────────
            cluster = state["clusters"][matched_cid]
            for k in keys:
                _assign_track(state, k, matched_cid, affected_cids)

    # ── Recompute centroids and representatives for all touched clusters ──────
    for cid in affected_cids:
        if cid == "noise" or cid not in state["clusters"]:
            continue
        cluster = state["clusters"][cid]
        cluster["centroid"] = compute_centroid(cluster["track_keys"], emb_dict)
        cluster["representative"] = find_representative(
            cluster["track_keys"], cluster["centroid"], emb_dict
        )

    prune_empty_clusters(state)

    track_count = len(track_keys)
    state["videos"][video_name] = {
        "name": video_name, "status": "processed", "track_count": track_count
    }
    state["current_video"] = video_name
    save_state(state)
    return jsonify(state)


@app.route("/api/reassign", methods=["POST"])
def api_reassign():
    body = request.get_json()
    target_cid = body["target_cluster_id"]

    state = load_state()
    emb_dict = load_embeddings()

    if target_cid not in state["clusters"]:
        return jsonify({"error": "Unknown cluster"}), 400

    affected = set()
    for key in body["track_keys"]:
        old_cid = state["track_assignments"].get(key, "noise")
        if old_cid == target_cid:
            continue
        old_keys = state["clusters"].get(old_cid, {}).get("track_keys", [])
        if key in old_keys:
            old_keys.remove(key)
        state["clusters"][target_cid]["track_keys"].append(key)
        state["track_assignments"][key] = target_cid
        affected.update([old_cid, target_cid])

    for cid in affected:
        if cid == "noise" or cid not in state["clusters"]:
            continue
        cluster = state["clusters"][cid]
        cluster["centroid"] = compute_centroid(cluster["track_keys"], emb_dict)
        cluster["representative"] = find_representative(
            cluster["track_keys"], cluster["centroid"], emb_dict
        )

    prune_empty_clusters(state)
    save_state(state)
    return jsonify(state)


@app.route("/api/approve", methods=["POST"])
def api_approve():
    body = request.get_json()
    video_name = body["video_name"]
    state = load_state()
    if video_name in state["videos"]:
        state["videos"][video_name]["status"] = "approved"
    all_vids = sorted_videos(load_embeddings())
    state["current_video"] = next(
        (v for v in all_vids if state["videos"].get(v, {}).get("status") != "approved"), None
    )
    save_state(state)
    return jsonify(state)


@app.route("/api/rename", methods=["POST"])
def api_rename():
    body = request.get_json()
    state = load_state()
    cid = body["cluster_id"]
    if cid in state["clusters"]:
        state["clusters"][cid]["name"] = body["name"]
    save_state(state)
    return jsonify(state)


@app.route("/api/new-cluster", methods=["POST"])
def api_new_cluster():
    body = request.get_json()
    track_key = body["track_key"]

    state = load_state()
    emb_dict = load_embeddings()

    new_num = next_cluster_num(state)
    new_cid = f"cluster_{new_num}"

    old_cid = state["track_assignments"].get(track_key, "noise")
    old_keys = state["clusters"].get(old_cid, {}).get("track_keys", [])
    if track_key in old_keys:
        old_keys.remove(track_key)
    if old_cid != "noise" and old_cid in state["clusters"]:
        cluster = state["clusters"][old_cid]
        cluster["centroid"] = compute_centroid(cluster["track_keys"], emb_dict)
        cluster["representative"] = find_representative(
            cluster["track_keys"], cluster["centroid"], emb_dict
        )

    centroid = l2_norm(emb_dict[track_key]) if track_key in emb_dict else None
    state["clusters"][new_cid] = {
        "id":           new_cid,
        "name":         f"Cluster {new_num}",
        "color":        next_color(state),
        "track_keys":   [track_key],
        "centroid":     centroid,
        "representative": track_key,
    }
    state["track_assignments"][track_key] = new_cid

    prune_empty_clusters(state)
    save_state(state)
    return jsonify(state)


@app.route("/api/reset", methods=["POST"])
def api_reset():
    if STATE_PATH.exists():
        STATE_PATH.unlink()
    return jsonify(load_state())


@app.route("/api/crop/<video>/<track_id>")
def api_crop(video, track_id):
    path = best_crop_path(video, track_id)
    if not path:
        return "Not found", 404
    return send_file(path, mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(debug=True, port=5000)