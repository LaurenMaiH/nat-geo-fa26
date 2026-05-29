from flask import Flask, jsonify, request, send_file
from pathlib import Path
from collections import defaultdict
import json
import numpy as np
import hdbscan as hdbscan_lib

app = Flask(__name__)

BASE_DIR = Path(__file__).parent.parent  # nat-geo/ (clustering-ui lives inside nat-geo/)
CROPS_DIR = BASE_DIR / "three_crop_output"
EMBEDDINGS_PATH = BASE_DIR / "embeddings.json"
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
        "hdbscan_params": {"min_cluster_size": 12, "min_samples": 3},
        "cosine_threshold": 0.15,
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
    body = request.get_json()
    seed_video = body["seed_video"]
    min_cluster_size = int(body.get("min_cluster_size", 12))
    min_samples = int(body.get("min_samples", 3))

    emb_dict = load_embeddings()
    all_vids = sorted_videos(emb_dict)
    if seed_video not in all_vids:
        return jsonify({"error": "Unknown video"}), 400

    seed_batch = set(all_vids[: all_vids.index(seed_video) + 1])
    track_keys = [k for k in emb_dict if k.split("/")[0] in seed_batch]
    embs = np.array([emb_dict[k] for k in track_keys], dtype=np.float32)

    labels = hdbscan_lib.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        core_dist_n_jobs=-1,
    ).fit_predict(embs)

    cluster_tracks = defaultdict(list)
    for key, label in zip(track_keys, labels):
        cluster_tracks[int(label)].append(key)

    clusters = {
        "noise": {
            "id": "noise", "name": "Unassigned", "color": "#6B7280",
            "track_keys": [], "centroid": None, "representative": None,
        }
    }
    assignments = {}
    color_idx = 0

    for label in sorted(cluster_tracks):
        keys = cluster_tracks[label]
        if label == -1:
            clusters["noise"]["track_keys"].extend(keys)
            for k in keys:
                assignments[k] = "noise"
        else:
            cid = f"cluster_{label}"
            centroid = compute_centroid(keys, emb_dict)
            clusters[cid] = {
                "id": cid,
                "name": f"Cluster {label}",
                "color": COLORS[color_idx % len(COLORS)],
                "track_keys": keys,
                "centroid": centroid,
                "representative": find_representative(keys, centroid, emb_dict),
            }
            color_idx += 1
            for k in keys:
                assignments[k] = cid

    track_counts = defaultdict(int)
    for k in emb_dict:
        track_counts[k.split("/")[0]] += 1

    videos = {
        v: {"name": v, "status": "approved" if v in seed_batch else "pending", "track_count": track_counts[v]}
        for v in all_vids
    }
    current_video = next((v for v in all_vids if videos[v]["status"] == "pending"), None)

    state = {
        "initialized": True,
        "seed_video": seed_video,
        "hdbscan_params": {"min_cluster_size": min_cluster_size, "min_samples": min_samples},
        "cosine_threshold": 0.15,
        "videos": videos,
        "clusters": clusters,
        "track_assignments": assignments,
        "current_video": current_video,
    }
    save_state(state)
    return jsonify(state)


@app.route("/api/process-video", methods=["POST"])
def api_process_video():
    body = request.get_json()
    video_name = body["video_name"]
    threshold = float(body.get("threshold", 0.15))

    state = load_state()
    emb_dict = load_embeddings()
    track_keys = [k for k in emb_dict if k.split("/")[0] == video_name]

    for key in track_keys:
        emb = np.array(l2_norm(emb_dict[key]), dtype=np.float32)
        best_cid, best_dist = "noise", float("inf")

        for cid, cluster in state["clusters"].items():
            if cid == "noise" or not cluster.get("centroid"):
                continue
            dist = 1.0 - float(np.dot(emb, np.array(cluster["centroid"], dtype=np.float32)))
            if dist < best_dist:
                best_dist, best_cid = dist, cid

        if best_dist > threshold:
            best_cid = "noise"

        old_cid = state["track_assignments"].get(key)
        if old_cid and old_cid != best_cid:
            old_keys = state["clusters"].get(old_cid, {}).get("track_keys", [])
            if key in old_keys:
                old_keys.remove(key)

        if key not in state["clusters"][best_cid]["track_keys"]:
            state["clusters"][best_cid]["track_keys"].append(key)
        state["track_assignments"][key] = best_cid

        # Moving-average centroid update for non-noise clusters
        if best_cid != "noise":
            cluster = state["clusters"][best_cid]
            n = len(cluster["track_keys"])
            if cluster["centroid"] and n > 1:
                old_c = np.array(cluster["centroid"], dtype=np.float32)
                cluster["centroid"] = l2_norm((old_c * (n - 1) + emb) / n)
            else:
                cluster["centroid"] = emb.tolist()

    for cid, cluster in state["clusters"].items():
        if cid != "noise" and cluster.get("centroid"):
            cluster["representative"] = find_representative(
                cluster["track_keys"], cluster["centroid"], emb_dict
            )

    state["videos"][video_name] = {"name": video_name, "status": "processed", "track_count": len(track_keys)}
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

    # Pick an unused cluster number
    nums = [int(k.split("_")[1]) for k in state["clusters"] if k.startswith("cluster_") and k.split("_")[1].isdigit()]
    new_num = max(nums) + 1 if nums else 0
    new_cid = f"cluster_{new_num}"

    # Remove track from its current cluster
    old_cid = state["track_assignments"].get(track_key, "noise")
    old_keys = state["clusters"].get(old_cid, {}).get("track_keys", [])
    if track_key in old_keys:
        old_keys.remove(track_key)
    if old_cid != "noise" and old_cid in state["clusters"]:
        cluster = state["clusters"][old_cid]
        cluster["centroid"] = compute_centroid(cluster["track_keys"], emb_dict)
        cluster["representative"] = find_representative(cluster["track_keys"], cluster["centroid"], emb_dict)

    # Pick the first color not already in use
    used_colors = {c["color"] for c in state["clusters"].values()}
    color = next((c for c in COLORS if c not in used_colors), COLORS[new_num % len(COLORS)])

    centroid = l2_norm(emb_dict[track_key]) if track_key in emb_dict else None
    state["clusters"][new_cid] = {
        "id": new_cid,
        "name": f"Cluster {new_num}",
        "color": color,
        "track_keys": [track_key],
        "centroid": centroid,
        "representative": track_key,
    }
    state["track_assignments"][track_key] = new_cid

    prune_empty_clusters(state)
    save_state(state)
    return jsonify(state)


@app.route("/api/crop/<video>/<track_id>")
def api_crop(video, track_id):
    path = best_crop_path(video, track_id)
    if not path:
        return "Not found", 404
    return send_file(path, mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
