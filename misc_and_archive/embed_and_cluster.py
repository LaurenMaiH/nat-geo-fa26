"""
Embed crops with BioClip 2, average per track, cluster with HDBSCAN, visualize with UMAP.

Pipeline per crop:
    raw → BiRefNet background removal → composite on black → CLAHE → pad → CLIP preprocess

Expected input structure:
    pipeline_output/
        <video_name>/
            id3_fish_f0042_best.jpg
            id3_fish_f0042_early.jpg
            ...
            
Caching:
    - Preprocessed crops are saved to PROCESSED_OUTPUT so BG removal + CLAHE
      only run once per crop.
    - Per-track embeddings are saved to EMBEDDINGS_PATH so the BioClip forward
      pass only runs once. Delete the JSON to force re-embedding (e.g. after
      swapping in a finetuned checkpoint).


Install:
    pip install open_clip_torch torch torchvision umap-learn hdbscan scikit-learn seaborn tqdm
    pip install "rembg[gpu]"
"""

import json
import numpy as np
import torch
import open_clip
import cv2
from PIL import Image
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
 
import hdbscan
import umap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import silhouette_score
 
from rembg import remove, new_session


# ============================================================
# CONFIG
# ============================================================
 
_MISSION_DIR      = Path(__file__).parent.parent / "processed_missions" / "mission_11"
PIPELINE_OUTPUT   = _MISSION_DIR / "three_crop_output"
PROCESSED_OUTPUT  = _MISSION_DIR / "three_crop_output_processed"  # bg-removed crops saved here
EMBEDDINGS_PATH   = _MISSION_DIR / "embeddings.json"              # key: "video/track_id", value: vector
MODEL_NAME        = "hf-hub:imageomics/bioclip-2"
CHECKPOINT_PATH   = None            # "checkpoints/best.pt" - set to None for pretrained
BATCH_SIZE        = 64
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
 
# Background removal — BiRefNet 
APPLY_BG_REMOVAL   = False
BG_COMPOSITE_COLOR = (0, 0, 0)   # matches pad_to_square pad color
 
# CLAHE — applied to each crop (after bg removal) before embedding
APPLY_CLAHE       = True
CLAHE_CLIP_LIMIT  = 3.0
CLAHE_TILE_SIZE   = (8, 8)   # smaller tiles than inference = more local contrast
 
# Area feature — appended to each track embedding before clustering
# 0.0 = disabled, higher = size matters more relative to appearance
AREA_FEATURE_WEIGHT = 0.3
 
# HDBSCAN
HDBSCAN_MIN_CLUSTER_SIZE = 12
HDBSCAN_MIN_SAMPLES      = 3
 
# UMAP
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST    = 0.1
 
# Visualization
SAMPLES_PER_CLUSTER  = 8
MAX_CLUSTERS_TO_SHOW = 20
 
print(f"Using device: {DEVICE}")

# ============================================================
# BACKGROUND REMOVAL (BiRefNet via rembg)
# ============================================================
 
try:
    bg_session = new_session("birefnet-general")
    print("Using BiRefNet session")
except Exception as e:
    print(f"BiRefNet unavailable ({e}), falling back to u2net")
    bg_session = new_session("u2net")
    print("Fell back to u2net session")
 
 
def remove_background(pil_img: Image.Image) -> Image.Image:
    """Remove background with rembg/BiRefNet, feather the edge, return RGBA."""
    result_rgba = remove(pil_img.convert("RGB"), session=bg_session)
    w_crop, h_crop = pil_img.size
 
    alpha_raw = np.array(result_rgba)[:, :, 3]
    alpha_mask = cv2.resize(
        alpha_raw.astype(np.float32),
        (w_crop, h_crop),
        interpolation=cv2.INTER_LANCZOS4,
    )
    alpha_mask = np.clip(alpha_mask, 0, 255).astype(np.uint8)
 
    # Feather edge
    alpha_f    = cv2.GaussianBlur(alpha_mask.astype(np.float32), (3, 3), sigmaX=0.8)
    alpha_mask = np.clip(alpha_f, 0, 255).astype(np.uint8)
 
    rgba = pil_img.convert("RGBA")
    rgba.putalpha(Image.fromarray(alpha_mask))
    return rgba
 
 
def composite_on_color(rgba: Image.Image, bg_color=(0, 0, 0)) -> Image.Image:
    """
    Flatten an RGBA image onto a solid background color, returning RGB.
    BioClip's preprocess expects RGB; black matches pad_to_square's pad color
    so the foreground sits on a uniform black canvas through the pipeline.
    """
    bg = Image.new("RGB", rgba.size, bg_color)
    bg.paste(rgba, mask=rgba.split()[3])
    return bg
 
# ============================================================
# CLAHE PREPROCESSING
# ============================================================
 
def apply_clahe(pil_img: Image.Image) -> Image.Image:
    """
    Apply CLAHE on the L channel of LAB colorspace to enhance
    local contrast without blowing out colors.
    """
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_SIZE)
    bgr   = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    lab   = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_ch, a, b = cv2.split(lab)
    lab   = cv2.merge([clahe.apply(l_ch), a, b])
    bgr   = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
 
 
# ============================================================
# PREPROCESSING — save processed crops to disk
# ============================================================
 
def preprocess_and_save_crops(paths: list[Path]) -> list[Path]:
    """
    For each raw crop: run BG removal + CLAHE, save to PROCESSED_OUTPUT
    mirroring the original directory structure. Already-processed files
    are skipped so reruns are fast.
    Returns the list of saved (processed) paths in the same order as input.
    """
    saved = []
    skipped = 0
    for p in tqdm(paths, desc="Preprocessing crops"):
        rel = p.relative_to(PIPELINE_OUTPUT)
        out = PROCESSED_OUTPUT / rel
        if out.exists():
            saved.append(out)
            skipped += 1
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            img = Image.open(p).convert("RGB")
            if APPLY_BG_REMOVAL:
                rgba = remove_background(img)
                img  = composite_on_color(rgba, BG_COMPOSITE_COLOR)
            if APPLY_CLAHE:
                img = apply_clahe(img)
            img.save(out)
            saved.append(out)
        except Exception as e:
            print(f"  Skipping {p.name}: {e}")
    print(f"Preprocessed {len(saved)}/{len(paths)} crops → {PROCESSED_OUTPUT}"
          + (f"  ({skipped} already cached)" if skipped else ""))
    return saved
 
 
# ============================================================
# TRACK ID PARSING
# If your filename convention changes, only edit this function.
# e.g. for "PLW_c001_frame00042_det00.jpg" you might do:
#     return stem.split("_det")[0]
# ============================================================
 
def parse_track_id(path: Path) -> str:
    """Extract track ID from a crop filename."""
    return path.stem.split("_")[0]   # "id3_fish_f0042_best" → "id3"
 
 
# ============================================================
# COLLECT CROPS
# ============================================================
 
def collect_crop_paths(pipeline_output: Path) -> list[Path]:
    """Gather all .jpg crops from pipeline_output/<video>/"""
    paths = sorted(pipeline_output.glob("*/*.jpg"))
    print(f"Found {len(paths)} crop images across "
          f"{len(set(p.parent.name for p in paths))} video(s)")
    return paths
 
 
# ============================================================
# PER-TRACK PIXEL AREA
# Computed from RAW crops (pre-BG-removal) since those are the
# original bounding-box sizes from the detector.
# ============================================================
 
def compute_track_areas(paths: list[Path]) -> dict[str, float]:
    """Return mean pixel area (w*h) for each track, averaged across its crops."""
    track_areas = defaultdict(list)
    for p in paths:
        try:
            w, h = Image.open(p).size
            track_areas[parse_track_id(p)].append(w * h)
        except Exception:
            pass
    return {tid: float(np.mean(areas)) for tid, areas in track_areas.items()}
 
 
# ============================================================
# CROP LOADING
# ============================================================
 
def load_crop(path: Path) -> Image.Image:
    """Load a preprocessed crop (already BG-removed + CLAHE)."""
    return Image.open(path).convert("RGB")
 
 
# ============================================================
# EMBEDDING
# ============================================================
 
def load_bioclip(model_name: str, device: str, checkpoint_path=None):
    print(f"Loading {model_name}...")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name)
 
    if checkpoint_path:
        print(f"Loading finetuned weights from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.visual.load_state_dict(ckpt["model_state"])
        print("Finetuned weights loaded ✓")
    else:
        print("Using pretrained weights")
 
    model = model.to(device).eval()
    print("BioClip 2 loaded ✓")
    return model, preprocess
 
 
def pad_to_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w == h:
        return img
    size   = max(w, h)
    padded = Image.new("RGB", (size, size), (0, 0, 0))
    padded.paste(img, ((size - w) // 2, (size - h) // 2))
    return padded
 
 
def extract_embeddings(
    paths: list[Path],
    model,
    preprocess,
    device: str,
    batch_size: int = 64,
) -> np.ndarray:
    all_emb = []
    for i in tqdm(range(0, len(paths), batch_size), desc="Extracting embeddings"):
        batch = []
        for p in paths[i:i + batch_size]:
            try:
                batch.append(preprocess(pad_to_square(load_crop(p))))
            except Exception as e:
                print(f"  Skipping {p.name}: {e}")
        if not batch:
            continue
        t = torch.stack(batch).to(device)
        with torch.no_grad(), torch.autocast(device):
            feats = model.encode_image(t)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        all_emb.append(feats.cpu().float().numpy())
    return np.vstack(all_emb)
 
 
# ============================================================
# EMBEDDING DATABASE  (JSON: "video/track_id" → vector)
# ============================================================
 
def save_embeddings(track_ids: list, embeddings: np.ndarray, processed_paths: list[Path],
                    out: Path = EMBEDDINGS_PATH):
    """
    Persist per-track embeddings to JSON.
    Key format: "<video>/<track_id>"  — unique across videos.
    Value: embedding vector as a float list.
    """
    track_to_video = {parse_track_id(p): p.parent.name for p in processed_paths}
    data = {
        f"{track_to_video.get(tid, 'unknown')}/{tid}": emb.tolist()
        for tid, emb in zip(track_ids, embeddings)
    }
    with open(out, "w") as f:
        json.dump(data, f)
    print(f"Saved {len(data)} embeddings → {out}")
 
 
def load_embeddings(path: Path = EMBEDDINGS_PATH):
    """
    Load embeddings saved by save_embeddings().
    Returns (track_ids, embeddings) where track_ids are the full "video/track_id" keys.
    Returns (None, None) if the file does not exist.
    """
    if not path.exists():
        return None, None
    with open(path) as f:
        data = json.load(f)
    track_ids  = list(data.keys())
    embeddings = np.array(list(data.values()), dtype=np.float32)
    print(f"Loaded {len(track_ids)} embeddings from {path}")
    return track_ids, embeddings
 
 
# ============================================================
# AVERAGE PER TRACK
# ============================================================
 
def average_per_track(
    paths: list[Path],
    embeddings: np.ndarray,
) -> tuple[list[str], np.ndarray]:
    """
    Average embeddings across all crops belonging to the same track ID.
    Returns (track_ids, averaged_embeddings).
    """
    track_embs = defaultdict(list)
    for path, emb in zip(paths, embeddings):
        tid = parse_track_id(path)
        track_embs[tid].append(emb)
 
    track_ids = sorted(track_embs.keys())
    averaged  = np.stack([np.mean(track_embs[tid], axis=0) for tid in track_ids])
 
    # re-normalise after averaging
    norms    = np.linalg.norm(averaged, axis=1, keepdims=True)
    averaged = averaged / np.where(norms == 0, 1, norms)
 
    print(f"Averaged to {len(track_ids)} tracks  (shape {averaged.shape})")
    return track_ids, averaged
 
 
# ============================================================
# AREA FEATURE INJECTION
# Appends a single normalized area dimension to each track embedding so
# size influences clustering alongside appearance. Useful for separating
# species that look similar but differ in scale.
# ============================================================
 
def append_area_feature(
    track_ids: list[str],
    embeddings: np.ndarray,
    track_areas: dict[str, float],
    weight: float = AREA_FEATURE_WEIGHT,
) -> np.ndarray:
    """
    Append a single normalized area dimension to each track embedding.
 
    Steps:
      1. Collect log(area) for each track (log compresses the wide range).
      2. Min-max normalize to [0, 1] across all tracks.
      3. Scale by `weight` so the feature has controlled influence.
      4. Concatenate and re-normalize the full vector to unit length.
 
    Track IDs may be either bare ("id3") or namespaced ("video/id3");
    the bare form is used to look up areas.
    """
    def bare(tid: str) -> str:
        return tid.split("/")[-1]
 
    areas = np.array(
        [track_areas.get(bare(tid), 0.0) for tid in track_ids],
        dtype=np.float32,
    )
    log_areas  = np.log1p(areas)                    # log1p avoids log(0)
    lo, hi     = log_areas.min(), log_areas.max()
    norm_areas = (log_areas - lo) / (hi - lo + 1e-8)  # → [0, 1]
    scaled     = (norm_areas * weight).reshape(-1, 1)
 
    augmented = np.hstack([embeddings, scaled])
    norms     = np.linalg.norm(augmented, axis=1, keepdims=True)
    augmented = augmented / np.where(norms == 0, 1, norms)
 
    print(f"Area feature appended (weight={weight}). "
          f"Embedding dim: {embeddings.shape[1]} → {augmented.shape[1]}")
    return augmented
 
 
# ============================================================
# CLUSTERING
# ============================================================
 
def cluster_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples: int = HDBSCAN_MIN_SAMPLES,
) -> np.ndarray:
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        core_dist_n_jobs=-1,
    )
    labels = clusterer.fit_predict(embeddings)
    unique, counts = np.unique(labels, return_counts=True)
    n_clusters = len([u for u in unique if u != -1])
    n_noise    = int((labels == -1).sum())
    print(f"\nHDBSCAN → {n_clusters} clusters, {n_noise} noise points "
          f"({n_noise / len(labels):.1%})")
    for u, c in zip(unique, counts):
        tag = " (noise)" if u == -1 else ""
        print(f"  Cluster {u:>3}: {c:>4} tracks{tag}")
    if n_clusters > 1 and n_noise < len(labels):
        non_noise = labels != -1
        sil = silhouette_score(embeddings[non_noise], labels[non_noise],
                               sample_size=min(3000, non_noise.sum()))
        print(f"Silhouette (non-noise): {sil:.4f}")
    return labels
 
 
# ============================================================
# VISUALIZATION
# ============================================================
 
def best_crop_for_track(tid, pipeline_output):
    """Return the 'best' crop image for a track ID, falling back to any crop."""
    bare = tid.split("/")[-1]  # handle "video/track_id" keys from load_embeddings
    matches = sorted(pipeline_output.glob(f"*/{bare}_*_best.jpg"))
    if matches:
        return matches[0]
    matches = sorted(pipeline_output.glob(f"*/{bare}_*.jpg"))
    return matches[0] if matches else None
 
 
def plot_clusters(
    labels: np.ndarray,
    track_ids: list[str],
    pipeline_output: Path,
    samples_per_cluster: int = SAMPLES_PER_CLUSTER,
    max_clusters: int = MAX_CLUSTERS_TO_SHOW,
):
    unique = np.unique(labels)
    show   = [lbl for lbl in sorted(unique) if lbl != -1][:max_clusters]
 
    _, axes = plt.subplots(
        len(show), samples_per_cluster,
        figsize=(samples_per_cluster * 1.8, len(show) * 2.2)
    )
    if len(show) == 1:
        axes = [axes]
 
    for row, lbl in enumerate(show):
        idxs   = np.where(labels == lbl)[0]
        chosen = np.random.choice(idxs, size=min(samples_per_cluster, len(idxs)), replace=False)
        for col in range(samples_per_cluster):
            ax = axes[row][col]
            ax.axis("off")
            if col < len(chosen):
                tid  = track_ids[chosen[col]]
                path = best_crop_for_track(tid, pipeline_output)
                if path:
                    try:
                        ax.imshow(load_crop(path))
                    except Exception:
                        pass
            if col == 0:
                ax.set_title(f"Cluster {lbl}  (n={len(idxs)})",
                             loc="left", fontsize=9, fontweight="bold", pad=4)
 
    plt.suptitle("HDBSCAN clusters", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()
 
 
def plot_all_in_cluster(
    lbl: int,
    labels: np.ndarray,
    track_ids: list[str],
    pipeline_output: Path,
    cols: int = 8,
):
    idxs = np.where(labels == lbl)[0]
    tag  = "Noise" if lbl == -1 else f"Cluster {lbl}"
    print(f"{tag}: {len(idxs)} tracks")
 
    rows = (len(idxs) + cols - 1) // cols
    _, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 2.2))
    axes = np.array(axes).reshape(-1, cols)
 
    for i, idx in enumerate(idxs):
        ax   = axes[i // cols][i % cols]
        ax.axis("off")
        tid  = track_ids[idx]
        path = best_crop_for_track(tid, pipeline_output)
        if path:
            try:
                ax.imshow(load_crop(path))
                ax.set_title(tid, fontsize=5, pad=2)
            except Exception:
                pass
 
    for j in range(len(idxs), rows * cols):
        axes[j // cols][j % cols].axis("off")
 
    plt.suptitle(f"{tag} — all {len(idxs)} tracks", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()
 
 
def plot_umap(embeddings: np.ndarray, labels: np.ndarray):
    print("Running UMAP...")
    reducer = umap.UMAP(
        n_components=2, metric="cosine",
        n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST,
        random_state=42, verbose=False,
    )
    coords  = reducer.fit_transform(embeddings)
    unique  = np.unique(labels)
    palette = sns.color_palette("tab20", n_colors=len(unique))
    cmap    = {lbl: palette[i] for i, lbl in enumerate(sorted(unique))}
    colors  = [cmap[lbl] for lbl in labels]
 
    plt.figure(figsize=(12, 8))
    plt.scatter(coords[:, 0], coords[:, 1], c=colors, s=8, alpha=0.7, linewidths=0)
    patches = [mpatches.Patch(color=cmap[lbl],
                               label=f"Cluster {lbl}" + (" (noise)" if lbl == -1 else ""))
               for lbl in sorted(unique)]
    plt.legend(handles=patches, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    plt.title(f"BioClip 2 — UMAP  (HDBSCAN, "
              f"{len([u for u in unique if u != -1])} clusters)", fontsize=13)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
 
 
# ============================================================
# MAIN
# ============================================================
 
def main():
    crop_paths = collect_crop_paths(PIPELINE_OUTPUT)
    if not crop_paths:
        raise RuntimeError(f"No crops found under {PIPELINE_OUTPUT}")
 
    print(f"Background removal:  {'ON' if APPLY_BG_REMOVAL else 'OFF'}")
    print(f"CLAHE preprocessing: {'ON' if APPLY_CLAHE else 'OFF'}")
    print(f"Area feature weight: {AREA_FEATURE_WEIGHT}")
 
    # ---- preprocess (cached on disk) ----
    processed_paths = preprocess_and_save_crops(crop_paths)
    if not processed_paths:
        raise RuntimeError("No crops survived preprocessing.")
 
    # ---- embed (cached in embeddings.json) ----
    track_ids, track_embs = load_embeddings()
    if track_ids is None:
        model, preprocess      = load_bioclip(MODEL_NAME, DEVICE, CHECKPOINT_PATH)
        per_crop_embs          = extract_embeddings(processed_paths, model, preprocess,
                                                    DEVICE, BATCH_SIZE)
        track_ids, track_embs  = average_per_track(processed_paths, per_crop_embs)
        save_embeddings(track_ids, track_embs, processed_paths)
    else:
        print("Skipping embedding — loaded from cache. Delete embeddings.json to re-embed.")
 
    # ---- optional area feature ----
    # Area is computed from RAW crops so original bbox sizes are used,
    # not the post-BG-removal canvases (which are still bbox-sized anyway,
    # but raw is the canonical source).
    if AREA_FEATURE_WEIGHT > 0:
        track_areas = compute_track_areas(crop_paths)
        track_embs  = append_area_feature(track_ids, track_embs, track_areas)
 
    # ---- cluster ----
    labels = cluster_hdbscan(track_embs)
 
    # ---- visualize ----
    plot_clusters(labels, track_ids, PROCESSED_OUTPUT)
 
    if -1 in labels:
        plot_all_in_cluster(-1, labels, track_ids, PROCESSED_OUTPUT)
 
    unique, counts = np.unique(labels, return_counts=True)
    for lbl, count in zip(unique, counts):
        if lbl != -1 and count > 100:
            plot_all_in_cluster(lbl, labels, track_ids, PROCESSED_OUTPUT)
 
    plot_umap(track_embs, labels)
 
 
main()