"""
Embed crops with BioClip 2, average per track, cluster with HDBSCAN, visualize with UMAP.

Expected input structure:
    pipeline_output/
        <video_name>/
            id3_fish_f0042_best.jpg
            id3_fish_f0042_early.jpg
            ...

Install:
    pip install open_clip_torch torch torchvision umap-learn hdbscan scikit-learn seaborn tqdm
"""

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


# ============================================================
# CONFIG
# ============================================================

PIPELINE_OUTPUT = Path("three_crop_output")
MODEL_NAME      = "hf-hub:imageomics/bioclip-2"
#CHECKPOINT_PATH = "checkpoints/best.pt"  # set to None to use pretrained weights
CHECKPOINT_PATH = None
BATCH_SIZE      = 64
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# CLAHE — applied to each crop before embedding
# Set APPLY_CLAHE = False to disable
APPLY_CLAHE       = True
CLAHE_CLIP_LIMIT  = 3.0
CLAHE_TILE_SIZE   = (8, 8)   # smaller tiles than inference = more local contrast

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
    l, a, b = cv2.split(lab)
    lab   = cv2.merge([clahe.apply(l), a, b])
    bgr   = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


# ============================================================
# EMBEDDING
# ============================================================

def load_bioclip(model_name: str, device: str, checkpoint_path: str = None):
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


def load_crop(path: Path) -> Image.Image:
    """Load a crop, optionally applying CLAHE."""
    img = Image.open(path).convert("RGB")
    if APPLY_CLAHE:
        img = apply_clahe(img)
    return img


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

def best_crop_for_track(tid: str, pipeline_output: Path) -> Path | None:
    """Return the 'best' crop image for a track ID, falling back to any crop."""
    matches = sorted(pipeline_output.glob(f"*/{tid}_*_best.jpg"))
    if matches:
        return matches[0]
    matches = sorted(pipeline_output.glob(f"*/{tid}_*.jpg"))
    return matches[0] if matches else None


def plot_clusters(
    labels: np.ndarray,
    track_ids: list[str],
    pipeline_output: Path,
    samples_per_cluster: int = SAMPLES_PER_CLUSTER,
    max_clusters: int = MAX_CLUSTERS_TO_SHOW,
):
    unique = np.unique(labels)
    show   = [l for l in sorted(unique) if l != -1][:max_clusters]

    fig, axes = plt.subplots(
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
                        ax.imshow(Image.open(path).convert("RGB"))
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
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 2.2))
    axes = np.array(axes).reshape(-1, cols)

    for i, idx in enumerate(idxs):
        ax   = axes[i // cols][i % cols]
        ax.axis("off")
        tid  = track_ids[idx]
        path = best_crop_for_track(tid, pipeline_output)
        if path:
            try:
                ax.imshow(Image.open(path).convert("RGB"))
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
    colors  = [cmap[l] for l in labels]

    plt.figure(figsize=(12, 8))
    plt.scatter(coords[:, 0], coords[:, 1], c=colors, s=8, alpha=0.7, linewidths=0)
    patches = [mpatches.Patch(color=cmap[l],
                               label=f"Cluster {l}" + (" (noise)" if l == -1 else ""))
               for l in sorted(unique)]
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

    print(f"CLAHE preprocessing: {'ON' if APPLY_CLAHE else 'OFF'}")
    model, preprocess = load_bioclip(MODEL_NAME, DEVICE, CHECKPOINT_PATH)

    per_crop_embs         = extract_embeddings(crop_paths, model, preprocess, DEVICE, BATCH_SIZE)
    track_ids, track_embs = average_per_track(crop_paths, per_crop_embs)
    labels                = cluster_hdbscan(track_embs)

    plot_clusters(labels, track_ids, PIPELINE_OUTPUT)

    # inspect noise cluster
    if -1 in labels:
        plot_all_in_cluster(-1, labels, track_ids, PIPELINE_OUTPUT)

    # inspect any cluster with more than 100 tracks
    unique, counts = np.unique(labels, return_counts=True)
    for lbl, count in zip(unique, counts):
        if lbl != -1 and count > 100:
            plot_all_in_cluster(lbl, labels, track_ids, PIPELINE_OUTPUT)

    plot_umap(track_embs, labels)


main()