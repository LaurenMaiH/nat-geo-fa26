"""
Finetune BioClip 2 on labeled crops using supervised contrastive loss
with a 3-level similarity hierarchy (species → group → metagroup).

Expected directory structure:
    crops/
        Coryphaenoides/
            img001.jpg
            img002.jpg
        Synaphobranchus/
            img003.jpg
        ...

Install:
    pip install open_clip_torch torch torchvision tqdm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import open_clip
import random
import numpy as np
import csv

from taxonomy import SPECIES_TO_GROUP, GROUP_TO_METAGROUP, get_similarity  # your file


# ============================================================
# CONFIG
# ============================================================

CROPS_DIR       = Path("crops")
CHECKPOINT_DIR  = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

MODEL_NAME      = "hf-hub:imageomics/bioclip-2"
EMBED_DIM       = 768        # BioClip 2 vision embedding size
PROJ_DIM        = 128        # contrastive projection head output dim

BATCH_SIZE      = 32         # pairs are built inside the loss, not the loader
EPOCHS          = 20
LR              = 2e-5       # low LR — we're finetuning, not training from scratch
WEIGHT_DECAY    = 1e-4
TEMPERATURE     = 0.07       # contrastive temperature
WARMUP_EPOCHS   = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# ============================================================
# DATASET
# ============================================================

class CropDataset(Dataset):
    def __init__(self, root: Path, transform):
        self.transform = transform
        self.samples = []  # (path, species_label)

        for species_dir in sorted(root.iterdir()):
            if not species_dir.is_dir():
                continue
            species = species_dir.name
            for img_path in species_dir.iterdir():
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    self.samples.append((img_path, species))

        # build label→index maps
        self.species_list = sorted(set(s for _, s in self.samples))
        self.species_to_idx = {s: i for i, s in enumerate(self.species_list)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, species = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), species


# ============================================================
# PROJECTION HEAD
# ============================================================

class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


# ============================================================
# HIERARCHICAL SUPERVISED CONTRASTIVE LOSS
# ============================================================

def build_similarity_matrix(labels: list[str]) -> torch.Tensor:
    """
    Build an NxN matrix of pairwise similarity scores using get_similarity().
    Values: 1.0 / 0.5 / 0.2 / 0.0
    """
    n = len(labels)
    sim = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            sim[i, j] = get_similarity(labels[i], labels[j])
    return sim  # on CPU, moved to device in loss fn


def hierarchical_supcon_loss(
    embeddings: torch.Tensor,   # (N, D) L2-normalised
    labels: list[str],
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Soft supervised contrastive loss where positives are weighted
    by the similarity hierarchy rather than being binary.

    For each anchor i:
        loss_i = -sum_j [ w_ij * log( exp(z_i·z_j/τ) / sum_{k≠i} exp(z_i·z_k/τ) ) ]
                  / sum_j w_ij

    where w_ij = get_similarity(label_i, label_j), excluding i==j.
    """
    n = embeddings.size(0)
    sim_matrix = build_similarity_matrix(labels).to(embeddings.device)

    # cosine similarities scaled by temperature
    logits = (embeddings @ embeddings.T) / temperature  # (N, N)

    # mask out self-similarity in denominator
    self_mask = torch.eye(n, dtype=torch.bool, device=embeddings.device)
    logits_no_self = logits.masked_fill(self_mask, -1e4)  # large negative, not -inf

    log_probs = F.log_softmax(logits_no_self, dim=1)  # numerically stable

    # weight by similarity, zero out diagonal
    weights = sim_matrix.clone()
    weights.fill_diagonal_(0)

    weight_sum = weights.sum(dim=1).clamp(min=1e-6)
    loss_per_anchor = -(weights * log_probs).sum(dim=1) / weight_sum

    # only compute loss for anchors that have at least one positive
    has_positive = (weights.sum(dim=1) > 0)
    if has_positive.sum() == 0:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    return loss_per_anchor[has_positive].mean()


# ============================================================
# TRAINING
# ============================================================

def get_transform(model_transform):
    """Wrap the model's default transform with extra augmentation for training."""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        model_transform,
    ])


def train():
    # --- load model ---
    print(f"Loading {MODEL_NAME}...")
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME)
    model = model.visual          # vision encoder only
    model = model.to(DEVICE)

    train_transform = get_transform(preprocess)
    dataset = CropDataset(CROPS_DIR, train_transform)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=4, pin_memory=True, drop_last=True)

    print(f"Dataset: {len(dataset)} images, {len(dataset.species_list)} species")

    proj_head = ProjectionHead(EMBED_DIM, PROJ_DIM).to(DEVICE)

    params = list(model.parameters()) + list(proj_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)

    total_steps = EPOCHS * len(loader)
    warmup_steps = WARMUP_EPOCHS * len(loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, total_steps=total_steps,
        pct_start=warmup_steps / total_steps,
    )

    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE == "cuda"))

    ckpt = None  # guard against unbound on first epoch
    log_path = CHECKPOINT_DIR / "train_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "loss", "lr"])

    best_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        proj_head.train()
        epoch_loss = 0.0

        with tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}", unit="batch") as pbar:
            for imgs, labels in pbar:
                imgs = imgs.to(DEVICE)

                with torch.amp.autocast('cuda', enabled=(DEVICE == "cuda")):
                    feats = model(imgs)                        # (N, EMBED_DIM)
                    feats = F.normalize(feats, dim=-1)
                    z = proj_head(feats)                       # (N, PROJ_DIM)
                    loss = hierarchical_supcon_loss(z, list(labels), TEMPERATURE)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                epoch_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}",
                                 lr=f"{scheduler.get_last_lr()[0]:.2e}")

        avg_loss = epoch_loss / len(loader)
        lr_now   = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch} — avg loss: {avg_loss:.4f}  lr: {lr_now:.2e}")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{avg_loss:.6f}", f"{lr_now:.2e}"])

        # save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "proj_head_state": proj_head.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loss": best_loss,
            }
            torch.save(ckpt, CHECKPOINT_DIR / "best.pt")
            print(f"  ✓ Saved best checkpoint (loss={best_loss:.4f})")

        # save latest checkpoint every 5 epochs
        if epoch % 5 == 0 and ckpt is not None:
            torch.save(ckpt, CHECKPOINT_DIR / f"epoch_{epoch:03d}.pt")

    print(f"\nDone. Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to {CHECKPOINT_DIR}")


if __name__ == "__main__":
    train()