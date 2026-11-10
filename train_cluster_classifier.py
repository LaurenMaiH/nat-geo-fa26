#!/usr/bin/env python3
"""
Train the logistic regression that suggests a family/genus for each cluster in
the review app (application/classifier.py).

Inputs are the files embed_csv_images.py writes for the labeled crops
(see the labeled_crops_eval notebook), e.g. on Drive:
  Matewos's Notebook/results/labeled_crops/image_labels.csv
  Matewos's Notebook/results/labeled_crops/images_embeddings.json

They must be embedded with the SAME fine-tuned BioCLIP (checkpoints/best.pt) the
pipeline uses, or the suggestions won't mean anything.

--feedback-dir adds the clusters reviewers named in the app (checkpoints/feedback/),
the same data the app retrains on in the background; the leave-one-mission-out
check then also scores the model on each reviewed mission.

Output: checkpoints/cluster_classifier.json (class names + weights per level, no pickle).
Runs on CPU in well under a minute.

  python train_cluster_classifier.py --labels .../image_labels.csv \
      --embeddings .../images_embeddings.json --checkpoint checkpoints/best.pt
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "application"))
from classifier import checkpoint_fingerprint, load_classifier, _predict_proba  # noqa: E402
from classifier_training import find_key_col, train_bundle  # noqa: E402



def _display_path(path):
    """Path relative to the repo when inside it, else just the file name (no local user paths in the bundle)."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(Path(__file__).resolve().parent))
    except ValueError:
        return path.name


def add_augmented(labels, emb, aug_labels, aug_embeddings):
    """Append augmented crops (rows with augmented_from) to the labels and embeddings."""
    aug_emb = json.loads(Path(aug_embeddings).read_text())
    aug = pd.read_csv(aug_labels)
    aug = aug[aug[find_key_col(aug, aug_emb, log=None)].astype(str).isin(set(aug_emb))].copy()
    aug["_key"] = aug[find_key_col(aug, aug_emb, log=None)].astype(str)
    print(f"+ {len(aug)} augmented crops of {aug['augmented_from'].nunique()} originals")
    return pd.concat([labels, aug], ignore_index=True), {**emb, **aug_emb}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True, help="image_labels.csv from embed_csv_images.py")
    ap.add_argument("--embeddings", required=True, help="images_embeddings.json from embed_csv_images.py")
    ap.add_argument("--out", default=str(Path(__file__).parent / "checkpoints" / "cluster_classifier.json"))
    ap.add_argument("--levels", default="family,genus")
    ap.add_argument("--min-per-class", type=int, default=5, help="classes with fewer examples are left out")
    ap.add_argument("--group-col", default="mission", help="column for the leave-one-group-out check")
    ap.add_argument("--min-prob", type=float, default=0.7, help="default confidence cutoff stored in the bundle")
    ap.add_argument("--key-col", help="labels column holding the embedding keys (auto-detected if omitted)")
    ap.add_argument("--checkpoint", help="the best.pt the embeddings came from; recorded so the app can warn on a mismatch")
    ap.add_argument("--aug-labels", help="augmented_labels.csv from augment_crops_colab.ipynb")
    ap.add_argument("--aug-embeddings", help="augmented_embeddings.json from the same notebook")
    ap.add_argument("--class-weight", choices=["none", "balanced"], default="none",
                    help="'balanced' weights rare classes up: better on rare animals, worse overall")
    ap.add_argument("--C", type=float, default=0.3, help="inverse regularisation strength")
    ap.add_argument("--feedback-dir", help="also train on reviewer-named clusters (the app's checkpoints/feedback/)")
    args = ap.parse_args()

    emb = json.loads(Path(args.embeddings).read_text())
    if not isinstance(emb, dict) or not emb:
        sys.exit("Embeddings JSON must be a non-empty {key: vector} object.")
    labels = pd.read_csv(args.labels)
    try:
        key_col = find_key_col(labels, emb, args.key_col)
    except ValueError as e:
        sys.exit(str(e))
    labels = labels[labels[key_col].astype(str).isin(set(emb))].copy()
    labels["_key"] = labels[key_col].astype(str)
    dim = len(next(iter(emb.values())))
    print(f"{len(labels)} labeled embeddings, dim {dim}")

    trained_on = _display_path(args.labels)
    if bool(args.aug_labels) != bool(args.aug_embeddings):
        sys.exit("--aug-labels and --aug-embeddings go together.")
    if args.aug_labels:
        labels, emb = add_augmented(labels, emb, args.aug_labels, args.aug_embeddings)
        trained_on += f" + {_display_path(args.aug_labels)}"
    if args.feedback_dir:
        import feedback

        fb_dir = Path(args.feedback_dir)
        current = load_classifier(args.out) if Path(args.out).exists() else None   # names it already knows
        names, pending = feedback.resolve_names(
            fb_dir, current, feedback.worms_lookup(fb_dir.parent / "worms_cache.json"))
        rows, fb_emb = feedback.collect_examples(fb_dir, names, dim)
        print(f"+ {len(rows)} examples from reviewer-named clusters in {fb_dir}")
        if pending:
            print(f"  (couldn't look up in WoRMS, skipped: {', '.join(pending)})")
        if rows:
            labels = pd.concat([labels, pd.DataFrame(rows)], ignore_index=True)
            emb = {**emb, **fb_emb}
            trained_on += f" + {_display_path(fb_dir)}"

    bundle, checks = train_bundle(
        labels, emb, levels=[l.strip() for l in args.levels.split(",") if l.strip()],
        min_per_class=args.min_per_class, min_prob=args.min_prob, group_col=args.group_col,
        class_weight=None if args.class_weight == "none" else args.class_weight, C=args.C)
    if not bundle["levels"]:
        sys.exit("Nothing trained.")

    fingerprint = checkpoint_fingerprint(args.checkpoint) if args.checkpoint else None
    if args.checkpoint and not fingerprint:
        print(f"WARNING: --checkpoint {args.checkpoint} not found; no fingerprint recorded.")
    bundle.update({
        "embedding": "BioCLIP-2 + fine-tuned best.pt",
        "trained_on": trained_on,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint_fingerprint": fingerprint,
    })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bundle))

    # Round-trip: the app's loader must give the same probabilities as sklearn
    loaded = load_classifier(out)
    for level, (Xc, Pc) in checks.items():
        err = np.abs(_predict_proba(loaded["levels"][level], Xc) - Pc).max()
        assert err < 1e-3, f"{level}: exported model disagrees with sklearn (max diff {err:.2e})"
    print(f"\nSaved {out} ({out.stat().st_size / 1e6:.1f} MB); app loader matches sklearn ✓")


if __name__ == "__main__":
    main()
