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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent / "application"))
from classifier import checkpoint_fingerprint, load_classifier, _predict_proba  # noqa: E402

CUTOFFS = [0.3, 0.5, 0.7]


def l2_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(n == 0, 1, n)


def find_key_col(labels: pd.DataFrame, emb: dict, key_col=None):
    """The labels column whose values are the embeddings' keys."""
    if key_col:
        return key_col
    keys = set(emb)
    best, best_hits = None, 0
    for c in labels.columns:
        hits = labels[c].astype(str).isin(keys).sum()
        if hits > best_hits:
            best, best_hits = c, hits
    if best is None:
        sys.exit("No column of the labels CSV matches the embedding keys; pass --key-col.")
    print(f"join: labels column '{best}' ↔ embedding keys ({best_hits}/{len(labels)} rows matched)")
    return best


def fit(X, y):
    # Same setup as eval_centroids.py's logreg (reproduces the notebook's numbers). Without
    # standardising, L2-normalised features are so small that C=1 crushes the weights.
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=3000, class_weight="balanced")).fit(X, y)


def export_linear(clf):
    """Fold the scaler into the weights: W·((x-μ)/σ) + b = (W/σ)·x + (b - W·μ/σ)."""
    scaler, lr = clf.named_steps["standardscaler"], clf.named_steps["logisticregression"]
    coef = lr.coef_ / scaler.scale_
    intercept = lr.intercept_ - coef @ scaler.mean_
    return {"classes": lr.classes_.tolist(),
            "coef": coef.astype(np.float32).tolist(),
            "intercept": intercept.astype(np.float32).tolist()}


def leave_one_group_out(X, y, groups):
    """Train on all groups but one, test on it. A test animal whose class never
    appears in the training groups counts as wrong (it can't be predicted)."""
    y_true, y_pred, top3_hit, max_p = [], [], [], []
    print(f"  {'held out':<10} {'n':>5} {'acc':>6}")
    for g in sorted(set(groups)):
        test = groups == g
        train = ~test
        if len(set(y[train])) < 2 or not test.any():
            continue
        clf = fit(X[train], y[train])
        P = clf.predict_proba(X[test])
        order = np.argsort(P, axis=1)[:, ::-1]
        pred = clf.classes_[order[:, 0]]
        top3 = [t in clf.classes_[o[:3]] for t, o in zip(y[test], order)]
        acc = float(np.mean(pred == y[test]))
        print(f"  {str(g):<10} {test.sum():>5} {acc:>6.3f}")
        y_true += list(y[test]); y_pred += list(pred); top3_hit += top3; max_p += list(P.max(axis=1))

    y_true, y_pred, max_p = np.array(y_true), np.array(y_pred), np.array(max_p)
    correct = y_true == y_pred
    print(f"  overall: accuracy {correct.mean():.3f}, balanced {balanced_accuracy_score(y_true, y_pred):.3f}, "
          f"top-3 {np.mean(top3_hit):.3f}")
    print("  if only suggestions above a cutoff are shown (the rest say 'unknown'):")
    for t in CUTOFFS:
        shown = max_p >= t
        acc = correct[shown].mean() if shown.any() else float("nan")
        print(f"    cutoff {t:.1f}: shown for {shown.mean():5.1%} of animals, correct {acc:.3f} of those")


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
    args = ap.parse_args()

    emb = json.loads(Path(args.embeddings).read_text())
    if not isinstance(emb, dict) or not emb:
        sys.exit("Embeddings JSON must be a non-empty {key: vector} object.")
    labels = pd.read_csv(args.labels)
    key_col = find_key_col(labels, emb, args.key_col)
    labels = labels[labels[key_col].astype(str).isin(set(emb))].copy()
    labels["_key"] = labels[key_col].astype(str)
    dim = len(next(iter(emb.values())))
    print(f"{len(labels)} labeled embeddings, dim {dim}")

    bundle = {
        "embedding_dim": dim,
        "min_prob": args.min_prob,
        "embedding": "BioCLIP-2 + fine-tuned best.pt",
        "trained_on": str(Path(args.labels).resolve()),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "min_per_class": args.min_per_class,
        "checkpoint_fingerprint": checkpoint_fingerprint(args.checkpoint) if args.checkpoint else None,
        "n_train": {},
        "levels": {},
    }
    if args.checkpoint and not bundle["checkpoint_fingerprint"]:
        print(f"WARNING: --checkpoint {args.checkpoint} not found; no fingerprint recorded.")

    for level in [l.strip() for l in args.levels.split(",") if l.strip()]:
        if level not in labels:
            print(f"\n{level}: no such column, skipped")
            continue
        df = labels[labels[level].notna() & (labels[level].astype(str).str.strip() != "")]
        counts = df[level].value_counts()
        keep = counts[counts >= args.min_per_class].index
        df = df[df[level].isin(keep)]
        if len(keep) < 2:
            print(f"\n{level}: fewer than 2 classes with {args.min_per_class}+ examples, skipped")
            continue

        X = l2_rows(np.asarray([emb[k] for k in df["_key"]], dtype=np.float32))
        y = df[level].astype(str).to_numpy()
        print(f"\n{level}: {len(df)} examples, {len(keep)} classes "
              f"(dropped {len(counts) - len(keep)} classes with < {args.min_per_class})")

        if args.group_col in df and df[args.group_col].nunique() >= 2:
            print(f"  leave-one-{args.group_col}-out:")
            leave_one_group_out(X, y, df[args.group_col].astype(str).to_numpy())
        else:
            print(f"  (no '{args.group_col}' column with 2+ values: skipping the held-out check)")

        clf = fit(X, y)
        bundle["levels"][level] = export_linear(clf)
        bundle["n_train"][level] = int(len(df))
        bundle.setdefault("_check", {})[level] = (X[:20], clf.predict_proba(X[:20]))

    if not bundle["levels"]:
        sys.exit("Nothing trained.")

    # Each genus's family (the most common one in the labels), so the app can keep
    # family and genus consistent and add genus probabilities up per family.
    if "genus" in bundle["levels"] and {"genus", "family"} <= set(labels.columns):
        pairs = labels.dropna(subset=["genus", "family"])
        bundle["genus_to_family"] = {
            g: pairs.loc[pairs["genus"].astype(str) == g, "family"].astype(str).mode().iloc[0]
            for g in bundle["levels"]["genus"]["classes"]
            if (pairs["genus"].astype(str) == g).any()
        }
        clash = pairs.groupby(pairs["genus"].astype(str))["family"].nunique()
        if (clash > 1).any():
            print(f"WARNING: genera labeled with more than one family (most common used): "
                  f"{', '.join(clash[clash > 1].index)}")

    checks = bundle.pop("_check")
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
