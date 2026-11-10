#!/usr/bin/env python3
"""
Where does the cluster name classifier do well or badly, and does augmentation help?

Leave-one-mission-out on the labeled crops with the app's training setup
(classifier_training.fit). Every original crop of the held-out mission is a test
crop, including animals the model has no class for (they count as wrong).
Augmented crops are only ever training data.

Reports per level: overall accuracy and top-3, accuracy at the app's 0.7 cutoff,
accuracy by how many training crops the true class has (rare vs common), and
per-class recall. With --aug-labels/--aug-embeddings it runs with and without the
augmented crops side by side.

  python eval_classifier.py
  python eval_classifier.py --aug-labels checkpoints/augmented_experiment/all/augmented_labels.csv \
      --aug-embeddings checkpoints/augmented_experiment/all/augmented_embeddings.json [--max-copies 1]
"""

import argparse
import json
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "application"))
from classifier_training import C, CLASS_WEIGHT, find_key_col, fit, is_augmented, l2_rows  # noqa: E402

CUTOFF = 0.75   # the model's min_prob (the app's default cutoff)
BUCKETS = [0, 9, 19, 49, 10**6]
BUCKET_NAMES = ["5-9", "10-19", "20-49", "50+"]


def load(labels_path, emb_path):
    emb = json.loads(Path(emb_path).read_text())
    labels = pd.read_csv(labels_path)
    key_col = find_key_col(labels, emb, log=None)
    labels = labels[labels[key_col].astype(str).isin(set(emb))].copy()
    labels["_key"] = labels[key_col].astype(str)
    return labels, emb


def lomo(labels, emb, level, min_per_class, **fit_kw):
    """One row per original test crop: true/pred class, confidence, top-3 hit, # training originals."""
    has = labels[level].notna() & (labels[level].astype(str).str.strip() != "")
    aug = is_augmented(labels)
    counts = labels.loc[has & ~aug, level].value_counts()
    keep = set(counts[counts >= min_per_class].index)
    X = l2_rows(np.asarray([emb[k] for k in labels["_key"]], dtype=np.float32))
    rows = []
    for m in sorted(labels.loc[~aug, "mission"].unique()):
        train = (has & labels[level].isin(keep) & (labels["mission"] != m)).to_numpy()
        test = (has & ~aug & (labels["mission"] == m)).to_numpy()
        clf = fit(X[train], labels.loc[train, level].astype(str).to_numpy(), **fit_kw)
        P = clf.predict_proba(X[test])
        order = np.argsort(P, axis=1)[:, ::-1]
        n_orig = Counter(labels.loc[train & ~aug.to_numpy(), level])
        for i, t in enumerate(labels.loc[test, level].astype(str)):
            rows.append({"mission": m, "true": t, "pred": clf.classes_[order[i, 0]], "p": P[i, order[i, 0]],
                         "top3": t in clf.classes_[order[i, :3]], "in_model": t in set(clf.classes_),
                         "n_train": n_orig.get(t, 0)})
    d = pd.DataFrame(rows)
    d["ok"] = d["true"] == d["pred"]
    return d


def summary(d):
    p = d[d["in_model"]]
    shown = d["p"] >= CUTOFF
    rare = p["n_train"] < 50
    return {
        "accuracy": d["ok"].mean(),
        "top-3": d["top3"].mean(),
        "balanced (avg per class)": p.groupby("true")["ok"].mean().mean(),
        f"shown at {CUTOFF}": shown.mean(),
        f"right when shown": d.loc[shown, "ok"].mean(),
        "rare classes (<50 crops)": p.loc[rare, "ok"].mean(),
        "common classes (50+)": p.loc[~rare, "ok"].mean(),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).parent / "checkpoints" / "labeled_crops"
    ap.add_argument("--labels", default=str(root / "image_labels.csv"))
    ap.add_argument("--embeddings", default=str(root / "images_embeddings.json"))
    ap.add_argument("--aug-labels")
    ap.add_argument("--aug-embeddings")
    ap.add_argument("--max-copies", type=int,
                    help="use only the first N augmented copies of each crop (keys end in __aug0, __aug1, ...)")
    ap.add_argument("--levels", default="family,genus")
    ap.add_argument("--min-per-class", type=int, default=5)
    ap.add_argument("--class-weight", choices=["none", "balanced"], default="none" if CLASS_WEIGHT is None else "balanced")
    ap.add_argument("--C", type=float, default=C)
    ap.add_argument("--per-class", action="store_true", help="print every class's recall")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    fit_kw = {"class_weight": None if args.class_weight == "none" else "balanced", "C": args.C}

    labels, emb = load(args.labels, args.embeddings)
    runs = {"original": (labels, emb)}
    if args.aug_labels:
        aug_labels, aug_emb = load(args.aug_labels, args.aug_embeddings)
        if args.max_copies is not None:
            copy = aug_labels["_key"].str.rsplit("__aug", n=1).str[1].astype(int)
            aug_labels = aug_labels[copy < args.max_copies]
        runs["+ augmented"] = (pd.concat([labels, aug_labels], ignore_index=True), {**emb, **aug_emb})
        print(f"{len(aug_labels)} augmented crops of {aug_labels['augmented_from'].nunique()} originals")
    print(f"{len(labels)} original crops · class_weight={fit_kw['class_weight']} C={fit_kw['C']} · "
          f"leave-one-mission-out, all original crops of the held-out mission are test crops")

    for level in [l.strip() for l in args.levels.split(",") if l.strip()]:
        res = {name: lomo(lab, e, level, args.min_per_class, **fit_kw) for name, (lab, e) in runs.items()}
        d0 = next(iter(res.values()))
        print(f"\n{'=' * 78}\n{level.upper()}: {len(d0)} test crops, "
              f"{(~d0['in_model']).mean():.1%} of them animals the model has no class for\n{'=' * 78}")
        table = pd.DataFrame({name: summary(d) for name, d in res.items()})
        print(table.to_string(float_format=lambda v: f"{v:.3f}"))

        print("\naccuracy by # training crops of the true class:")
        by = {}
        for name, d in res.items():
            p = d[d["in_model"]].copy()
            p["bucket"] = pd.cut(p["n_train"], BUCKETS, labels=BUCKET_NAMES)
            by[name] = p.groupby("bucket", observed=False)["ok"].mean()
            by["n test crops"] = p.groupby("bucket", observed=False)["ok"].size()
        print(pd.DataFrame(by).to_string(float_format=lambda v: f"{v:.3f}"))

        rec = pd.DataFrame({name: d[d["in_model"]].groupby("true")["ok"].mean() for name, d in res.items()})
        rec["n test"] = d0[d0["in_model"]].groupby("true").size()
        rec["n train"] = d0[d0["in_model"]].groupby("true")["n_train"].mean().round().astype(int)
        rec = rec.sort_values("n train")
        if args.per_class:
            print("\nrecall per class (fewest training crops first):")
            print(rec.to_string(float_format=lambda v: f"{v:.2f}"))
        elif len(runs) > 1:
            diff = rec["+ augmented"] - rec["original"]
            print(f"\nper class: {(diff > 0).sum()} improved, {(diff < 0).sum()} got worse, "
                  f"{(diff == 0).sum()} unchanged (--per-class for the full list)")


if __name__ == "__main__":
    main()
