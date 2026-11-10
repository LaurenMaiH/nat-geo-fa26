#!/usr/bin/env python3
"""
Does retraining on reviewers' names make the suggestions better?

Simulates reviewers with the expert-labeled crops, using the app's own code
(feedback.pick_representatives, classifier_training.train_bundle,
classifier.suggest_for_tracks):

  for each mission M (the "new mission"):
    base model  = trained on the other missions
    M's deployments are split in half: "reviewed" and "test"
    reviewed crops are grouped by their true genus (or family when there's no
    genus) into stand-in clusters; up to 20 representatives each become feedback
    retrained     = base data + feedback
    both are scored on M's test half, which no model ever sees

Scenarios: a fraction of the clusters named (25/50/100%), impure clusters (10%
of each cluster's crops are other animals), and a careless reviewer (20% of the
names wrong). Repeated over several random splits and averaged.

  python eval_feedback.py [--labels ...] [--embeddings ...] [--seeds 3]
"""

import argparse
import contextlib
import io
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "application"))
import classifier  # noqa: E402
from classifier_training import find_key_col, l2_rows, train_bundle  # noqa: E402
from feedback import pick_representatives  # noqa: E402

CUTOFF = 0.7
SCENARIOS = [  # name, fraction of clusters named, impurity, wrong-name rate
    ("base (no feedback)", 0.0, 0.0, 0.0),
    ("25% of clusters named", 0.25, 0.0, 0.0),
    ("50% of clusters named", 0.5, 0.0, 0.0),
    ("100% of clusters named", 1.0, 0.0, 0.0),
    ("100%, impure clusters (10%)", 1.0, 0.1, 0.0),
    ("100%, 20% of names wrong", 1.0, 0.0, 0.2),
]


def blank(v):
    return pd.isna(v) or str(v).strip() == ""


def make_feedback(rev, emb, frac, impurity, wrong, rng, g2f_all):
    """Stand-in clusters from the reviewed crops → (rows, emb, names given)."""
    rev = rev.copy()
    rev["_name"] = [f if blank(g) else g for g, f in zip(rev["genus"], rev["family"])]
    names = sorted(rev["_name"].unique())
    named = [n for n in names if rng.random() < frac] if frac < 1 else names
    genera = sorted(g2f_all)
    rows, fb_emb, given = [], {}, set()
    for n in named:
        keys = list(rev.loc[rev["_name"] == n, "_key"])
        if impurity:
            others = list(rev.loc[rev["_name"] != n, "_key"])
            k = int(round(len(keys) * impurity / (1 - impurity)))
            if others and k:
                keys += list(rng.choice(others, size=min(k, len(others)), replace=False))
        row = rev[rev["_name"] == n].iloc[0]
        family, genus = row["family"], ("" if blank(row["genus"]) else row["genus"])
        if wrong and rng.random() < wrong:   # careless reviewer: a different genus
            genus = rng.choice([g for g in genera if g != genus])
            family = g2f_all[genus]
        given.add(genus or family)
        for key, _ in pick_representatives(keys, emb):
            k = f"fb/{n}/{key}"
            rows.append({"_key": k, "family": family, "genus": genus, "mission": "feedback"})
            fb_emb[k] = emb[key]
    return rows, fb_emb, given


def load_bundle(bundle):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(bundle, f)
    with contextlib.redirect_stdout(io.StringIO()):
        b = classifier.load_classifier(Path(f.name))
    classifier._cache.pop(f.name, None)
    Path(f.name).unlink()
    return b


def score(bundle, test, emb, fed):
    """Per test crop, what the app would show."""
    out = []
    for _, r in test.iterrows():
        s = classifier.suggest_for_tracks([r["_key"]], emb, bundle, CUTOFF)
        true_g = None if blank(r["genus"]) else r["genus"]
        true_f = None if blank(r["family"]) else r["family"]
        pred_g = s["genus"][0][0] if s.get("genus") else None
        pred_f = s["family"][0][0] if s.get("family") else None
        shown = s["label"] != "unknown"
        right = shown and ((s["level"] == "genus" and s["label"] == true_g) or
                           (s["level"] == "family" and s["label"] == true_f))
        out.append({"genus_ok": pred_g == true_g if true_g else np.nan,
                    "family_ok": pred_f == true_f if true_f else np.nan,
                    "shown": shown, "shown_ok": right if shown else np.nan,
                    "fed": (true_g or true_f) in fed})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).parent / "checkpoints" / "labeled_crops"
    ap.add_argument("--labels", default=str(root / "image_labels.csv"))
    ap.add_argument("--embeddings", default=str(root / "images_embeddings.json"))
    ap.add_argument("--seeds", type=int, default=3, help="random deployment splits per mission")
    args = ap.parse_args()

    emb = json.loads(Path(args.embeddings).read_text())
    labels = pd.read_csv(args.labels)
    key_col = find_key_col(labels, emb, log=None)
    labels = labels[labels[key_col].astype(str).isin(set(emb))].copy()
    labels["_key"] = labels[key_col].astype(str)
    keys = list(labels["_key"])
    X = l2_rows(np.asarray([emb[k] for k in keys], dtype=np.float32))
    emb = dict(zip(keys, X.tolist()))
    pairs = labels[~labels["genus"].map(blank) & ~labels["family"].map(blank)]
    g2f_all = pairs.groupby("genus")["family"].agg(lambda s: s.mode().iloc[0]).to_dict()

    results = defaultdict(list)
    for mission in sorted(labels["mission"].unique()):
        other = labels[labels["mission"] != mission]
        this = labels[labels["mission"] == mission]
        deps = sorted(this["deployment"].unique())
        for seed in range(args.seeds):
            rng = np.random.default_rng(seed)
            reviewed_deps = set(rng.permutation(deps)[: len(deps) // 2])
            rev = this[this["deployment"].isin(reviewed_deps)]
            test = this[~this["deployment"].isin(reviewed_deps)]
            print(f"{mission} split {seed}: {len(rev)} reviewed crops, {len(test)} test crops", flush=True)
            for name, frac, impurity, wrong in SCENARIOS:
                rows, fb_emb, fed = make_feedback(rev, emb, frac, impurity, wrong,
                                                  np.random.default_rng(seed + 100), g2f_all)
                train = pd.concat([other, pd.DataFrame(rows)], ignore_index=True) if rows else other
                bundle, _ = train_bundle(train, {**emb, **fb_emb}, min_prob=CUTOFF, evaluate=False, log=None)
                df = score(load_bundle(bundle), test, emb, fed)
                df["mission"] = mission
                results[name].append(df)

    print(f"\nScored on the held-out half of each mission ({args.seeds} splits x 4 missions). "
          f"'shown' = suggestion at the {CUTOFF} cutoff.")
    print(f"'named animals' = test animals whose genus/family a reviewer named in the other half.\n")
    hdr = f"{'scenario':<30} {'genus acc':>9} {'family acc':>10} {'shown':>6} {'shown & right':>13}  " \
          f"{'genus acc: named / not named animals':>36}"
    print(hdr)
    print("-" * len(hdr))
    # Same test crops in the same order in every scenario; "named" = named in the 100% scenario
    fed_df = pd.concat(results["100% of clusters named"], ignore_index=True)["fed"]
    for name, *_ in SCENARIOS:
        df = pd.concat(results[name], ignore_index=True)
        g_fed = df.loc[fed_df.values, "genus_ok"].mean()
        g_not = df.loc[~fed_df.values, "genus_ok"].mean()
        print(f"{name:<30} {df['genus_ok'].mean():>9.3f} {df['family_ok'].mean():>10.3f} "
              f"{df['shown'].mean():>6.1%} {df['shown_ok'].mean():>13.3f}  {g_fed:>17.3f} / {g_not:.3f}")

    print("\nPer mission, genus accuracy (base → 100% named):")
    base = pd.concat(results[SCENARIOS[0][0]], ignore_index=True)
    full = pd.concat(results["100% of clusters named"], ignore_index=True)
    for m in sorted(base["mission"].unique()):
        print(f"  {m}: {base.loc[base.mission == m, 'genus_ok'].mean():.3f} → "
              f"{full.loc[full.mission == m, 'genus_ok'].mean():.3f}")


if __name__ == "__main__":
    main()
