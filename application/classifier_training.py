"""
Training for the cluster name classifier (classifier.py), shared by the offline
script (train_cluster_classifier.py) and the app's background retrainer (feedback.py).

A bundle is one logistic regression per level (family, genus) on L2-normalised
BioCLIP embeddings, exported as plain JSON weights.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CUTOFFS = [0.3, 0.5, 0.7]


def _quiet(*_args, **_kwargs):
    pass


def l2_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(n == 0, 1, n)


def find_key_col(labels, emb: dict, key_col=None, log=print):
    """The labels column whose values are the embeddings' keys."""
    log = log or _quiet
    if key_col:
        return key_col
    keys = set(emb)
    best, best_hits = None, 0
    for c in labels.columns:
        hits = labels[c].astype(str).isin(keys).sum()
        if hits > best_hits:
            best, best_hits = c, hits
    if best is None:
        raise ValueError("No column of the labels CSV matches the embedding keys; pass --key-col.")
    log(f"join: labels column '{best}' ↔ embedding keys ({best_hits}/{len(labels)} rows matched)")
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


def leave_one_group_out(X, y, groups, log=print):
    """Train on all groups but one, test on it. A test animal whose class never
    appears in the training groups counts as wrong (it can't be predicted)."""
    y_true, y_pred, top3_hit, max_p = [], [], [], []
    log(f"  {'held out':<10} {'n':>5} {'acc':>6}")
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
        log(f"  {str(g):<10} {test.sum():>5} {acc:>6.3f}")
        y_true += list(y[test]); y_pred += list(pred); top3_hit += top3; max_p += list(P.max(axis=1))

    y_true, y_pred, max_p = np.array(y_true), np.array(y_pred), np.array(max_p)
    correct = y_true == y_pred
    log(f"  overall: accuracy {correct.mean():.3f}, balanced {balanced_accuracy_score(y_true, y_pred):.3f}, "
        f"top-3 {np.mean(top3_hit):.3f}")
    log("  if only suggestions above a cutoff are shown (the rest say 'unknown'):")
    for t in CUTOFFS:
        shown = max_p >= t
        acc = correct[shown].mean() if shown.any() else float("nan")
        log(f"    cutoff {t:.1f}: shown for {shown.mean():5.1%} of animals, correct {acc:.3f} of those")


def train_bundle(labels, emb, levels=("family", "genus"), min_per_class=5, min_prob=0.7,
                 group_col="mission", evaluate=True, log=print):
    """
    Train one model per level. `labels` is a DataFrame with a "_key" column (the
    embedding key) plus the level columns; `emb` maps key → vector.

    Returns (bundle, checks): the JSON-ready bundle, and per level a few training
    rows with sklearn's probabilities so the caller can verify the export.
    """
    log = log or _quiet
    dim = len(emb[labels["_key"].iloc[0]])
    bundle = {"embedding_dim": dim, "min_prob": min_prob, "min_per_class": min_per_class,
              "n_train": {}, "levels": {}}
    checks = {}

    for level in levels:
        if level not in labels:
            log(f"\n{level}: no such column, skipped")
            continue
        df = labels[labels[level].notna() & (labels[level].astype(str).str.strip() != "")]
        counts = df[level].value_counts()
        keep = counts[counts >= min_per_class].index
        df = df[df[level].isin(keep)]
        if len(keep) < 2:
            log(f"\n{level}: fewer than 2 classes with {min_per_class}+ examples, skipped")
            continue

        X = l2_rows(np.asarray([emb[k] for k in df["_key"]], dtype=np.float32))
        y = df[level].astype(str).to_numpy()
        log(f"\n{level}: {len(df)} examples, {len(keep)} classes "
            f"(dropped {len(counts) - len(keep)} classes with < {min_per_class})")

        if evaluate:
            if group_col in df and df[group_col].nunique() >= 2:
                log(f"  leave-one-{group_col}-out:")
                leave_one_group_out(X, y, df[group_col].astype(str).to_numpy(), log)
            else:
                log(f"  (no '{group_col}' column with 2+ values: skipping the held-out check)")

        clf = fit(X, y)
        bundle["levels"][level] = export_linear(clf)
        bundle["n_train"][level] = int(len(df))
        checks[level] = (X[:20], clf.predict_proba(X[:20]))

    # Each genus's family (the most common one in the labels), so the app can keep
    # family and genus consistent and add genus probabilities up per family.
    if "genus" in bundle["levels"] and {"genus", "family"} <= set(labels.columns):
        pairs = labels[labels["genus"].notna() & labels["family"].notna()]
        pairs = pairs[(pairs["genus"].astype(str).str.strip() != "") & (pairs["family"].astype(str).str.strip() != "")]
        bundle["genus_to_family"] = {
            g: pairs.loc[pairs["genus"].astype(str) == g, "family"].astype(str).mode().iloc[0]
            for g in bundle["levels"]["genus"]["classes"]
            if (pairs["genus"].astype(str) == g).any()
        }
        clash = pairs.groupby(pairs["genus"].astype(str))["family"].nunique()
        if (clash > 1).any():
            log(f"WARNING: genera labeled with more than one family (most common used): "
                f"{', '.join(clash[clash > 1].index)}")

    return bundle, checks
