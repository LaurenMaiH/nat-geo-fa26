"""
Family / genus name suggestions for clusters.

A logistic regression trained offline on labeled BioCLIP embeddings
(train_cluster_classifier.py) is stored as plain JSON: class names, weights
and biases per level. Predicting is a softmax over X·Wᵀ + b, so the app needs
neither scikit-learn nor pickle to use it.

Family and genus are kept consistent through the taxonomy: a confident genus
brings its own family, and an unsure genus model still gives a family when
its genera add up within one (Plesionika 0.4 + Heterocarpus 0.3 → Pandalidae 0.7).
The separate family model is only the last resort, for families without genus
labels. It's the weaker of the two on pipeline crops (rare families win under
balanced class weights), so it must never contradict a confident genus.

The suggestion never changes a cluster; the reviewer accepts it via /api/rename.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

TOP_K = 3

_cache: dict = {}


def checkpoint_fingerprint(path: Path):
    """Cheap id for a checkpoint file: its size + a hash of the first 16 MB (best.pt is ~3.6 GB)."""
    path = Path(path)
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(16 * 1024 * 1024))
    return f"{path.stat().st_size}:{h.hexdigest()[:16]}"


def load_classifier(path: Path, checkpoint_path: Path = None):
    """
    Load the bundle once and cache it. Returns None if the file is missing or
    unreadable, so the app works exactly as before without it.
    """
    path = Path(path)
    if str(path) in _cache:
        return _cache[str(path)]

    bundle = None
    if not path.exists():
        print(f"Cluster classifier not found at {path}: no name suggestions.")
    else:
        try:
            raw = json.loads(path.read_text())
            bundle = {
                "embedding_dim": int(raw["embedding_dim"]),
                "min_prob":      float(raw.get("min_prob", 0.5)),
                "levels": {
                    lvl: {
                        "classes":   list(m["classes"]),
                        "coef":      np.asarray(m["coef"], dtype=np.float32),
                        "intercept": np.asarray(m["intercept"], dtype=np.float32),
                    }
                    for lvl, m in raw["levels"].items()
                },
                "genus_to_family": dict(raw.get("genus_to_family") or {}),
                "meta": {k: v for k, v in raw.items() if k not in ("levels", "genus_to_family")},
            }
            print(f"Cluster classifier loaded from {path} "
                  f"({', '.join(f'{l}: {len(m['classes'])} classes' for l, m in bundle['levels'].items())})")

            trained_fp = raw.get("checkpoint_fingerprint")
            if trained_fp and checkpoint_path is not None:
                current_fp = checkpoint_fingerprint(checkpoint_path)
                if current_fp and current_fp != trained_fp:
                    print(f"WARNING: cluster classifier was trained on embeddings from a different "
                          f"BioCLIP checkpoint than {checkpoint_path}. Retrain it; suggestions may be wrong.")
        except Exception as e:
            print(f"Could not load cluster classifier {path} ({type(e).__name__}: {e}): no name suggestions.")
            bundle = None

    _cache[str(path)] = bundle
    return bundle


def _predict_proba(model, X):
    logits = X @ model["coef"].T + model["intercept"]
    if logits.shape[1] == 1:            # sklearn stores a 2-class model as a single column
        p1 = 1.0 / (1.0 + np.exp(-logits[:, 0]))
        return np.stack([1.0 - p1, p1], axis=1)
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def _top(classes, p):
    return [[classes[i], round(float(p[i]), 4)] for i in np.argsort(p)[::-1][:TOP_K]]


def suggest_for_tracks(track_keys, emb_dict, bundle, min_prob=None):
    """
    Suggest a name for a group of tracks.

    Each track is classified separately and the probabilities are averaged, which
    is steadier than classifying the centroid alone. In order, the first that
    reaches min_prob wins:
      0. the family model's top family, if it's one with no genus classes and the
         family model is surer than the genus model (the genus model can only answer
         with genera it knows, so it confidently misnames these)
      1. top genus                                → that genus (family from the taxonomy)
      2. top family of the summed genus probs     → that family
      3. top family of the family model           → that family
    otherwise "unknown".

    Returns None when there's no classifier, no embeddings, or the embeddings don't
    match what it was trained on (e.g. 512-d placeholder vectors).
    """
    if bundle is None:
        return None
    vecs = [emb_dict[k] for k in track_keys if k in emb_dict]
    if not vecs:
        return None
    X = np.asarray(vecs, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != bundle["embedding_dim"]:
        return None
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.where(norms == 0, 1, norms)

    min_prob = bundle["min_prob"] if min_prob is None else float(min_prob)
    g2f = bundle.get("genus_to_family", {})
    result = {"label": "unknown", "level": None, "prob": None, "family_name": None,
              "source": None, "n_tracks": len(vecs)}

    # (name, family, prob, level, source) candidates in priority order
    candidates = []
    genus_family = None
    gm = bundle["levels"].get("genus")
    if gm is not None:
        pg = _predict_proba(gm, X).mean(axis=0)
        result["genus"] = _top(gm["classes"], pg)
        name, p = result["genus"][0]
        candidates.append((name, g2f.get(name), p, "genus", "genus"))
        fams = [g2f.get(c) for c in gm["classes"]]
        if g2f and all(fams):
            names = sorted(set(fams))
            pf = np.array([sum(q for q, f in zip(pg, fams) if f == n) for n in names])
            genus_family = _top(names, pf)
            name, p = genus_family[0]
            candidates.append((name, name, p, "family", "genus_sum"))

    family_model = None
    fm = bundle["levels"].get("family")
    if fm is not None:
        family_model = _top(fm["classes"], _predict_proba(fm, X).mean(axis=0))
        name, p = family_model[0]
        candidates.append((name, name, p, "family", "family_model"))

    if family_model is not None and gm is not None and g2f:
        name, p = family_model[0]
        if name not in set(g2f.values()) and p >= result["genus"][0][1]:
            candidates.insert(0, (name, name, p, "family", "family_model"))

    for name, family, p, level, source in candidates:
        if p >= min_prob:
            result.update(label=name, family_name=family, prob=p, level=level, source=source)
            break

    # The family list shown next to the genus must agree with it, so it comes from the
    # genus model unless the family model is what made the suggestion.
    shown_family = family_model if (result["source"] == "family_model" or genus_family is None) else genus_family
    if shown_family is not None:
        result["family"] = shown_family
    return result
