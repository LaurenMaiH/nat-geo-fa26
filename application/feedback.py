"""
Reviewer names → training examples → retrained name classifier.

Every cluster a reviewer has named (by accepting a suggestion or typing a name)
is snapshotted into a per-mission feedback file: the name plus up to
MAX_EXAMPLES representative track embeddings. A background Retrainer resolves
each name to a family/genus (the model's own classes first, then WoRMS), trains
on the base labeled crops + every feedback file, and writes a separate
retrained bundle that the app prefers over the base one. The base bundle is
never overwritten; deleting the retrained file goes back to it.

Feedback files are rebuilt from the current state.json, so labels follow the
cluster: moving tracks out, trashing them, or renaming the cluster updates them.
"""

import hashlib
import json
import os
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import classifier
from classifier_training import find_key_col, l2_rows, train_bundle

MAX_EXAMPLES = 20       # tracks per named cluster, so one big cluster can't drown the base data
NAMES_FILE = "_names.json"
DEBOUNCE_S = 5.0
_AUTO_NAME_RE = re.compile(r"^(Cluster \d+|Unassigned)$")


def is_training_name(name) -> bool:
    """A reviewer-given name (not the app's own "Cluster N")."""
    name = (name or "").strip()
    return bool(name) and not _AUTO_NAME_RE.match(name)


# ── Picking examples ──────────────────────────────────────────────────────────

def pick_representatives(keys, emb_dict, k=MAX_EXAMPLES, seed=0):
    """
    Up to k tracks that cover every look in the cluster, as [(key, group_size)].

    Isolated tracks (far from all the others, so most likely mis-assigned) are
    dropped first. K-means then splits the rest into k groups of similar tracks
    and the track nearest each group's center is picked, so near-duplicates (one
    animal filmed many times) collapse into one example and a minority look
    still gets its own. Deterministic for the same tracks.
    """
    keys = sorted(k_ for k_ in keys if k_ in emb_dict)
    if len(keys) <= k:
        return [(key, 1) for key in keys]
    X = l2_rows(np.asarray([emb_dict[key] for key in keys], dtype=np.float32))

    # A track is isolated when its nearest neighbour is much further than usual for this cluster.
    S = X @ X.T
    np.fill_diagonal(S, -np.inf)
    nn = S.max(axis=1)
    med = np.median(nn)
    mad = np.median(np.abs(nn - med)) * 1.4826
    keep = nn >= med - max(3 * mad, 0.05)
    if keep.sum() < k:
        keep = np.zeros(len(keys), bool)
        keep[np.argsort(nn)[::-1][:k]] = True
    keys = [key for key, kp in zip(keys, keep) if kp]
    X = X[keep]
    if len(keys) <= k:
        return [(key, 1) for key in keys]

    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(X)
    picks = []
    for j in range(k):
        idx = np.flatnonzero(km.labels_ == j)
        if len(idx) == 0:
            continue
        d = ((X[idx] - km.cluster_centers_[j]) ** 2).sum(axis=1)
        picks.append((keys[idx[d.argmin()]], int(len(idx))))
    return sorted(picks)


def _signature(track_keys):
    return hashlib.sha1("\n".join(sorted(track_keys)).encode()).hexdigest()[:16]


def _read_json(path, default):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def _write_json_atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, path)


def sync_feedback(state, emb_dict, path, deployment=None, mission=None) -> bool:
    """
    Rewrite a mission's feedback file from its state: every named, non-empty
    cluster with its representative examples. Clusters whose tracks haven't
    changed reuse their previous picks. Returns True if anything that affects
    training changed (a name, or a cluster's tracks).
    """
    path = Path(path)
    prev = _read_json(path, {}).get("clusters", {})
    clusters = {}
    for cid, c in state.get("clusters", {}).items():
        if cid == "noise" or not is_training_name(c.get("name")) or not c.get("track_keys"):
            continue
        sig = _signature(c["track_keys"])
        old = prev.get(cid)
        if old and old.get("signature") == sig:
            examples = old["examples"]
        else:
            examples = {key: {"weight": w, "embedding": [round(float(x), 6) for x in emb_dict[key]]}
                        for key, w in pick_representatives(c["track_keys"], emb_dict)}
        if examples:
            clusters[cid] = {"name": c["name"].strip(), "via": c.get("name_via", "rename"),
                             "signature": sig, "examples": examples}

    def fingerprint(cs):
        return {cid: (r["name"], r["signature"]) for cid, r in cs.items()}

    changed = fingerprint(clusters) != fingerprint(prev)
    if changed or (clusters and not path.exists()):
        if clusters:
            _write_json_atomic(path, {"deployment": deployment, "mission": mission,
                                      "updated_at": datetime.now().isoformat(timespec="seconds"),
                                      "clusters": clusters})
        elif path.exists():
            path.unlink()
    return changed


def feedback_files(feedback_dir):
    d = Path(feedback_dir)
    return sorted(p for p in d.glob("*.json") if p.name != NAMES_FILE) if d.exists() else []


# ── Names → taxa ──────────────────────────────────────────────────────────────

class LookupUnavailable(Exception):
    """WoRMS couldn't be reached; try the name again on the next retrain."""


def resolve_label(name, bundle, lookup=None):
    """
    The family/genus a reviewer's name stands for, or None if it isn't one
    (e.g. "Shadows", or a rank above family like "Decapoda").

    The model's own classes are matched first (no network). Otherwise
    lookup(name) returns a WoRMS lineage [{"rank", "name"}, ...] or None;
    it may raise LookupUnavailable.
    """
    n = " ".join((name or "").split())
    if not n:
        return None
    if bundle is not None:
        g2f = bundle.get("genus_to_family", {})
        genera = {c.lower(): c for c in bundle["levels"].get("genus", {}).get("classes", [])}
        families = {c.lower(): c for c in bundle["levels"].get("family", {}).get("classes", [])}
        families.update({f.lower(): f for f in g2f.values()})
        if n.lower() in genera:
            g = genera[n.lower()]
            return {"genus": g, "family": g2f.get(g), "source": "model"}
        if n.lower() in families:
            return {"genus": None, "family": families[n.lower()], "source": "model"}
    if lookup is None:
        raise LookupUnavailable("no taxonomy lookup configured")
    lineage = lookup(n[0].upper() + n[1:])
    if not lineage:
        return None
    ranks = {(r.get("rank") or "").lower(): r.get("name") for r in lineage}
    if ranks.get("genus") and ranks.get("family"):
        return {"genus": ranks["genus"], "family": ranks["family"], "source": "worms"}
    if ranks.get("family"):
        return {"genus": None, "family": ranks["family"], "source": "worms"}
    return None


def worms_lookup(cache_path):
    """lineage_by_name from worms_taxonomy.py (repo root), with network errors as LookupUnavailable."""
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from worms_taxonomy import Worms

    worms = Worms(cache_path)

    def lookup(name):
        try:
            lineage = worms.lineage_by_name(name)
        except Exception as e:  # noqa: BLE001  (network / HTTP errors)
            raise LookupUnavailable(f"{type(e).__name__}: {e}") from e
        worms.save()
        return lineage
    return lookup


def resolve_names(feedback_dir, bundle, lookup):
    """
    Resolve every name in the feedback files that isn't cached yet, and save the
    cache (feedback_dir/_names.json). Returns (names, pending): names maps
    name → {"family", "genus", "source"} or None; pending lists names WoRMS
    couldn't be asked about this time.
    """
    cache_path = Path(feedback_dir) / NAMES_FILE
    names = _read_json(cache_path, {})
    wanted = {rec["name"] for p in feedback_files(feedback_dir)
              for rec in _read_json(p, {}).get("clusters", {}).values()}
    pending, changed = [], False
    for name in sorted(wanted - set(names)):
        try:
            names[name] = resolve_label(name, bundle, lookup)
            changed = True
        except LookupUnavailable:
            pending.append(name)
    if changed:
        _write_json_atomic(cache_path, names)
    return names, pending


def collect_examples(feedback_dir, names, dim):
    """Training rows from the feedback files: ([{"_key", "family", "genus", "mission"}], {key: vector})."""
    rows, emb = [], {}
    for p in feedback_files(feedback_dir):
        data = _read_json(p, {})
        group = f"feedback:{data.get('deployment')}/{data.get('mission') or p.stem}"
        for cid, rec in data.get("clusters", {}).items():
            taxon = names.get(rec["name"])
            if not taxon or not taxon.get("family"):
                continue
            for key, ex in rec["examples"].items():
                if len(ex["embedding"]) != dim:
                    continue
                k = f"{group}/{cid}/{key}"
                rows.append({"_key": k, "family": taxon["family"], "genus": taxon.get("genus") or "",
                             "mission": group})
                emb[k] = ex["embedding"]
    return rows, emb


# ── Background retraining ─────────────────────────────────────────────────────

class Retrainer:
    """
    Retrains on base data + feedback in a daemon thread, DEBOUNCE_S after the last
    schedule() call, one run at a time. Never touches state.json: the UI sees the
    new `version` in status() and asks the app to refresh suggestions.

    config() returns the paths to use (read at run time):
      base_model, retrained_model, feedback_dir, base_labels, base_embeddings, worms_cache
    """

    def __init__(self, config, auto=True, debounce=DEBOUNCE_S):
        self.config = config
        self.auto = auto
        self.debounce = debounce
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._due = None
        self._thread = None
        self._base_cache = None
        self._status = {"state": "idle", "version": None, "message": "", "n_feedback": 0,
                        "n_clusters": 0, "new_classes": [], "pending_names": []}

    def status(self):
        with self._lock:
            s = dict(self._status)
        cfg = self.config()
        retrained = classifier.load_classifier(cfg["retrained_model"]) if Path(cfg["retrained_model"]).exists() else None
        if s["version"] is None and retrained:
            s["version"] = retrained["meta"].get("version")
        s["names"] = _read_json(Path(cfg["feedback_dir"]) / NAMES_FILE, {})
        return s

    def _set(self, **kw):
        with self._lock:
            self._status.update(kw)

    def schedule(self):
        self._set(state="pending", message="Waiting for more edits…")
        if not self.auto:
            return
        with self._lock:
            self._due = time.monotonic() + self.debounce
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._loop, daemon=True, name="classifier-retrain")
                self._thread.start()
        self._wake.set()

    def _loop(self):
        while True:
            self._wake.wait()
            while True:
                with self._lock:
                    wait = self._due - time.monotonic()
                if wait <= 0:
                    break
                time.sleep(min(wait, 1.0))
            self._wake.clear()
            try:
                self.run_once()
            except Exception as e:  # noqa: BLE001  (keep the thread alive; report it)
                self._set(state="error", message=f"Retraining failed: {type(e).__name__}: {e}")
                print(f"Classifier retraining failed: {type(e).__name__}: {e}")
            if self._wake.is_set():   # more edits arrived while training
                self._set(state="pending")

    def _load_base(self, cfg):
        import pandas as pd

        labels_p, emb_p = Path(cfg["base_labels"]), Path(cfg["base_embeddings"])
        if not labels_p.exists() or not emb_p.exists():
            return None
        stamp = (str(labels_p), labels_p.stat().st_mtime, str(emb_p), emb_p.stat().st_mtime)
        if self._base_cache and self._base_cache[0] == stamp:
            return self._base_cache[1]
        emb = json.loads(emb_p.read_text())
        labels = pd.read_csv(labels_p)
        key_col = find_key_col(labels, emb, log=None)
        labels = labels[labels[key_col].astype(str).isin(set(emb))].copy()
        labels["_key"] = labels[key_col].astype(str)
        self._base_cache = (stamp, (labels, emb))
        return labels, emb

    def run_once(self):
        """Resolve names, retrain, and swap in the new model. Returns the new version or None."""
        import pandas as pd

        cfg = self.config()
        self._set(state="training", message="Retraining…")
        base_model = classifier.load_classifier(cfg["base_model"])
        lookup = worms_lookup(cfg["worms_cache"]) if cfg.get("worms_cache") else None
        names, pending = resolve_names(cfg["feedback_dir"], base_model, lookup)

        base = self._load_base(cfg)
        if base is None:
            self._set(state="error", pending_names=pending,
                      message=f"Labels saved, but the model can't retrain: base training data not found "
                              f"({cfg['base_labels']}, {cfg['base_embeddings']}).")
            return None
        labels, emb = base
        dim = len(next(iter(emb.values())))
        rows, fb_emb = collect_examples(cfg["feedback_dir"], names, dim)
        retrained_path = Path(cfg["retrained_model"])

        if not rows:
            if retrained_path.exists():
                retrained_path.unlink()
            classifier._cache.pop(str(retrained_path), None)
            self._set(state="idle", version=None, n_feedback=0, n_clusters=0, new_classes=[],
                      pending_names=pending, message="No usable labels yet: using the original model.")
            return None

        all_labels = pd.concat([labels, pd.DataFrame(rows)], ignore_index=True)
        all_emb = {**emb, **fb_emb}
        min_prob = base_model["min_prob"] if base_model else 0.7
        bundle, _ = train_bundle(all_labels, all_emb, min_prob=min_prob, evaluate=False, log=None)

        base_classes = set()
        if base_model:
            for m in base_model["levels"].values():
                base_classes |= set(m["classes"])
        new_classes = sorted({c for m in bundle["levels"].values() for c in m["classes"]} - base_classes)
        version = f"{time.time_ns()}"
        n_clusters = len({r["_key"].rsplit("/", 1)[0] for r in rows})
        bundle.update({
            "embedding": "BioCLIP-2 + fine-tuned best.pt",
            "trained_on": f"{cfg['base_labels']} + {n_clusters} reviewer-named clusters",
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "checkpoint_fingerprint": (base_model or {}).get("meta", {}).get("checkpoint_fingerprint"),
            "version": version, "n_feedback": len(rows), "n_feedback_clusters": n_clusters,
            "new_classes": new_classes,
        })
        _write_json_atomic(retrained_path, bundle)
        classifier._cache.pop(str(retrained_path), None)
        msg = f"Model updated from {n_clusters} named cluster(s)"
        if new_classes:
            msg += f" · new: {', '.join(new_classes)}"
        if pending:
            msg += f" · couldn't look up yet: {', '.join(pending)}"
        self._set(state="idle", version=version, n_feedback=len(rows), n_clusters=n_clusters,
                  new_classes=new_classes, pending_names=pending, message=msg)
        print(msg)
        return version
