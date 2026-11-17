"""
Reviewer names → training examples → retrained name classifier.

Every cluster a reviewer has named (by accepting a suggestion or typing a name)
is snapshotted into a per-mission feedback file: the name plus up to
MAX_EXAMPLES representative track embeddings. A background Retrainer resolves
each name to a family/genus (the model's own classes first, then WoRMS) and
trains a new model version on the base labeled crops + every feedback file,
starting from the current version's weights. Versions are saved in models/
(ModelStore) and the newest is used; v1 is the base model, which is never
overwritten.

Feedback files are rebuilt from the current state.json, so labels follow the
cluster: moving tracks out, trashing them, renaming the cluster, or removing it
from training (train_excluded) updates them. Every label change is logged in
_history.jsonl, which the Labels tab shows and undoes from.
"""

import hashlib
import json
import os
import re
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np

import classifier
from classifier_training import find_key_col, l2_rows, train_bundle

MAX_EXAMPLES = 20       # tracks per named cluster, so one big cluster can't drown the base data
NAMES_FILE = "_names.json"
HISTORY_FILE = "_history.jsonl"
KEEP_VERSIONS = 20      # saved versions besides v1 (the base model) and the current one
DEBOUNCE_S = 5.0
_EMPTY_HASH = hashlib.sha1(b"[]").hexdigest()[:16]   # data_hash of "no reviewer labels" (v1)
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
        if (cid == "noise" or not is_training_name(c.get("name")) or not c.get("track_keys")
                or c.get("train_excluded")):
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


def label_key(deployment, mission, cluster_id):
    return f"{deployment}/{mission}/{cluster_id}"


def feedback_labels(feedback_dir, names):
    """{label_key: name} for every feedback cluster whose name resolves to a taxon."""
    out = {}
    for p in feedback_files(feedback_dir):
        data = _read_json(p, {})
        for cid, rec in data.get("clusters", {}).items():
            if names.get(rec["name"]):
                out[label_key(data.get("deployment"), data.get("mission"), cid)] = rec["name"]
    return out


# ── History of label changes ──────────────────────────────────────────────────

def append_history(feedback_dir, **entry):
    """Append one event to _history.jsonl and return it (with its id and time)."""
    entry = {"id": uuid.uuid4().hex[:12], "at": datetime.now().isoformat(timespec="seconds"),
             "ts": time.time(), **entry}
    path = Path(feedback_dir) / HISTORY_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def read_history(feedback_dir):
    """All events, oldest first."""
    path = Path(feedback_dir) / HISTORY_FILE
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def describe(entry):
    """One line for a label event, as the Labels tab and model versions show it."""
    a, frm, to = entry.get("action"), entry.get("from"), entry.get("to")
    where = entry.get("cluster_id", "")
    if a in ("confirm", "rename", "unconfirm"):
        return f"{where}: {frm or '—'} → {to or '—'}"
    if a == "exclude":
        return f"{where}: \"{entry.get('name')}\" removed from training"
    if a == "include":
        return f"{where}: \"{entry.get('name')}\" used for training again"
    if a == "undo":
        return f"undo: {entry.get('summary', '')}"
    if a == "reset":
        return "back to the original model"
    if a == "mission_reset":
        return f"{entry.get('mission')} restarted: its labels removed"
    if a == "version":
        return entry.get("summary") or f"model v{to}"
    return a or ""


# ── Model versions ────────────────────────────────────────────────────────────

class ModelStore:
    """
    Saved model versions. v1 is the base model file itself (never copied or
    changed); retrains add models/v0002.json, v0003.json, ... and models/index.json
    lists every version and which one is current. Only the last KEEP_VERSIONS
    retrained files are kept on disk (their index entries stay).
    """

    def __init__(self, models_dir, base_model):
        self.dir = Path(models_dir)
        self.base = Path(base_model)
        self._lock = threading.Lock()

    def _index(self):
        idx = _read_json(self.dir / "index.json", None)
        if not idx:
            idx = {"current": 1, "versions": [{"version": 1, "parent": None, "file": None,
                                               "created_at": None, "reason": "original model",
                                               "changes": [], "n_labels": 0, "new_classes": []}]}
        return idx

    def _save(self, idx):
        _write_json_atomic(self.dir / "index.json", idx)

    def versions(self):
        return self._index()["versions"]

    def current(self):
        return self._index()["current"]

    def current_entry(self):
        idx = self._index()
        return next((v for v in idx["versions"] if v["version"] == idx["current"]), idx["versions"][0])

    def path(self, version=None):
        """File of a version (the current one by default); v1 is the base model file."""
        idx = self._index()
        v = idx["current"] if version is None else version
        entry = next((e for e in idx["versions"] if e["version"] == v), None)
        if not entry or not entry.get("file"):
            return self.base
        p = self.dir / entry["file"]
        return p if p.exists() else self.base

    def add(self, bundle, **meta):
        """Save a new version made from the current one and make it current."""
        with self._lock:
            idx = self._index()
            n = max(e["version"] for e in idx["versions"]) + 1
            fname = f"v{n:04d}.json"
            entry = {"version": n, "parent": idx["current"], "file": fname,
                     "created_at": datetime.now().isoformat(timespec="seconds"), "created_ts": time.time(),
                     **meta}
            bundle = {**bundle, "version": n, "parent": idx["current"]}
            _write_json_atomic(self.dir / fname, bundle)
            idx["versions"].append(entry)
            idx["current"] = n
            # prune old files (never v1, never the current one)
            files = [e for e in idx["versions"] if e.get("file") and e["version"] != n]
            for e in files[:-KEEP_VERSIONS] if len(files) > KEEP_VERSIONS else []:
                (self.dir / e["file"]).unlink(missing_ok=True)
                e["file"] = None
                e["pruned"] = True
            self._save(idx)
            return n

    def set_current(self, version):
        with self._lock:
            idx = self._index()
            idx["current"] = version
            self._save(idx)


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
      base_model, models_dir, feedback_dir, base_labels, base_embeddings, worms_cache,
      and optionally aug_labels, aug_embeddings

    Each run trains on ALL current labels, starting from the current version's
    weights, and saves the result as the next version (unless the labels are
    exactly those the current version was trained on, e.g. after an undo).
    """

    def __init__(self, config, auto=True, debounce=DEBOUNCE_S):
        self.config = config   # see store()
        self.auto = auto
        self.debounce = debounce
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._due = None
        self._thread = None
        self._base_cache = None
        self._status = {"state": "idle", "version": None, "message": "", "n_feedback": 0,
                        "n_clusters": 0, "new_classes": [], "pending_names": []}

    def store(self):
        cfg = self.config()
        return ModelStore(cfg["models_dir"], cfg["base_model"])

    def status(self):
        with self._lock:
            s = dict(self._status)
        cfg = self.config()
        cur = self.store().current_entry()
        s.update(version=cur["version"], n_clusters=cur.get("n_labels", 0), new_classes=cur.get("new_classes", []),
                 n_feedback=cur.get("n_feedback", 0))
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
        # Augmented crops (augment_crops_colab.ipynb) are used when both files are there
        aug_ps = [Path(cfg[k]) for k in ("aug_labels", "aug_embeddings") if cfg.get(k)]
        aug_ps = aug_ps if len(aug_ps) == 2 and all(p.exists() for p in aug_ps) else []
        stamp = tuple((str(p), p.stat().st_mtime) for p in [labels_p, emb_p, *aug_ps])
        if self._base_cache and self._base_cache[0] == stamp:
            return self._base_cache[1]

        def read(lp, ep):
            emb = json.loads(ep.read_text())
            labels = pd.read_csv(lp)
            key_col = find_key_col(labels, emb, log=None)
            labels = labels[labels[key_col].astype(str).isin(set(emb))].copy()
            labels["_key"] = labels[key_col].astype(str)
            return labels, emb

        labels, emb = read(labels_p, emb_p)
        if aug_ps:
            aug_labels, aug_emb = read(*aug_ps)
            labels, emb = pd.concat([labels, aug_labels], ignore_index=True), {**emb, **aug_emb}
        self._base_cache = (stamp, (labels, emb))
        return labels, emb

    def run_once(self):
        """Resolve names, retrain, and swap in the new model. Returns the new version or None."""
        import pandas as pd

        cfg = self.config()
        store = self.store()
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
        current = store.current_entry()
        data_hash = hashlib.sha1(json.dumps(sorted((r["_key"], r["family"], r["genus"]) for r in rows))
                                 .encode()).hexdigest()[:16]
        changes = self._changes_since(cfg["feedback_dir"], current)

        if current.get("data_hash", _EMPTY_HASH) == data_hash:   # e.g. undone within the debounce
            self._set(state="idle", pending_names=pending, message=f"Model v{current['version']} is up to date.")
            return None
        if not rows:   # every label was removed: the original model is exactly right again
            store.set_current(1)
            self._log_version(cfg, 1, changes, "No labels left: back to the original model (v1)")
            self._set(state="idle", pending_names=pending, message="No usable labels: using the original model.")
            return 1

        all_labels = pd.concat([labels, pd.DataFrame(rows)], ignore_index=True)
        all_emb = {**emb, **fb_emb}
        min_prob = base_model["min_prob"] if base_model else 0.7
        meta = (base_model or {}).get("meta", {})
        fit_kw = {k: meta[k] for k in ("class_weight", "C") if k in meta}   # same setup as the base model
        prev = classifier.load_classifier(store.path())   # build on the current version
        bundle, _ = train_bundle(all_labels, all_emb, min_prob=min_prob, evaluate=False, log=None,
                                 init=prev, **fit_kw)

        base_classes = set()
        if base_model:
            for m in base_model["levels"].values():
                base_classes |= set(m["classes"])
        new_classes = sorted({c for m in bundle["levels"].values() for c in m["classes"]} - base_classes)
        trained_labels = feedback_labels(cfg["feedback_dir"], names)
        n_clusters = len(trained_labels)
        bundle.update({
            "embedding": "BioCLIP-2 + fine-tuned best.pt",
            "trained_on": f"{cfg['base_labels']} + {n_clusters} reviewer-named clusters",
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "checkpoint_fingerprint": (base_model or {}).get("meta", {}).get("checkpoint_fingerprint"),
            "n_feedback": len(rows), "n_feedback_clusters": n_clusters, "new_classes": new_classes,
        })
        version = store.add(bundle, n_labels=n_clusters, n_feedback=len(rows), new_classes=new_classes,
                            data_hash=data_hash, labels=trained_labels,
                            changes=[describe(e) for e in changes], reason=", ".join(describe(e) for e in changes[-3:]))
        self._log_version(cfg, version, changes,
                          f"Model v{version} trained on {n_clusters} label{'' if n_clusters == 1 else 's'}")
        msg = f"Model v{version} learned from {n_clusters} label(s)"
        if new_classes:
            msg += f" · new: {', '.join(new_classes)}"
        if pending:
            msg += f" · couldn't look up yet: {', '.join(pending)}"
        self._set(state="idle", pending_names=pending, message=msg)
        print(msg)
        return version

    @staticmethod
    def _changes_since(feedback_dir, version_entry):
        """Label events after the given version was made (the changes the next version includes)."""
        since = version_entry.get("created_ts") or 0
        events = [e for e in read_history(feedback_dir) if e.get("ts", 0) > since and e.get("action") != "version"]
        # a change undone before this retrain cancels out: leave both out of the description
        undone = {e.get("undoes") for e in events if e.get("action") == "undo"}
        cancelled = undone & {e["id"] for e in events}
        return [e for e in events if e["id"] not in cancelled and e.get("undoes") not in cancelled]

    @staticmethod
    def _log_version(cfg, version, changes, summary):
        append_history(cfg["feedback_dir"], action="version", to=version, summary=summary,
                       includes=[e["id"] for e in changes])
