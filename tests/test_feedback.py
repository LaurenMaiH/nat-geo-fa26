import json

import numpy as np
import pytest

import classifier
import feedback
from test_classifier import DIM, _vec, bundle_path  # noqa: F401  (bundle_path is a fixture)

# WoRMS answers for the stubbed lookup
LINEAGES = {
    "Munida": [{"rank": "Order", "name": "Decapoda"}, {"rank": "Family", "name": "Munididae"},
               {"rank": "Genus", "name": "Munida"}],
    "Decapoda": [{"rank": "Phylum", "name": "Arthropoda"}, {"rank": "Order", "name": "Decapoda"}],
}


def fake_lookup(name):
    return LINEAGES.get(name)


# ── resolve_label ─────────────────────────────────────────────────────────────

def test_resolve_label(bundle_path):
    b = classifier.load_classifier(bundle_path)
    assert feedback.resolve_label("hydrolagus", b) == {"genus": "Hydrolagus", "family": "Chimaeridae", "source": "model"}
    assert feedback.resolve_label("Munididae", b) == {"genus": None, "family": "Munididae", "source": "model"}
    assert feedback.resolve_label("munida", b, fake_lookup) == {"genus": "Munida", "family": "Munididae", "source": "worms"}
    assert feedback.resolve_label("Decapoda", b, fake_lookup) is None    # above family
    assert feedback.resolve_label("Shadows", b, fake_lookup) is None     # not a taxon

    def offline(_):
        raise feedback.LookupUnavailable("no internet")
    with pytest.raises(feedback.LookupUnavailable):
        feedback.resolve_label("Munida", b, offline)
    assert feedback.resolve_label("Hydrolagus", b, offline)["genus"] == "Hydrolagus"   # known names need no network


# ── pick_representatives ──────────────────────────────────────────────────────

def test_pick_representatives_covers_every_look():
    rng = np.random.default_rng(0)
    emb, group = {}, {}
    for g, (axis, n) in enumerate([(0, 40), (1, 15), (2, 5)]):
        for i in range(n):
            emb[f"g{g}_{i}"] = _vec(axis, rng, noise=0.02)
            group[f"g{g}_{i}"] = g
    emb["outlier"] = _vec(7, rng, noise=0.02)

    picks = feedback.pick_representatives(list(emb), emb)
    keys = [k for k, _ in picks]
    assert len(keys) == feedback.MAX_EXAMPLES == len(set(keys))
    assert "outlier" not in keys
    assert {group[k] for k in keys} == {0, 1, 2}
    assert sum(w for _, w in picks) == 60
    assert feedback.pick_representatives(list(reversed(list(emb))), emb) == picks   # deterministic

    small = {k: emb[k] for k in list(emb)[:5]}
    assert feedback.pick_representatives(list(small), small) == [(k, 1) for k in sorted(small)]


# ── App: names → feedback file → retrained model ──────────────────────────────

@pytest.fixture
def app_fb(app_env, bundle_path, monkeypatch):
    """A mission with 30 tracks of an animal the base model doesn't know (axis 6) and 3 Centrophorus."""
    rng = np.random.default_rng(5)
    emb = {f"v1__{i}": _vec(6, rng) for i in range(1, 31)}
    emb.update({f"v1__{i}": _vec(3, rng) for i in range(31, 34)})
    app_env.make_mission("dep", "M1", embeddings=emb)
    app_env.select("dep", "M1")
    tmp = bundle_path.parent
    monkeypatch.setattr(app_env.app, "CLUSTER_CLASSIFIER_PATH", bundle_path)
    monkeypatch.setattr(app_env.app, "CLASSIFIER_BASE_LABELS", str(tmp / "image_labels.csv"))
    monkeypatch.setattr(app_env.app, "CLASSIFIER_BASE_EMBEDDINGS", str(tmp / "images_embeddings.json"))
    monkeypatch.setattr(feedback, "worms_lookup", lambda _cache: fake_lookup)
    state = app_env.client.post("/api/initialize", json={"seed_video": "v1"}).get_json()
    app_env.big = next(cid for cid, c in state["clusters"].items() if "v1__1" in c["track_keys"])
    app_env.fb_path = tmp / "feedback" / "dep__M1.json"
    app_env.retrained = tmp / "cluster_classifier.retrained.json"
    return app_env


def test_rename_records_capped_examples(app_fb):
    c = app_fb.client
    assert not app_fb.fb_path.exists()
    c.post("/api/rename", json={"cluster_id": app_fb.big, "name": "Munida", "via": "accept"})
    rec = json.loads(app_fb.fb_path.read_text())["clusters"][app_fb.big]
    assert (rec["name"], rec["via"]) == ("Munida", "accept")
    assert len(rec["examples"]) == 20
    assert app_fb.app._retrainer.status()["state"] == "pending"

    c.post("/api/rename", json={"cluster_id": app_fb.big, "name": "Munida "})   # typed: same label
    rec = json.loads(app_fb.fb_path.read_text())["clusters"][app_fb.big]
    assert (rec["name"], rec["via"]) == ("Munida", "accept")   # unchanged: nothing new to learn

    c.post("/api/rename", json={"cluster_id": app_fb.big, "name": "Cluster 1"})
    assert not app_fb.fb_path.exists()


def test_labels_follow_reassigned_tracks(app_fb):
    c = app_fb.client
    state = c.post("/api/rename", json={"cluster_id": app_fb.big, "name": "Munida"}).get_json()
    before = json.loads(app_fb.fb_path.read_text())["clusters"][app_fb.big]
    moved = state["clusters"][app_fb.big]["track_keys"][:25]
    c.post("/api/trash/add", json={"track_keys": moved})
    after = json.loads(app_fb.fb_path.read_text())["clusters"][app_fb.big]
    assert after["signature"] != before["signature"]
    assert len(after["examples"]) == 5 and not set(after["examples"]) & set(moved)


def test_retrain_learns_new_genus_and_reset_restores(app_fb, bundle_path):
    c = app_fb.client
    base_bytes = bundle_path.read_bytes()
    state = c.post("/api/suggestions/refresh", json={"min_prob": 0.5}).get_json()
    assert state["clusters"][app_fb.big]["suggestion"]["label"] != "Munida"

    c.post("/api/rename", json={"cluster_id": app_fb.big, "name": "Munida"})
    version = app_fb.app._retrainer.run_once()
    assert version and app_fb.retrained.exists()
    assert bundle_path.read_bytes() == base_bytes   # the original model is never overwritten

    status = c.get("/api/classifier/status").get_json()
    assert status["version"] == version and status["new_classes"] == ["Munida"]
    assert status["names"]["Munida"] == {"genus": "Munida", "family": "Munididae", "source": "worms"}
    bundle = json.loads(app_fb.retrained.read_text())
    assert "Munida" in bundle["levels"]["genus"]["classes"]
    assert bundle["genus_to_family"]["Munida"] == "Munididae"

    state = c.post("/api/suggestions/refresh", json={}).get_json()
    s = state["clusters"][app_fb.big]["suggestion"]
    assert (s["label"], s["family_name"]) == ("Munida", "Munididae")

    c.post("/api/classifier/reset")
    assert not app_fb.retrained.exists()
    state = c.post("/api/suggestions/refresh", json={}).get_json()
    assert state["clusters"][app_fb.big]["suggestion"]["label"] != "Munida"


def test_non_taxon_names_are_not_trained_on(app_fb):
    app_fb.client.post("/api/rename", json={"cluster_id": app_fb.big, "name": "Shadows"})
    assert app_fb.fb_path.exists()                 # recorded...
    assert app_fb.app._retrainer.run_once() is None
    assert not app_fb.retrained.exists()           # ...but nothing to learn
    assert app_fb.app._retrainer.status()["names"]["Shadows"] is None


def test_missing_base_data_keeps_labels(app_fb, monkeypatch):
    monkeypatch.setattr(app_fb.app, "CLASSIFIER_BASE_LABELS", "/nonexistent/image_labels.csv")
    app_fb.client.post("/api/rename", json={"cluster_id": app_fb.big, "name": "Munida"})
    assert app_fb.app._retrainer.run_once() is None
    status = app_fb.client.get("/api/classifier/status").get_json()
    assert status["state"] == "error" and "base training data not found" in status["message"]
    assert app_fb.fb_path.exists() and not app_fb.retrained.exists()


def test_restart_drops_the_missions_labels(app_fb):
    app_fb.client.post("/api/rename", json={"cluster_id": app_fb.big, "name": "Munida"})
    app_fb.client.post("/api/reset")
    assert not app_fb.fb_path.exists()
