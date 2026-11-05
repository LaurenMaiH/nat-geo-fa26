import csv
import io
import json
import subprocess
import sys

import numpy as np
import pytest

import classifier
from conftest import REPO

DIM = 8
# genus → family, each genus around its own axis
GENERA = {"Heterocarpus": ("Pandalidae", 0), "Plesionika": ("Pandalidae", 1),
          "Hydrolagus": ("Chimaeridae", 2), "Centrophorus": ("Centrophoridae", 3)}


def _vec(axis, rng, noise=0.05):
    v = np.zeros(DIM)
    v[axis] = 1.0
    return (v + rng.normal(0, noise, DIM)).tolist()


@pytest.fixture
def bundle_path(tmp_path):
    """A classifier bundle made by train_cluster_classifier.py itself from synthetic labeled crops."""
    rng = np.random.default_rng(0)
    emb, rows = {}, []
    for genus, (family, axis) in GENERA.items():
        for i in range(12):
            key = f"{genus}_{i}"
            emb[key] = _vec(axis, rng)
            rows.append({"image": key, "family": family, "genus": genus, "mission": "AB"[i % 2]})
    for i in range(12):   # a family only ever identified to family level (no genus)
        emb[f"munid_{i}"] = _vec(4, rng)
        rows.append({"image": f"munid_{i}", "family": "Munididae", "genus": "", "mission": "AB"[i % 2]})
    rows.append({"image": "rare_0", "family": "Rareidae", "genus": "Rarus", "mission": "A"})
    emb["rare_0"] = _vec(5, rng)

    labels = tmp_path / "image_labels.csv"
    with open(labels, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "family", "genus", "mission"])
        w.writeheader()
        w.writerows(rows)
    embeddings = tmp_path / "images_embeddings.json"
    embeddings.write_text(json.dumps(emb))

    out = tmp_path / "cluster_classifier.json"
    r = subprocess.run([sys.executable, str(REPO / "train_cluster_classifier.py"),
                        "--labels", str(labels), "--embeddings", str(embeddings), "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "app loader matches sklearn" in r.stdout
    classifier._cache.clear()
    yield out
    classifier._cache.clear()


# ── Training output ───────────────────────────────────────────────────────────

def test_bundle_contents(bundle_path):
    raw = json.loads(bundle_path.read_text())
    assert raw["embedding_dim"] == DIM
    assert sorted(raw["levels"]["genus"]["classes"]) == sorted(GENERA)
    assert "Rareidae" not in raw["levels"]["family"]["classes"]   # < min-per-class
    assert raw["n_train"] == {"family": 60, "genus": 48}
    assert raw["genus_to_family"] == {g: f for g, (f, _) in GENERA.items()}


# ── suggest_for_tracks ────────────────────────────────────────────────────────

def test_suggests_genus_when_confident(bundle_path):
    rng = np.random.default_rng(1)
    b = classifier.load_classifier(bundle_path)
    emb = {f"v__{i}": _vec(2, rng) for i in range(3)}
    s = classifier.suggest_for_tracks(list(emb), emb, b, min_prob=0.3)
    assert (s["label"], s["level"], s["n_tracks"]) == ("Hydrolagus", "genus", 3)
    assert (s["family_name"], s["source"]) == ("Chimaeridae", "genus")
    assert s["genus"][0][0] == "Hydrolagus" and s["family"][0][0] == "Chimaeridae"
    assert len(s["genus"]) == 3 and s["genus"][0][1] >= s["genus"][1][1] >= s["genus"][2][1]


def test_falls_back_to_family_then_unknown(bundle_path):
    b = classifier.load_classifier(bundle_path)
    # halfway between the two Pandalidae genera: genus is uncertain, family is not
    v = np.zeros(DIM); v[0] = v[1] = 1.0
    emb = {"v__1": v.tolist()}
    s = classifier.suggest_for_tracks(["v__1"], emb, b, min_prob=0.6)
    assert s["genus"][0][1] < 0.6
    assert (s["label"], s["level"], s["source"]) == ("Pandalidae", "family", "genus_sum")
    assert s["family"][0][0] == "Pandalidae"
    s = classifier.suggest_for_tracks(["v__1"], emb, b, min_prob=0.999)
    assert (s["label"], s["level"], s["prob"]) == ("unknown", None, None)


def test_family_never_contradicts_confident_genus(bundle_path):
    """A confident genus brings its own family, even if the family model disagrees."""
    b = classifier.load_classifier(bundle_path)
    fm = b["levels"]["family"]
    rng = np.random.default_rng(3)
    emb = {f"v__{i}": _vec(0, rng) for i in range(3)}   # Heterocarpus
    # sabotage the family model so it prefers Chimaeridae for everything
    fm["intercept"] = fm["intercept"].copy()
    fm["intercept"][fm["classes"].index("Chimaeridae")] += 50
    s = classifier.suggest_for_tracks(list(emb), emb, b, min_prob=0.5)
    assert (s["label"], s["family_name"]) == ("Heterocarpus", "Pandalidae")
    assert s["family"][0][0] == "Pandalidae"


def test_family_model_used_for_family_without_genera(bundle_path):
    b = classifier.load_classifier(bundle_path)
    v = np.zeros(DIM); v[4] = 1.0   # Munididae: never labeled to genus
    s = classifier.suggest_for_tracks(["v__1"], {"v__1": v.tolist()}, b, min_prob=0.6)
    assert (s["label"], s["source"]) == ("Munididae", "family_model")
    assert s["family"][0][0] == "Munididae"


def test_no_suggestion_without_classifier_or_matching_embeddings(bundle_path):
    b = classifier.load_classifier(bundle_path)
    assert classifier.suggest_for_tracks(["v__1"], {"v__1": [1.0] * DIM}, None) is None
    assert classifier.suggest_for_tracks(["v__1"], {"v__1": [1.0] * 512}, b) is None   # placeholder dim
    assert classifier.suggest_for_tracks(["missing"], {}, b) is None


def test_missing_or_broken_bundle_loads_as_none(tmp_path):
    classifier._cache.clear()
    assert classifier.load_classifier(tmp_path / "nope.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert classifier.load_classifier(bad) is None


# ── App integration ───────────────────────────────────────────────────────────

def _csv(resp):
    return list(csv.reader(io.StringIO(resp.get_data(as_text=True))))


@pytest.fixture
def app_with_classifier(app_env, bundle_path, monkeypatch):
    rng = np.random.default_rng(2)
    emb = {f"v1__{i}": _vec(2, rng) for i in range(1, 4)}            # Hydrolagus
    emb.update({f"v1__{i}": _vec(3, rng) for i in range(4, 7)})      # Centrophorus
    app_env.make_mission("dep", "M1", embeddings=emb)
    app_env.select("dep", "M1")
    monkeypatch.setattr(app_env.app, "CLUSTER_CLASSIFIER_PATH", bundle_path)
    return app_env


def test_initialize_adds_suggestions_without_renaming(app_with_classifier):
    state = app_with_classifier.client.post("/api/initialize", json={"seed_video": "v1"}).get_json()
    real = {cid: c for cid, c in state["clusters"].items() if cid != "noise"}
    assert {c["suggestion"]["label"] for c in real.values()} == {"Hydrolagus", "Centrophorus"}
    assert all(c["name"].startswith("Cluster ") for c in real.values())
    assert "suggestion" not in state["clusters"]["noise"]


def test_suggestion_follows_reassign_and_accept_renames(app_with_classifier):
    c = app_with_classifier.client
    state = c.post("/api/initialize", json={"seed_video": "v1"}).get_json()
    hyd = next(cid for cid, cl in state["clusters"].items() if cl.get("suggestion", {}).get("label") == "Hydrolagus")
    cen = next(cid for cid, cl in state["clusters"].items() if cl.get("suggestion", {}).get("label") == "Centrophorus")

    # move every Centrophorus track into the Hydrolagus cluster: its suggestion must update
    state = c.post("/api/reassign", json={"target_cluster_id": hyd,
                                          "track_keys": state["clusters"][cen]["track_keys"]}).get_json()
    assert state["clusters"][hyd]["suggestion"]["n_tracks"] == 6

    state = c.post("/api/rename", json={"cluster_id": hyd, "name": "Hydrolagus"}).get_json()
    assert state["clusters"][hyd]["name"] == "Hydrolagus"


def test_refresh_endpoint_sets_cutoff(app_with_classifier):
    c = app_with_classifier.client
    c.post("/api/initialize", json={"seed_video": "v1"})
    state = c.post("/api/suggestions/refresh", json={"min_prob": 1.0}).get_json()
    assert state["classifier_min_prob"] == 1.0
    assert all(cl["suggestion"]["label"] == "unknown"
               for cid, cl in state["clusters"].items() if cid != "noise")
    assert c.post("/api/suggestions/refresh", json={"min_prob": 2}).status_code == 400
    assert c.post("/api/suggestions/refresh", json={"min_prob": "x"}).status_code == 400


def test_exports_include_suggestion(app_with_classifier):
    c = app_with_classifier.client
    c.post("/api/initialize", json={"seed_video": "v1"})
    header, *rows = _csv(c.get("/api/export/summary.csv"))
    assert header[-4:] == ["suggested_label", "suggested_level", "suggested_prob", "suggested_family"]
    assert {r[-4] for r in rows if r[2] != "noise"} == {"Hydrolagus", "Centrophorus"}
    header, *rows = _csv(c.get("/api/export/tracks.csv"))
    assert header[-4:] == ["suggested_label", "suggested_level", "suggested_prob", "suggested_family"]
    assert all(r[-4] in ("Hydrolagus", "Centrophorus") for r in rows)


def test_app_unchanged_without_classifier(app_env):
    app_env.make_mission("dep", "M1", embeddings={"v1__1": [1.0] + [0.0] * 7, "v1__2": [1.0] + [0.0] * 7})
    app_env.select("dep", "M1")
    state = app_env.client.post("/api/initialize", json={"seed_video": "v1"}).get_json()
    assert not any("suggestion" in cl for cl in state["clusters"].values())
