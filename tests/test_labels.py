import json

import numpy as np

from classifier_training import export_linear, fit
from test_classifier import _vec, bundle_path  # noqa: F401  (fixture)
from test_feedback import app_fb  # noqa: F401  (fixture)


def _labels(c):
    return {(r["mission"], r["cluster_id"]): r for r in c.get("/api/labels").get_json()["labels"]}


# ── Warm start ────────────────────────────────────────────────────────────────

def test_warm_start_matches_a_fresh_fit_with_a_new_class():
    rng = np.random.default_rng(0)
    X = np.array([_vec(a, rng, 0.3) for a in [0, 1, 2, 3] for _ in range(30)])
    y = np.repeat(["a", "b", "c", "d"], 30)
    prev = export_linear(fit(X[y != "d"], y[y != "d"]))         # the previous version didn't know "d"
    warm, cold = fit(X, y, init=prev), fit(X, y)
    assert warm.warm_started and not cold.warm_started
    assert np.abs(warm.predict_proba(X) - cold.predict_proba(X)).max() < 0.02


# ── Versions ──────────────────────────────────────────────────────────────────

def test_each_label_change_makes_a_new_version_built_on_the_last(app_fb, bundle_path):
    c, r = app_fb.client, app_fb.app._retrainer
    base = bundle_path.read_bytes()
    c.post("/api/rename", json={"cluster_id": app_fb.big, "name": "Munida"})
    assert r.run_once() == 2
    state = c.get("/api/state").get_json()
    other = next(cid for cid, cl in state["clusters"].items() if cid not in ("noise", app_fb.big))
    c.post("/api/rename", json={"cluster_id": other, "name": "Centrophorus"})
    assert r.run_once() == 3

    v = c.get("/api/classifier/versions").get_json()
    assert v["current"] == 3
    assert [(e["version"], e["parent"]) for e in v["versions"]] == [(3, 2), (2, 1), (1, None)]
    assert v["versions"][0]["n_labels"] == 2 and "Centrophorus" in v["versions"][0]["reason"]
    assert bundle_path.read_bytes() == base                      # v1 is the untouched base file
    assert r.run_once() is None                                  # same labels: no new version

    c.post("/api/classifier/reset")
    assert c.get("/api/classifier/status").get_json()["version"] == 1
    assert (app_fb.models / "v0003.json").exists()               # versions are kept


# ── Undo ──────────────────────────────────────────────────────────────────────

def test_undo_before_the_retrain_makes_no_new_version(app_fb):
    c = app_fb.client
    state = c.post("/api/rename", json={"cluster_id": app_fb.big, "name": "Munida", "via": "accept"}).get_json()
    change = state["_label_change"]
    assert change["action"] == "confirm" and change["to"] == "Munida"
    assert "_label_change" not in json.loads(
        (app_fb.root / "dep" / "M1" / "state.json").read_text())   # toast info isn't saved

    assert c.post("/api/labels/undo", json={"history_id": change["history_id"]}).status_code == 200
    assert c.get("/api/state").get_json()["clusters"][app_fb.big]["name"].startswith("Cluster ")
    assert not app_fb.fb_path.exists()
    assert app_fb.app._retrainer.run_once() is None             # nothing to learn: still v1


def test_undo_after_a_retrain_removes_the_label_from_the_model(app_fb):
    c, r = app_fb.client, app_fb.app._retrainer
    hid = c.post("/api/rename", json={"cluster_id": app_fb.big, "name": "Munida"}).get_json()["_label_change"]["history_id"]
    assert r.run_once() == 2
    c.post("/api/labels/undo", json={"history_id": hid})
    assert r.run_once() == 1                                     # no labels left: the original model again
    assert c.post("/api/labels/undo", json={"history_id": hid}).status_code == 409   # already undone


def test_undo_is_refused_after_a_later_rename(app_fb):
    c = app_fb.client
    hid = c.post("/api/rename", json={"cluster_id": app_fb.big, "name": "Munida"}).get_json()["_label_change"]["history_id"]
    c.post("/api/rename", json={"cluster_id": app_fb.big, "name": "Munididae"})
    res = c.post("/api/labels/undo", json={"history_id": hid})
    assert res.status_code == 409 and "Munididae" in res.get_json()["error"]


def test_renames_that_dont_change_the_label_are_not_logged(app_fb):
    c = app_fb.client
    assert c.post("/api/rename", json={"cluster_id": app_fb.big, "name": "Cluster 99"}).get_json()["_label_change"] is None
    assert c.get("/api/labels/history").get_json() == []


# ── The Labels tab ────────────────────────────────────────────────────────────

def test_labels_list_and_edits_in_a_mission_that_isnt_open(app_fb):
    c, env = app_fb.client, app_fb
    rng = np.random.default_rng(9)
    env.make_mission("dep", "M2", embeddings={f"v2__{i}": _vec(6, rng) for i in range(1, 9)})
    env.select("dep", "M2")
    s2 = c.post("/api/initialize", json={"seed_video": "v2"}).get_json()
    m2 = next(cid for cid in s2["clusters"] if cid != "noise")
    c.post("/api/rename", json={"cluster_id": m2, "name": "Shadows"})
    env.select("dep", "M1")
    c.post("/api/rename", json={"cluster_id": app_fb.big, "name": "Munida"})
    app_fb.app._retrainer.run_once()

    rows = _labels(c)
    assert rows[("M1", app_fb.big)]["status"] == "confirmed"
    assert rows[("M1", app_fb.big)]["taxon"]["family"] == "Munididae" and rows[("M1", app_fb.big)]["in_model"]
    assert len(rows[("M1", app_fb.big)]["examples"]) == 20
    assert rows[("M2", m2)]["status"] == "not_taxon" and not rows[("M2", m2)]["is_open_mission"]

    # rename M2's cluster from the Labels tab (M1 stays open), then remove it from training, then undo that
    res = c.post("/api/labels/rename", json={"deployment": "dep", "mission": "M2", "cluster_id": m2, "name": "Munida"})
    assert res.get_json()["_label_change"]["action"] == "rename" and not res.get_json()["open_mission_changed"]
    assert (app_fb.fb_path.parent / "dep__M2.json").exists()
    res = c.post("/api/labels/exclude", json={"deployment": "dep", "mission": "M2", "cluster_id": m2, "excluded": True})
    assert _labels(c)[("M2", m2)]["status"] == "excluded"
    m2_file = app_fb.fb_path.parent / "dep__M2.json"
    assert not m2_file.exists() or m2 not in json.loads(m2_file.read_text())["clusters"]
    c.post("/api/labels/undo", json={"history_id": res.get_json()["_label_change"]["history_id"]})
    assert _labels(c)[("M2", m2)]["status"] == "confirmed"

    history = c.get("/api/labels/history").get_json()
    assert [h["action"] for h in history if h["action"] != "version"][:3] == ["undo", "exclude", "rename"]
    excl = next(h for h in history if h["action"] == "exclude")
    assert excl["undone"] and not excl["undoable"]


def test_crops_from_other_missions_reject_unsafe_ids(app_fb):
    assert app_fb.client.get("/api/crop/v1/1?deployment=dep&mission=../M1").status_code == 404
    assert app_fb.client.get("/api/crop/v1/1?deployment=dep&mission=M1").status_code == 404   # no crop files here
