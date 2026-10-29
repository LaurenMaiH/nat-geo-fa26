import csv
import io
import json
import os
import time

import pytest

VEC_A = [1.0, 0.0, 0.0]
VEC_B = [0.0, 1.0, 0.0]


def _csv_rows(resp):
    return list(csv.reader(io.StringIO(resp.get_data(as_text=True))))


# ── ID / path validation ──────────────────────────────────────────────────────

@pytest.mark.parametrize("body", [
    {"mission_id": "../x", "deployment_id": "dep"},
    {"mission_id": "..\\x", "deployment_id": "dep"},
    {"mission_id": "M1", "deployment_id": "../x"},
    {"mission_id": "M 1", "deployment_id": "dep"},
])
def test_pipeline_start_rejects_bad_ids(app_env, body):
    body = {**body, "source_type": "dropbox", "source_root": "/X"}
    resp = app_env.client.post("/api/pipeline/start", json=body)
    assert resp.status_code == 400
    assert "Invalid" in resp.get_json()["error"]
    assert list(app_env.root.iterdir()) == []
    assert not (app_env.root.parent / "x").exists()


def test_select_review_mission_rejects_bad_id(app_env):
    app_env.make_mission("dep", "M1")
    app_env.select("dep", "M1")
    for bad in ("../x", "..\\M1", "M1/..", "C:M1"):
        resp = app_env.client.post("/api/select-review-mission", json={"mission_id": bad})
        assert resp.status_code == 400, bad
    assert app_env.client.post("/api/select-review-mission", json={"mission_id": "M1"}).status_code == 200


def test_pipeline_missions_rejects_bad_deployment(app_env):
    assert app_env.client.get("/api/pipeline/missions?deployment_id=../x").status_code == 400


@pytest.mark.parametrize("key", [
    "../../outside__1", "..\\..\\outside__1", "a/b__1", "C:x__1", "vid__..", "vid__1/2", "no_separator",
])
def test_trash_delete_rejects_crafted_keys(app_env, key):
    m = app_env.make_mission("dep", "M1", embeddings={"vid__1": VEC_A})
    app_env.select("dep", "M1")
    before = (m / "M1_embeddings.json").read_text()
    resp = app_env.client.post("/api/trash/delete", json={"track_keys": [key]})
    assert resp.status_code == 400
    assert "Invalid track key" in resp.get_json()["error"]
    assert (m / "M1_embeddings.json").read_text() == before


def test_trash_delete_valid_key_still_works(app_env):
    m = app_env.make_mission("dep", "M1", embeddings={"vid__1": VEC_A, "vid__2": VEC_B})
    crop = m / "M1_crops" / "vid" / "id0001_fish_f00010_mid.jpg"
    crop.parent.mkdir()
    crop.write_bytes(b"jpg")
    app_env.select("dep", "M1")
    resp = app_env.client.post("/api/trash/delete", json={"track_keys": ["vid__1"]})
    assert resp.status_code == 200
    assert not crop.exists()
    assert set(json.loads((m / "M1_embeddings.json").read_text())) == {"vid__2"}


@pytest.mark.parametrize("url", [
    "/api/crop/..%5C..%5Coutside/1",
    "/api/crop/vid/..",
    "/api/crop/vid/*",
    "/api/crop/a..b/1",
    "/api/video/..%5Coutside/proxy-status",
    "/api/video/C:outside/proxy",
    "/api/video/..%5Coutside/detections",
])
def test_url_names_with_path_parts_rejected(app_env, url):
    app_env.make_mission("dep", "M1")
    app_env.select("dep", "M1")
    resp = app_env.client.get(url)
    assert resp.status_code == 400
    assert "Invalid" in resp.get_json()["error"]


# ── Detections CSV: extensions and bad rows ───────────────────────────────────

CSV_MP4_UPPER = (
    "clip.MP4,0,1,fish,0.1,0.1,0.2,0.2,0.9\n"
    "clip.MP4,1,1,fish,0.1,0.1,0.2,0.2,0.8\n"
    "clip.MP4,2,1,eel,0.1,0.1,0.2,0.2,0.7\n"
    "clip.MP4,-1,,,,,,,\n"            # sentinel row left by an older crashed run
    "clip.MP4,3,1,fi\n"               # half-written last line
)


def test_detections_loader_handles_uppercase_extension_and_bad_rows(app_env, capsys):
    app_env.make_mission("dep", "M1", embeddings={"clip__1": VEC_A}, csv_text=CSV_MP4_UPPER)
    app_env.select("dep", "M1")
    assert "skipped 2 unreadable row(s)" in capsys.readouterr().out
    assert sorted(app_env.app.DETECTIONS_CACHE["clip"]) == [0, 1, 2]

    dets = app_env.client.get("/api/video/clip/detections").get_json()["detections"]
    assert sorted(int(f) for f in dets) == [0, 1, 2]

    header, *rows = _csv_rows(app_env.client.get("/api/export/tracks.csv"))
    row = dict(zip(header, rows[0]))
    assert (row["video"], row["track_id"], row["n_detections"], row["top_class"]) == ("clip", "1", "3", "fish")


# ── Zero-track clips: review navigation and summary export ────────────────────

def _zero_track_mission(app_env):
    emb = {"v1__1": VEC_A, "v1__2": VEC_A, "v3__1": VEC_B}
    manifest = {
        "v0": {"filename": "v0.mp4", "path": "M1/v0.mp4", "status": "complete", "n_tracks": 0},
        "v1": {"filename": "v1.mp4", "path": "M1/v1.mp4", "status": "complete", "n_tracks": 2},
        "v2": {"filename": "v2.MOV", "path": "M1/v2.MOV", "status": "complete", "n_tracks": 0},
        "v3": {"filename": "v3.mp4", "path": "M1/v3.mp4", "status": "complete", "n_tracks": 1},
        "v4": {"filename": "v4.mp4", "path": "M1/v4.mp4", "status": "in_progress"},
    }
    csv_text = "".join(f"v1.mp4,{f},{t},fish,0.1,0.1,0.2,0.2,0.9\n" for f in range(3) for t in (1, 2))
    csv_text += "v3.mp4,0,1,eel,0.1,0.1,0.2,0.2,0.9\n"
    m = app_env.make_mission("dep", "M1", embeddings=emb, csv_text=csv_text, manifest=manifest)
    app_env.select("dep", "M1")
    return m


def test_zero_track_clips_are_listed(app_env):
    _zero_track_mission(app_env)
    videos = {v["name"]: v for v in app_env.client.get("/api/videos").get_json()}
    assert list(videos) == ["v0", "v1", "v2", "v3"]          # v4 isn't finished
    assert videos["v0"]["no_detections"] and videos["v0"]["track_count"] == 0
    assert not videos["v1"]["no_detections"] and videos["v1"]["track_count"] == 2
    state = app_env.client.get("/api/state").get_json()
    assert state["videos"]["v2"]["no_detections"]


def test_zero_track_seed_rejected(app_env):
    _zero_track_mission(app_env)
    resp = app_env.client.post("/api/initialize", json={"seed_video": "v0"})
    assert resp.status_code == 400
    assert "no detections" in resp.get_json()["error"]


def test_review_navigation_through_zero_track_clips(app_env):
    _zero_track_mission(app_env)
    c = app_env.client
    state = c.post("/api/initialize", json={"seed_video": "v1"}).get_json()
    assert state["current_video"] == "v0"
    assert set(state["videos"]) == {"v0", "v1", "v2", "v3"}

    state = c.post("/api/process-video", json={"video_name": "v0"}).get_json()
    assert state["videos"]["v0"] == {"name": "v0", "status": "processed", "track_count": 0}
    state = c.post("/api/approve", json={"video_name": "v0"}).get_json()
    assert state["videos"]["v0"]["status"] == "approved" and state["current_video"] == "v1"
    assert c.get("/api/video/v0/tracks").get_json() == []

    state = c.post("/api/approve", json={"video_name": "v1"}).get_json()
    assert state["current_video"] == "v2"
    state = c.post("/api/process-video", json={"video_name": "v2"}).get_json()
    state = c.post("/api/approve", json={"video_name": "v2"}).get_json()
    assert state["current_video"] == "v3"
    assert c.post("/api/previous", json={"video_name": "v3"}).get_json()["current_video"] == "v2"
    assert c.post("/api/previous", json={"video_name": "v1"}).get_json()["current_video"] == "v0"


def test_approve_clip_added_after_initialization(app_env):
    m = _zero_track_mission(app_env)
    c = app_env.client
    c.post("/api/initialize", json={"seed_video": "v1"})
    manifest = json.loads((m / "M1_manifest.json").read_text())
    manifest["v9"] = {"filename": "v9.mp4", "path": "M1/v9.mp4", "status": "complete", "n_tracks": 0}
    (m / "M1_manifest.json").write_text(json.dumps(manifest))
    state = c.post("/api/approve", json={"video_name": "v9"}).get_json()
    assert state["videos"]["v9"]["status"] == "approved"
    assert state["current_video"] is None


def test_summary_export_has_none_rows_for_zero_track_clips(app_env):
    _zero_track_mission(app_env)
    c = app_env.client
    c.post("/api/initialize", json={"seed_video": "v1"})
    header, *rows = _csv_rows(c.get("/api/export/summary.csv"))
    assert header == ["mission_id", "video", "cluster_id", "cluster_name", "n_tracks", "n_detections", "max_n"]
    per_video = [r for r in rows if r[1] != "ALL"]
    assert ["M1", "v0", "none", "", "0", "0", "0"] in per_video
    assert ["M1", "v2", "none", "", "0", "0", "0"] in per_video
    assert [r[1] for r in per_video] == sorted(r[1] for r in per_video)
    v1 = [r for r in per_video if r[1] == "v1"]
    assert sum(int(r[4]) for r in v1) == 2 and sum(int(r[5]) for r in v1) == 6
    assert not any(r[1] in ("v3", "v4") for r in per_video)   # v3 unreviewed, v4 unfinished
    assert not any(r[2] == "none" for r in rows if r[1] == "ALL")


def test_summary_export_none_row_when_all_tracks_trashed(app_env):
    _zero_track_mission(app_env)
    c = app_env.client
    c.post("/api/initialize", json={"seed_video": "v1"})
    c.post("/api/trash/add", json={"track_keys": ["v1__1", "v1__2"]})
    rows = _csv_rows(c.get("/api/export/summary.csv"))[1:]
    assert ["M1", "v1", "none", "", "0", "0", "0"] in rows


def test_mission_without_manifest_lists_embeddings_only(app_env):
    """Missions processed before manifests (e.g. the Palau sample) behave as before."""
    app_env.make_mission("dep", "M1", embeddings={"a__1": VEC_A, "b__1": VEC_B, "b__2": VEC_B})
    app_env.select("dep", "M1")
    videos = app_env.client.get("/api/videos").get_json()
    assert [(v["name"], v["track_count"], v["no_detections"]) for v in videos] == [
        ("a", 1, False), ("b", 2, False)]
    app_env.client.post("/api/initialize", json={"seed_video": "a"})
    rows = _csv_rows(app_env.client.get("/api/export/summary.csv"))[1:]
    assert not any(r[2] == "none" for r in rows)


# ── Pipeline start ordering ───────────────────────────────────────────────────

def _local_videos(tmp_path, names):
    folder = tmp_path / "videos" / "M1" / "Video"
    folder.mkdir(parents=True)
    for n in names:
        (folder / n).write_bytes(b"")
    return tmp_path / "videos"


def _start_body(root, deployment_id="newdep"):
    return {"mission_id": "M1", "deployment_id": deployment_id,
            "source_type": "local", "source_root": str(root)}


def test_start_while_running_returns_409_and_creates_no_deployment(app_env, tmp_path, monkeypatch):
    monkeypatch.delenv("LOCAL_VIDEO_ROOT", raising=False)
    root = _local_videos(tmp_path, ["a.mp4"])
    monkeypatch.setitem(app_env.app._pipeline_state, "running", True)
    resp = app_env.client.post("/api/pipeline/start", json=_start_body(root))
    assert resp.status_code == 409
    assert not (app_env.root / "newdep").exists()


def test_start_creates_deployment_only_when_starting(app_env, tmp_path, monkeypatch):
    monkeypatch.delenv("LOCAL_VIDEO_ROOT", raising=False)
    started = []
    monkeypatch.setattr(app_env.app.threading, "Thread",
                        lambda target, args, daemon: type("T", (), {"start": lambda self: started.append(args)})())
    root = _local_videos(tmp_path, ["a.mp4"])
    # validation failure: nothing created
    resp = app_env.client.post("/api/pipeline/start", json={**_start_body(root), "mission_path": "nope"})
    assert resp.status_code == 400 and not (app_env.root / "newdep").exists()
    resp = app_env.client.post("/api/pipeline/start", json=_start_body(root))
    assert resp.status_code == 200 and started
    assert (app_env.root / "newdep" / "deployment.json").exists()
    assert not (app_env.root / "newdep" / "M1").exists()   # the pipeline creates it, not the route


def test_start_rejects_stem_collisions(app_env, tmp_path, monkeypatch):
    monkeypatch.delenv("LOCAL_VIDEO_ROOT", raising=False)
    root = _local_videos(tmp_path, ["a.mp4", "a.MOV", "b.mp4"])
    resp = app_env.client.post("/api/pipeline/start", json=_start_body(root))
    assert resp.status_code == 400
    assert "a.MOV and a.mp4" in resp.get_json()["error"]
    assert not (app_env.root / "newdep").exists()


# ── Proxy status per deployment ───────────────────────────────────────────────

def test_proxy_status_is_per_deployment(app_env):
    a = app_env.make_mission("depA", "M1")
    app_env.make_mission("depB", "M1")
    c = app_env.client

    app_env.select("depA", "M1")
    (a / "proxy_videos").mkdir()
    (a / "proxy_videos" / "clip_proxy.mp4").write_bytes(b"mp4")
    assert c.get("/api/video/clip/proxy-status").get_json()["status"] == "ready"
    app_env.app._proxy_status[str(a / "proxy_videos" / "clip_proxy.mp4")] = "ready"

    app_env.select("depB", "M1")
    assert c.get("/api/video/clip/proxy-status").get_json()["status"] == "missing"
    assert c.get("/api/video/clip/proxy").status_code == 404

    app_env.select("depA", "M1")
    assert c.get("/api/video/clip/proxy").status_code == 200


# ── Startup cleanup ───────────────────────────────────────────────────────────

def test_cleanup_stale_temp_files(app_env, tmp_path, monkeypatch):
    old = time.time() - 3600
    proxies = app_env.make_mission("dep", "M1") / "proxy_videos"
    proxies.mkdir()
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    monkeypatch.setattr(app_env.app.tempfile, "gettempdir", lambda: str(tmp))

    def make(path, is_dir=False, mtime=old):
        if is_dir:
            path.mkdir()
            (path / "video.mp4").write_bytes(b"x")
            os.utime(path / "video.mp4", (mtime, mtime))
        else:
            path.write_bytes(b"x")
        os.utime(path, (mtime, mtime))
        return path

    stale = [make(proxies / "a_proxy.tmp.mp4"), make(proxies / ".a_proxy_k2j3h.mp4"),
             make(tmp / "natgeo_abc", is_dir=True), make(tmp / "natgeo_dl_x", is_dir=True),
             make(tmp / "natgeo_file.mp4")]
    keep = [make(proxies / "a_proxy.mp4"), make(proxies / "b_proxy.tmp.mp4", mtime=time.time()),
            make(tmp / "natgeo_running", is_dir=True, mtime=time.time()),
            make(tmp / "other_tool", is_dir=True)]

    monkeypatch.setitem(app_env.app._pipeline_state, "running", True)
    app_env.app._cleanup_stale_temp_files()
    assert all(p.exists() for p in stale + keep)

    monkeypatch.setitem(app_env.app._pipeline_state, "running", False)
    app_env.app._cleanup_stale_temp_files()
    assert not any(p.exists() for p in stale)
    assert all(p.exists() for p in keep)
