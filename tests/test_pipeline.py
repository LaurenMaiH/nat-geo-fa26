import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from conftest import DETECTIONS_HEADER, REPO
from video_sources import LocalSource, Video

PIPELINE_FILE = REPO / "overnight_pipeline_optimized.py"


def _load_pipeline():
    spec = importlib.util.spec_from_file_location("pipeline_under_test", PIPELINE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def pipeline():
    return _load_pipeline()


# ── Import side effects ───────────────────────────────────────────────────────

def test_import_creates_no_folders(monkeypatch):
    _load_pipeline()   # warm the heavy imports (torch, ultralytics, ...) first
    processed = REPO / "processed_missions"
    before = sorted(p.name for p in processed.iterdir()) if processed.exists() else None
    created = []
    real_mkdir, real_makedirs = Path.mkdir, os.makedirs
    monkeypatch.setattr(Path, "mkdir", lambda self, *a, **k: (created.append(self), real_mkdir(self, *a, **k))[1])
    monkeypatch.setattr(os, "makedirs", lambda p, *a, **k: (created.append(p), real_makedirs(p, *a, **k))[1])
    _load_pipeline()
    assert created == []
    after = sorted(p.name for p in processed.iterdir()) if processed.exists() else None
    assert after == before


def test_output_paths_follow_settings_made_after_import(pipeline, tmp_path):
    pipeline.MISSION_ID = "M7"
    pipeline.OUTPUT_DIR = tmp_path / "dep" / "M7"
    paths = pipeline.output_paths()
    assert paths.csv == tmp_path / "dep" / "M7" / "M7_detections.csv"
    assert paths.manifest == tmp_path / "dep" / "M7" / "M7_manifest.json"
    assert not paths.dir.exists()
    pipeline.MISSION_ID = "../x"
    with pytest.raises(ValueError, match="Invalid MISSION_ID"):
        pipeline.output_paths()


# ── Resume planning ───────────────────────────────────────────────────────────

ENTRIES = [Video(n, f"M/{n}") for n in ("a.mp4", "b.MP4", "c.mov", "d.mp4")]
MANIFEST = {
    "a": {"status": "complete", "n_tracks": 4},
    "b": {"status": "in_progress"},
    "c": {"status": "complete", "n_tracks": 0},     # zero-detection video
}


def _names(videos):
    return [v.name for v in videos]


def test_plan_skips_complete_including_zero_track(pipeline):
    plan = pipeline.plan_videos(ENTRIES, MANIFEST)
    assert _names(plan.todo) == ["b.MP4", "d.mp4"]
    assert _names(plan.done) == ["a.mp4", "c.mov"]


def test_plan_with_selected_videos(pipeline):
    plan = pipeline.plan_videos(ENTRIES, MANIFEST, selected=["c.mov", "d.mp4", "zz.mp4"])
    assert _names(plan.todo) == ["d.mp4"]
    assert _names(plan.done) == ["c.mov"]
    assert plan.missing == {"zz.mp4"}


def test_plan_with_start_from(pipeline):
    plan = pipeline.plan_videos(ENTRIES, MANIFEST, start_from="c.mov")
    assert plan.start_found and _names(plan.todo) == ["d.mp4"]
    plan = pipeline.plan_videos(ENTRIES, {}, start_from="b.MP4")
    assert _names(plan.todo) == ["b.MP4", "c.mov", "d.mp4"]
    plan = pipeline.plan_videos(ENTRIES, {}, start_from="nope.mp4")
    assert plan.start_found is False and _names(plan.todo) == _names(ENTRIES)


def test_legacy_completion_uses_last_csv_row(pipeline, tmp_path, monkeypatch):
    monkeypatch.delenv("LOCAL_VIDEO_ROOT", raising=False)
    source = LocalSource(str(tmp_path))
    csv_path = tmp_path / "det.csv"
    csv_path.write_text(DETECTIONS_HEADER + "a.mp4,0,1,fish,0,0,0,0,0.9\nb.MP4,0,2,fish,0,0,0,0,0.9\n")
    emb = {"a__1": [1.0], "b__2": [1.0], "b__3": [1.0]}
    legacy = pipeline.legacy_completion(source, ENTRIES, {}, csv_path, emb)
    assert sorted(legacy) == ["a", "b"]
    assert legacy["b"]["status"] == "complete" and legacy["b"]["n_tracks"] == 2
    assert legacy["a"]["filename"] == "a.mp4" and legacy["a"]["path"] == "M/a.mp4"
    # a manifest that already has completion records is never second-guessed
    assert pipeline.legacy_completion(source, ENTRIES, MANIFEST, csv_path, emb) == {}


def test_purge_video_outputs(pipeline, tmp_path):
    paths = SimpleNamespace(csv=tmp_path / "det.csv", crops=tmp_path / "crops")
    paths.csv.write_text(DETECTIONS_HEADER
                         + "a.mp4,0,1,fish,0,0,0,0,0.9\n"
                         + "b.MP4,0,1,fish,0,0,0,0,0.9\n"
                         + "c.mov,-1,,,,,,,\n"          # old sentinel row
                         + "b.MP4,1,1,fi\n")            # partial last line
    (paths.crops / "b").mkdir(parents=True)
    (paths.crops / "a").mkdir()
    emb = {"a__1": [1.0], "b__1": [1.0]}
    assert pipeline.purge_video_outputs({"b"}, paths, emb) is True
    assert paths.csv.read_text().splitlines() == [DETECTIONS_HEADER.strip(), "a.mp4,0,1,fish,0,0,0,0,0.9"]
    assert not (paths.crops / "b").exists() and (paths.crops / "a").exists()
    assert emb == {"a__1": [1.0]}
    assert pipeline.purge_video_outputs({"zz"}, paths, emb) is False


# ── main(): completion records, resume, crash cleanup ─────────────────────────

@pytest.fixture
def fake_run(pipeline, tmp_path, monkeypatch):
    """main() with the models and per-video detection replaced by fakes."""
    monkeypatch.delenv("LOCAL_VIDEO_ROOT", raising=False)
    videos = tmp_path / "videos"
    videos.mkdir()
    pipeline.MISSION_ID   = "M1"
    pipeline.OUTPUT_DIR   = tmp_path / "out" / "M1"
    pipeline.SOURCE       = LocalSource(str(videos))
    pipeline.MISSION_PATH = "."
    pipeline.START_FROM   = None
    pipeline.SELECTED_VIDEOS = None
    monkeypatch.setattr(pipeline, "load_model_and_config", lambda: (None, None))
    monkeypatch.setattr(pipeline, "load_embedding_model", lambda: ("finetuned", None, None))
    monkeypatch.setattr(pipeline, "get_embeddings_batch",
                        lambda info, imgs: np.ones((len(imgs), 4), dtype=np.float32))
    processed = []

    def fake_process_video(video_path, media_id, video_name, model, tracker, csv_path, crops_dir, **_):
        processed.append(media_id)
        new = not csv_path.exists()
        with open(csv_path, "a") as f:
            f.write(DETECTIONS_HEADER if new else "")
            if "fish" in video_name:
                f.write(f"{media_id},0,1,fish,0,0,0,0,0.9\n")
        if "fish" not in video_name:
            return {}
        crop = crops_dir / video_name / "id0001_fish_f00000_mid.jpg"
        crop.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(crop), np.zeros((8, 8, 3), dtype=np.uint8))
        return {1: [str(crop)]}

    monkeypatch.setattr(pipeline, "process_video", fake_process_video)
    for name in ("empty.mp4", "fish.MP4", "zz_empty.mov"):
        (videos / name).write_bytes(b"")
    return SimpleNamespace(videos=videos, processed=processed, paths=pipeline.output_paths())


def test_main_records_completion_and_resume_skips_everything(pipeline, fake_run):
    pipeline.main()
    assert fake_run.processed == ["empty.mp4", "fish.MP4", "zz_empty.mov"]
    manifest = json.loads(fake_run.paths.manifest.read_text())
    assert {k: (v["status"], v["n_tracks"], v["weights"]) for k, v in manifest.items()} == {
        "empty": ("complete", 0, "finetuned"),
        "fish": ("complete", 1, "finetuned"),
        "zz_empty": ("complete", 0, "finetuned"),
    }
    assert all(v["finished_at"] for v in manifest.values())
    assert set(json.loads(fake_run.paths.embeddings.read_text())) == {"fish__1"}
    assert "-1" not in fake_run.paths.csv.read_text()            # no sentinel rows any more

    fake_run.processed.clear()
    pipeline.main()
    assert fake_run.processed == []                              # trailing zero-track clip not redone


def test_main_redoes_unfinished_video_cleanly(pipeline, fake_run):
    pipeline.main()
    # simulate a run killed while processing fish.MP4 a second time
    manifest = json.loads(fake_run.paths.manifest.read_text())
    manifest["fish"]["status"] = "in_progress"
    fake_run.paths.manifest.write_text(json.dumps(manifest))
    with open(fake_run.paths.csv, "a") as f:
        f.write("fish.MP4,5,9,fish,0,0,0,0,0.9\nfish.MP4,6,9,fi")
    (fake_run.paths.crops / "fish" / "id0009_fish_f00005_mid.jpg").write_bytes(b"partial")

    fake_run.processed.clear()
    pipeline.main()
    assert fake_run.processed == ["fish.MP4"]
    rows = fake_run.paths.csv.read_text().splitlines()
    assert [r for r in rows if r.startswith("fish")] == ["fish.MP4,0,1,fish,0,0,0,0,0.9"]
    assert sorted(p.name for p in (fake_run.paths.crops / "fish").iterdir()) == ["id0001_fish_f00000_mid.jpg"]
    assert json.loads(fake_run.paths.manifest.read_text())["fish"]["status"] == "complete"


def test_main_stops_on_stem_collision(pipeline, fake_run):
    (fake_run.videos / "fish.mov").write_bytes(b"")
    with pytest.raises(ValueError, match="fish.MP4 and fish.mov"):
        pipeline.main()
    assert fake_run.processed == [] and not fake_run.paths.dir.exists()


def test_main_refuses_placeholder_embeddings(pipeline, fake_run, monkeypatch):
    monkeypatch.setattr(pipeline, "load_embedding_model", lambda: ("placeholder", None, None))
    with pytest.raises(RuntimeError, match="random placeholders"):
        pipeline.main()
    assert fake_run.processed == []


# ── Model downloads (gdown) ───────────────────────────────────────────────────

class _FakeVisual:
    def __init__(self):
        self.loaded = None

    def load_state_dict(self, state):
        self.loaded = state


class _FakeClipModel:
    def __init__(self):
        self.visual = _FakeVisual()

    def eval(self):
        return self


@pytest.fixture
def embed_env(pipeline, tmp_path, monkeypatch):
    calls = []
    fake_open_clip = SimpleNamespace(
        create_model_and_transforms=lambda *a, **k: (_FakeClipModel(), None, "preprocess"))
    monkeypatch.setitem(__import__("sys").modules, "open_clip", fake_open_clip)
    monkeypatch.setattr(pipeline, "CHECKPOINT_PATH", tmp_path / "checkpoints" / "best.pt")
    return SimpleNamespace(calls=calls, ckpt=tmp_path / "checkpoints" / "best.pt")


def test_checkpoint_downloaded_by_file_id(pipeline, embed_env, monkeypatch):
    import torch

    def fake_download(*args, **kwargs):
        embed_env.calls.append((args, kwargs))
        torch.save({"model_state": {"w": 1}}, kwargs["output"])
        return kwargs["output"]

    monkeypatch.setattr(pipeline.gdown, "download", fake_download)
    weights, model, _ = pipeline.load_embedding_model()
    assert embed_env.calls == [((), {"id": "1Qec1Gzxl8zAazuLof8a2LUSrbOR7GmtT",
                                     "output": str(embed_env.ckpt), "quiet": False})]
    assert weights == "finetuned" and model.visual.loaded == {"w": 1}


def test_checkpoint_download_code_error_is_not_swallowed(pipeline, embed_env, monkeypatch):
    def broken(*args, **kwargs):
        raise TypeError("download() got an unexpected keyword argument 'fuzzy'")
    monkeypatch.setattr(pipeline.gdown, "download", broken)
    with pytest.raises(TypeError):
        pipeline.load_embedding_model()


@pytest.mark.parametrize("error", ["oserror", "gdown"])
def test_checkpoint_network_failure_falls_back_to_pretrained(pipeline, embed_env, monkeypatch, error):
    def failing(*args, **kwargs):
        (embed_env.ckpt.parent / "best.ptab12.part").write_bytes(b"partial")
        if error == "gdown":
            raise pipeline.gdown.exceptions.FileURLRetrievalError("Too many users have viewed this file")
        raise ConnectionError("network unreachable")
    monkeypatch.setattr(pipeline.gdown, "download", failing)
    weights, model, _ = pipeline.load_embedding_model()
    assert weights == "pretrained" and model.visual.loaded is None
    assert list(embed_env.ckpt.parent.iterdir()) == []


def test_corrupted_checkpoint_reported_clearly(pipeline, embed_env, monkeypatch):
    embed_env.ckpt.parent.mkdir(parents=True)
    embed_env.ckpt.write_bytes(b"not a checkpoint")
    monkeypatch.setattr(pipeline.gdown, "download", lambda *a, **k: pytest.fail("must not download"))
    with pytest.raises(RuntimeError, match=r"(?s)best\.pt.*Delete it"):
        pipeline.load_embedding_model()


def test_yolo_world_downloaded_by_file_id(pipeline, tmp_path, monkeypatch):
    calls = []

    class FakeYOLO:
        def __init__(self, path):
            self.model = SimpleNamespace(eval=lambda: None)

        def set_classes(self, classes):
            pass

    monkeypatch.setattr(pipeline, "YOLO_CACHE_DIR", tmp_path / "ultralytics")
    monkeypatch.setattr(pipeline, "settings", SimpleNamespace(update=lambda d: None))
    monkeypatch.setattr(pipeline, "YOLOWorld", FakeYOLO)
    monkeypatch.setattr(pipeline.gdown, "download", lambda *a, **k: calls.append((a, k)))
    pipeline.load_model_and_config()
    assert calls == [((), {"id": "1hh576zOzpUqgWSjIdSkR4EwgYQ8kH406",
                           "output": str(tmp_path / "ultralytics" / "yolov8x-worldv2.pt"), "quiet": False})]
