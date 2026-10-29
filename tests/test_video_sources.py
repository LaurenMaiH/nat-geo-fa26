import pytest

from video_sources import (
    Video, check_stem_collisions, completed_videos, has_completion_records, is_complete,
    stem_collisions, video_stem,
)


@pytest.mark.parametrize("name, stem", [
    ("PLW_dscm_03_c001.mp4", "PLW_dscm_03_c001"),
    ("PLW_dscm_03_c001.MP4", "PLW_dscm_03_c001"),
    ("clip.MOV", "clip"),
    ("clip.Mov", "clip"),
    ("clip.mp4.mov", "clip.mp4"),
    ("PLW_dscm_03_c001", "PLW_dscm_03_c001"),   # already a stem
    ("notes.txt", "notes.txt"),                 # not a video: unchanged
])
def test_video_stem(name, stem):
    assert video_stem(name) == stem


def test_stem_collisions_names_the_files():
    videos = [Video(n, n) for n in ("a.mp4", "a.MOV", "b.mp4", "B.mov", "c.mp4")]
    assert stem_collisions(videos) == [["a.MOV", "a.mp4"], ["B.mov", "b.mp4"]]
    with pytest.raises(ValueError, match=r"a\.MOV and a\.mp4.*B\.mov and b\.mp4"):
        check_stem_collisions(videos)


def test_no_collisions():
    check_stem_collisions([Video(n, n) for n in ("a.mp4", "b.MOV")])


def test_completion_records():
    manifest = {
        "a": {"filename": "a.mp4", "status": "complete", "n_tracks": 3},
        "b": {"filename": "b.mp4", "status": "complete", "n_tracks": 0},
        "c": {"filename": "c.mp4", "status": "in_progress"},
        "d": {"filename": "d.mp4", "status": "failed", "error": "boom"},
    }
    assert completed_videos(manifest) == {"a": 3, "b": 0}
    assert is_complete(manifest["b"]) and not is_complete(manifest["c"]) and not is_complete(None)
    assert has_completion_records(manifest)
    assert not has_completion_records({"a": {"filename": "a.mp4", "path": "x/a.mp4"}})
    assert not has_completion_records({})
