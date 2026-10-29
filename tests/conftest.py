import json
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "application"))
sys.path.insert(0, str(REPO))

# hdbscan ships compiled DLLs that some Windows machines block (Smart App Control).
# The tests here don't depend on HDBSCAN's exact clustering, so fall back to
# scikit-learn's implementation when the real package can't be imported.
try:
    import hdbscan  # noqa: F401
except ImportError:
    from sklearn.cluster import HDBSCAN as _SkHDBSCAN

    class _HDBSCAN:
        def __init__(self, min_cluster_size=5, min_samples=None, metric="euclidean", **_):
            self._model = _SkHDBSCAN(min_cluster_size=min_cluster_size,
                                     min_samples=min_samples, metric=metric)

        def fit_predict(self, X):
            return self._model.fit_predict(X)

    sys.modules["hdbscan"] = types.SimpleNamespace(HDBSCAN=_HDBSCAN)


DETECTIONS_HEADER = "media_id,frame,track_id,class,x,y,width,height,confidence\n"


@pytest.fixture
def app_env(tmp_path, monkeypatch):
    """The Flask app pointed at an empty processed_missions/ under tmp_path."""
    import app as app_module

    root = tmp_path / "processed_missions"
    root.mkdir()
    monkeypatch.setattr(app_module, "_deployments_root", lambda: root)
    monkeypatch.setattr(app_module, "_current_deployment_id", None)
    monkeypatch.setattr(app_module, "_review_mission_id", None)
    monkeypatch.setattr(app_module, "DETECTIONS_CACHE", {})
    monkeypatch.setattr(app_module, "_proxy_status", {})
    monkeypatch.setitem(app_module._pipeline_state, "running", False)
    app_module.app.config["TESTING"] = True

    def make_mission(deployment_id, mission_id, embeddings=None, csv_text=None, manifest=None,
                     source=None):
        dep = root / deployment_id
        dep.mkdir(exist_ok=True)
        (dep / "deployment.json").write_text(json.dumps({
            "deployment_id": deployment_id,
            "source": source or {"type": "dropbox", "root": f"/{deployment_id}"},
        }))
        m = dep / mission_id
        (m / f"{mission_id}_crops").mkdir(parents=True)
        (m / f"{mission_id}_embeddings.json").write_text(json.dumps(embeddings or {}))
        if csv_text is not None:
            (m / f"{mission_id}_detections.csv").write_text(DETECTIONS_HEADER + csv_text)
        if manifest is not None:
            (m / f"{mission_id}_manifest.json").write_text(json.dumps(manifest))
        return m

    def select(deployment_id, mission_id):
        monkeypatch.setattr(app_module, "_current_deployment_id", deployment_id)
        monkeypatch.setattr(app_module, "_review_mission_id", mission_id)
        monkeypatch.setattr(app_module, "DETECTIONS_CACHE",
                            app_module._load_deployment_detections(deployment_id))

    return types.SimpleNamespace(
        app=app_module, client=app_module.app.test_client(), root=root,
        make_mission=make_mission, select=select,
    )
