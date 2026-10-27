"""
Where mission videos come from: a Dropbox folder or a local/external drive.

A deployment's source is configured in its deployment.json:
    {"source": {"type": "local",   "root": "Palau_2026"}}
    {"source": {"type": "dropbox", "root": "/DOEX0096_Palau"}}
Older files with only "dropbox_root" are treated as Dropbox.

Both sources speak in folders/paths *relative to the source root* (a Dropbox
folder may also be given as an absolute "/..." path, as before), and hand back
a local file to read via `local_copy()`:
  - LocalSource yields the file itself (never copied)
  - DropboxSource downloads it to a tempfile and deletes it afterwards

Local roots: if LOCAL_VIDEO_ROOT is set (Docker: /data/videos), roots are
relative to it and may not escape it. If unset, roots must be absolute paths.
"""

import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

# .mov is read by both ffmpeg (resize/proxy) and OpenCV's bundled FFmpeg backend.
VIDEO_EXTENSIONS = (".mp4", ".mov")


def is_video_name(name: str) -> bool:
    # Skip hidden files, incl. macOS "._foo.mp4" metadata files on exFAT drives
    return not name.startswith(".") and name.lower().endswith(VIDEO_EXTENSIONS)


def video_stem(name: str) -> str:
    """'PLW_dscm_03_c001.MP4' → 'PLW_dscm_03_c001' (other names unchanged)."""
    return Path(name).stem if name.lower().endswith(VIDEO_EXTENSIONS) else name


@dataclass(frozen=True)
class Video:
    name: str      # original filename, e.g. "PLW_dscm_03_c001.MP4"
    rel_path: str  # POSIX path relative to the source root ("/..." = absolute Dropbox path)

    @property
    def stem(self) -> str:
        return video_stem(self.name)


# ── Deployment config ─────────────────────────────────────────────────────────

def source_config(deployment: dict) -> dict:
    """The deployment's {"type", "root"}; legacy files with only dropbox_root → Dropbox."""
    src = deployment.get("source")
    if src and src.get("type") in ("local", "dropbox"):
        return {"type": src["type"], "root": src.get("root", "")}
    return {"type": "dropbox", "root": deployment.get("dropbox_root", "")}


def make_source(config: dict):
    if config.get("type") == "local":
        return LocalSource(config.get("root", ""))
    return DropboxSource(config.get("root", ""))


# ── Local folder ──────────────────────────────────────────────────────────────

def local_video_root():
    value = os.environ.get("LOCAL_VIDEO_ROOT", "").strip()
    return Path(value) if value else None


def resolve_local_root(root: str) -> Path:
    """Turn a user-entered local root into an absolute path, or raise ValueError."""
    root = (root or "").strip()
    if not root:
        raise ValueError("Local folder path is required")
    base = local_video_root()
    if base is not None:
        base = base.resolve()
        path = (base / root).resolve()
        if not path.is_relative_to(base):
            raise ValueError(f"{root!r} is outside LOCAL_VIDEO_ROOT ({base})")
        return path
    path = Path(root).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{root!r} must be an absolute path (LOCAL_VIDEO_ROOT is not set)")
    return path.resolve()


class LocalSource:
    type = "local"

    def __init__(self, root: str):
        self.root = resolve_local_root(root)

    def default_mission_folder(self, mission_id: str) -> str:
        return f"{mission_id}/Video"

    def _resolve(self, rel: str) -> Path:
        path = (self.root / rel).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"{rel!r} is outside the deployment folder {self.root}")
        return path

    def describe(self, folder: str) -> str:
        return str(self._resolve(folder))

    def check_folder(self, folder: str):
        path = self._resolve(folder)
        if not path.is_dir():
            raise FileNotFoundError(f"Video folder not found: {path}")

    def list_videos(self, folder: str) -> list[Video]:
        self.check_folder(folder)
        path = self._resolve(folder)
        return sorted(
            (Video(p.name, p.relative_to(self.root).as_posix())
             for p in path.iterdir() if p.is_file() and is_video_name(p.name)),
            key=lambda v: v.name,
        )

    @contextmanager
    def local_copy(self, video: Video):
        path = self._resolve(video.rel_path)
        if not path.is_file():
            raise FileNotFoundError(f"Video not found: {path}")
        yield path


# ── Dropbox ───────────────────────────────────────────────────────────────────

_DROPBOX_ENV = ("DROPBOX_APP_KEY", "DROPBOX_APP_SECRET", "DROPBOX_REFRESH_TOKEN")


def connect_dropbox():
    missing = [k for k in _DROPBOX_ENV if not os.environ.get(k)]
    if missing:
        raise RuntimeError(
            f"Dropbox credentials missing ({', '.join(missing)}). "
            "Add them to .env, or use a local-folder deployment."
        )
    import dropbox
    dbx = dropbox.Dropbox(
        app_key=os.environ["DROPBOX_APP_KEY"],
        app_secret=os.environ["DROPBOX_APP_SECRET"],
        oauth2_refresh_token=os.environ["DROPBOX_REFRESH_TOKEN"],
    )
    try:
        dbx.files_list_folder("")
        print("Dropbox connection OK")
    except Exception as e:
        raise RuntimeError(f"Dropbox connection failed: {e}")
    return dbx


class DropboxSource:
    type = "dropbox"

    def __init__(self, root: str, dbx=None):
        self.root = (root or "").rstrip("/")
        self._dbx = dbx

    @property
    def dbx(self):
        if self._dbx is None:
            self._dbx = connect_dropbox()
        return self._dbx

    def default_mission_folder(self, mission_id: str) -> str:
        return f"{mission_id}/Video"

    def _abs(self, rel: str) -> str:
        return rel if rel.startswith("/") else f"{self.root}/{rel}"

    def _rel(self, abs_path: str) -> str:
        prefix = f"{self.root}/"
        return abs_path[len(prefix):] if self.root and abs_path.startswith(prefix) else abs_path

    def describe(self, folder: str) -> str:
        return f"Dropbox:{self._abs(folder)}"

    def list_videos(self, folder: str) -> list[Video]:
        import dropbox
        folder = self._abs(folder)
        result  = self.dbx.files_list_folder(folder)
        entries = list(result.entries)
        while result.has_more:
            result = self.dbx.files_list_folder_continue(result.cursor)
            entries.extend(result.entries)
        return sorted(
            (Video(e.name, self._rel(str(PurePosixPath(folder) / e.name)))
             for e in entries
             if isinstance(e, dropbox.files.FileMetadata) and is_video_name(e.name)),
            key=lambda v: v.name,
        )

    @contextmanager
    def local_copy(self, video: Video):
        tmpdir = Path(tempfile.mkdtemp(prefix="natgeo_dl_"))
        try:
            local_path = tmpdir / video.name
            self.dbx.files_download_to_file(str(local_path), self._abs(video.rel_path))
            yield local_path
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ── Manifest ──────────────────────────────────────────────────────────────────
# processed_missions/<deployment>/<mission>/<mission>_manifest.json
#   {video_stem: {"source": "local"|"dropbox", "path": <rel to source root>, "filename": ...}}

def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def write_manifest(path: Path, manifest: dict):
    """Atomic write: temp file in the same folder, then rename over the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.stem}_", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(manifest, f, indent=2)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def manifest_entry(source, video: Video) -> dict:
    return {"source": source.type, "path": video.rel_path, "filename": video.name}
