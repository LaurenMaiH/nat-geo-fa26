"""
Extract whole-frame stills from Pristine Seas dropcam videos for every row of a
data-summary spreadsheet (e.g. SLB_DOEX0102_Data_Summary).

Each row has a `deployment` (SLB_2024_dscm_18) and an `observationTimestamp` (UTC).
For each row the script:
  1. finds the deployment's folder anywhere under the expedition, tolerating the
     naming differences between expeditions (SLB_2024_dscm_01 vs SLB-2024-dscm-01,
     HAW vs Haw, Video vs VIdeo vs VIDEO, deployments/ sub-folders, ...),
  2. builds a clip table (start time + duration) for every video in it, cached to
     disk so it is only built once per deployment. Start times come from, in order:
       xml        the Sony .XML sidecar next to each clip (2021+ expeditions)
       container  the creation_time stored inside the MP4/MOV (needs ffmpeg)
     Clips with neither (e.g. 2011-2017 AVCHD .MTS) are reported, not guessed,
  3. matches the timestamp to the clip that was recording at that moment,
  4. extracts a frame at that moment, streamed from a Dropbox temporary link with
     ffmpeg (a few MB instead of the ~1 GB clip), falling back to a full download,
  5. writes the frames plus a manifest CSV that links every frame to its sheet row.

Clock sanity checks. Camera clocks are sometimes wrong (SLB 2024 has deployments
whose camera *and* sheet say 2028). The sheet and the XML share the same clock,
so timestamps are matched as-is. The dry run checks each deployment's
cameraSeafloorArrival against its clips and flags deployments where nothing lines
up, including when a whole-year shift would fix it.

ALWAYS run with --dry-run first. It matches every row and writes match_report.csv
and deployment_report.csv without extracting anything.

Examples
  python frame_extraction.py --sheet SLB_DOEX0102_Data_Summary.csv --dry-run
  python frame_extraction.py --sheet SLB_DOEX0102_Data_Summary.csv --limit 3
  python frame_extraction.py --sheet sheet.csv --offsets=-2,0,2
  python frame_extraction.py --sheet sheet.csv --expedition DOEX0112

Requires: ffmpeg (on PATH, or `pip install imageio-ffmpeg`), and
          pip install dropbox python-dotenv tqdm openpyxl
Reads DROPBOX_APP_KEY / DROPBOX_APP_SECRET / DROPBOX_REFRESH_TOKEN from .env
"""

import argparse
import bisect
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath

TOP = "/Pristine Seas Dropcam Data"

VIDEO_EXTS = {".mp4", ".mov", ".mts", ".m2ts", ".mkv", ".avi", ".mxf", ".m4v"}
# folders that hold copies/edits rather than the raw camera clips
SKIP_DIR_RE = re.compile(r"highlight|selects|proxy|tator|premiere|edited|^html$|^mnt$|^disk$|"
                         r"^misc$|\.appledouble", re.I)

LINEAGE_COLS = [
    "scientificName", "taxonRank", "aphiaId", "phylum", "class", "subclass", "order",
    "suborder", "family", "subfamily", "genus", "subgenus", "species", "subspecies",
    "morphospecies", "individualCount", "categoricalAbundance", "identificationRemarks",
]


# ── time helpers ──────────────────────────────────────────────────────────────
def parse_ts(value):
    """Parse '2024-10-18 07:04:16Z', '2024-10-18T07:04:16Z', or a datetime -> aware UTC."""
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        s = str(value).strip()
        if not s or s in {"--", "NA", "nan", "NaN"}:
            return None
        s = s.replace("T", " ").replace("Z", "").strip()
        s = re.sub(r"\.\d+", "", s)  # drop fractional seconds
        s = re.sub(r"([+-]\d\d)(\d\d)$", r"\1:\2", s)  # +0000 -> +00:00 for fromisoformat
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def canon(name):
    """Normalise a deployment name: SLB-2024-dscm-01 / SLB_2024_DSCM_1 -> slb_2024_dscm_1."""
    s = re.sub(r"[\s\-]+", "_", name.strip().lower())
    return re.sub(r"\d+", lambda m: str(int(m.group())), s)


def canon_no_year(name):
    """Same as canon() but without a 4-digit year token: NIU_2023_dscm_02 -> niu_dscm_2."""
    return re.sub(r"_(19|20)\d\d(?=_|$)", "", canon(name))


def slug(text, n=40):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_")[:n] or "unknown"


def year_shift(dt, years):
    try:
        return dt.replace(year=dt.year + years)
    except ValueError:  # 29 Feb
        return dt.replace(year=dt.year + years, day=28)


# ── spreadsheet ───────────────────────────────────────────────────────────────
KNOWN_COLS = ["deployment", "observationTimestamp", "cameraSeafloorArrival", "tentativeId"] + LINEAGE_COLS


def _normalise_headers(rows):
    """Rename columns to the spelling the script uses, ignoring case, so older sheets with
    'Deployment' / 'ObservationTimestamp' / 'AphiaID' headers work the same as newer ones."""
    if not rows:
        return rows
    canon_map = {c.lower(): c for c in KNOWN_COLS}
    rename = {h: canon_map[h.strip().lower()] for h in rows[0]
              if h and h.strip().lower() in canon_map and h != canon_map[h.strip().lower()]}
    if not rename:
        return rows
    print(f"Renamed sheet columns: {', '.join(f'{a} -> {b}' for a, b in rename.items())}")
    return [{rename.get(k, k): v for k, v in r.items()} for r in rows]


def read_sheet(path):
    return _normalise_headers(_read_sheet_raw(path))


def _read_sheet_raw(path):
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xlsm"):
        try:
            import openpyxl
        except ImportError:
            sys.exit("Reading .xlsx needs openpyxl:  python -m pip install openpyxl")
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        rows = wb.worksheets[0].iter_rows(values_only=True)
        header = [str(h).strip() if h is not None else "" for h in next(rows)]
        return [dict(zip(header, r)) for r in rows if any(v is not None for v in r)]
    with open(path, newline="", encoding="utf-8-sig") as f:
        first = f.readline()
        f.seek(0)
        delim = "\t" if first.count("\t") > first.count(",") else ","
        return list(csv.DictReader(f, delimiter=delim))


# ── ffmpeg ────────────────────────────────────────────────────────────────────
def find_ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def probe(ffmpeg, src, timeout=120):
    """Read creation_time and duration from a video's container header (no ffprobe needed)."""
    try:
        r = subprocess.run([ffmpeg, "-hide_banner", "-i", src], capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, None
    ct = re.search(r"creation_time\s*:\s*(\S+)", r.stderr)
    du = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", r.stderr)
    dur = int(du[1]) * 3600 + int(du[2]) * 60 + float(du[3]) if du else None
    return (parse_ts(ct[1]) if ct else None), dur


def run_ffmpeg(ffmpeg, src, offset, out_path, max_width=None, timeout=900):
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{offset:.3f}", "-i", src,
           "-frames:v", "1", "-q:v", "2"]
    if max_width:
        cmd += ["-vf", f"scale='min({int(max_width)},iw)':-2"]
    cmd.append(str(out_path))
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "timeout"
    ok = r.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0
    if not ok and out_path.exists():
        out_path.unlink()
    return ok, r.stderr.strip()[-300:]


# ── Dropbox ───────────────────────────────────────────────────────────────────
def connect():
    import dropbox
    from dotenv import load_dotenv

    load_dotenv(Path.cwd() / ".env")
    load_dotenv(Path(__file__).resolve().parent / ".env")
    for k in ("DROPBOX_APP_KEY", "DROPBOX_APP_SECRET", "DROPBOX_REFRESH_TOKEN"):
        if k not in os.environ:
            sys.exit(f"{k} is not set. Check that your .env is found and has all 3 values.")
    return dropbox.Dropbox(
        app_key=os.environ["DROPBOX_APP_KEY"],
        app_secret=os.environ["DROPBOX_APP_SECRET"],
        oauth2_refresh_token=os.environ["DROPBOX_REFRESH_TOKEN"],
    )


def list_folder(dbx, path, recursive=False):
    """Return [{"p": path, "s": size}] for files and [{"p": path, "d": 1}] for folders."""
    import dropbox

    res = dbx.files_list_folder(path, recursive=recursive, limit=2000)
    out = []
    while True:
        for e in res.entries:
            if isinstance(e, dropbox.files.FileMetadata):
                out.append({"p": e.path_display, "s": e.size})
            elif isinstance(e, dropbox.files.FolderMetadata):
                out.append({"p": e.path_display, "d": 1})
        if not res.has_more:
            return out
        res = dbx.files_list_folder_continue(res.cursor)


def expedition_listing(dbx, expedition, cache_dir, refresh=False):
    """Recursive listing of one expedition, cached as JSON (same format as dropbox_survey.py)."""
    cache = Path(cache_dir) / f"{expedition}.json"
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())
    print(f"  listing {expedition} (first time only, can take a few minutes for big ones)...")
    entries = list_folder(dbx, f"{TOP}/{expedition}", recursive=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(entries))
    return entries


def index_deployments(expedition, entries):
    """Map canonical deployment name -> {path, videos: {folder: [files]}, xml: {folder: {stem: path}}}.

    A deployment folder is the shallowest folder whose own video files (or those of
    its non-skipped sub-folders) are raw clips. Works for <exp>/<dep>/Video/*.mp4,
    <exp>/deployments/<dep>/Video/*.mp4, <exp>/<dep>/Video/AVCHD/BDMV/STREAM/*.MTS, ...
    """
    root = PurePosixPath(f"{TOP}/{expedition}")
    deps = {}
    for e in entries:
        if "d" in e:
            continue
        p = PurePosixPath(e["p"])
        if p.name.startswith("._"):
            continue
        ext = p.suffix.lower()
        if ext not in VIDEO_EXTS and ext != ".xml":
            continue
        rel = p.relative_to(root).parts[:-1]
        if not rel or any(SKIP_DIR_RE.search(seg) for seg in rel):
            continue
        # deployment = first segment that isn't a generic grouping folder
        i = 0
        while i < len(rel) - 1 and rel[i].lower() in {"deployments", "deployment", "dropcam", "drops"}:
            i += 1
        dep_path = str(root.joinpath(*rel[: i + 1]))
        d = deps.setdefault(canon(rel[i]), {"name": rel[i], "path": dep_path,
                                            "videos": defaultdict(list), "xml": defaultdict(dict)})
        if ext == ".xml":
            d["xml"][str(p.parent)][p.stem.lower()] = str(p)
        else:
            d["videos"][str(p.parent)].append({"path": str(p), "bytes": e.get("s", 0)})
    return {k: v for k, v in deps.items() if v["videos"]}


def find_expeditions(dbx, hint):
    names = sorted(e["p"].rsplit("/", 1)[1] for e in list_folder(dbx, TOP) if "d" in e)
    exps = [n for n in names if n.upper().startswith("DOEX")]
    if hint:
        exps = [n for n in exps if hint.lower() in n.lower()]
    return exps


def locate_deployments(dbx, deps, hint, cache_dir):
    """Find each sheet deployment's folder. Searches the hinted expedition first, then the rest."""
    exps = find_expeditions(dbx, hint)
    if not exps:
        sys.exit(f"No expedition folder under {TOP} matches '{hint}'.")
    cached = [x for x in exps if (Path(cache_dir) / f"{x}.json").exists()]
    exps = cached + [x for x in exps if x not in cached]
    found, todo = {}, set(deps)
    for exp in exps:
        if not todo:
            break
        index = index_deployments(exp, expedition_listing(dbx, exp, cache_dir))
        by_no_year = defaultdict(list)
        for k, v in index.items():
            by_no_year[canon_no_year(v["name"])].append(v)
        for dep in sorted(todo):
            hit = index.get(canon(dep))
            if hit is None:
                cands = by_no_year.get(canon_no_year(dep), [])
                hit = cands[0] if len(cands) == 1 else None
            if hit:
                found[dep] = {**hit, "expedition": exp}
                todo.discard(dep)
    return found


# ── clip tables ───────────────────────────────────────────────────────────────
def _xml_el(root, name):
    for el in root.iter():
        if el.tag.split("}")[-1] == name:
            return el
    return None


def with_retries(fn, tries=5, base_wait=2):
    """Call fn(); on a network error wait 2, 4, 8, 16 s and try again. Re-raises the last error."""
    for attempt in range(tries):
        try:
            return fn()
        except Exception:  # noqa: BLE001
            if attempt == tries - 1:
                raise
            time.sleep(base_wait * 2 ** attempt)


def _parse_xml(content):
    root = ET.fromstring(content)
    start = parse_ts(_xml_el(root, "CreationDate").get("value"))
    frames = int(_xml_el(root, "Duration").get("value"))
    ltc = _xml_el(root, "LtcChangeTable")
    fps = float(ltc.get("tcFps")) if ltc is not None else 30.0
    return start, frames / fps


def _clip_from_xml(dbx, xml_path):
    _, resp = with_retries(lambda: dbx.files_download(xml_path))
    return _parse_xml(resp.content)


def _xml_for(dep, video_path):
    p = PurePosixPath(video_path)
    return dep["xml"].get(str(p.parent), {}).get(p.stem.lower())


def build_clip_table(dbx, dep, cache_csv, ffmpeg=None, progress=None):
    """Return clips as dicts: clip, path, start (aware UTC or None), dur (s), bytes, source.

    Cached as CSV. Clips with no usable start time are kept with start=None so the
    report can say how many there were.

    Network safety: a clip whose XML sidecar exists but could not be downloaded (after
    retries) gets source="error", and a table containing any "error" clip is NOT cached,
    so the next run rebuilds it. A cached table that has a clip missing its time even
    though an XML exists for it (written by an older version during a network drop) is
    ignored and rebuilt.
    """
    if cache_csv.exists():
        with open(cache_csv, newline="") as f:
            cached = [{"clip": r["clip"], "path": r["path"], "start": parse_ts(r["start_utc"]),
                       "dur": float(r["dur_s"]) if r["dur_s"] else None, "bytes": int(r["bytes"] or 0),
                       "source": r.get("source", "xml")}
                      for r in csv.DictReader(f)]
        stale = [c for c in cached if c["source"] not in ("xml", "container") and _xml_for(dep, c["path"])]
        if not stale:
            return cached
        print(f"  {dep['name']}: cached clip table has {len(stale)} clip(s) with an XML but no time "
              f"(likely a past network drop) - rebuilding")

    def one(item):
        folder, v = item
        p = PurePosixPath(v["path"])
        start = dur = None
        source = "none"
        xml = dep["xml"].get(folder, {}).get(p.stem.lower())
        if xml:
            try:
                _, resp = with_retries(lambda: dbx.files_download(xml))
            except Exception as ex:  # noqa: BLE001
                # network failure: don't fall back to the container clock, mark it so it isn't cached
                print(f"  ! could not download {PurePosixPath(xml).name} after retries: {ex}")
                return {"clip": p.stem, "path": v["path"], "start": None, "dur": None,
                        "bytes": v["bytes"], "source": "error"}
            try:
                start, dur = _parse_xml(resp.content)
                source = "xml"
            except Exception as ex:  # noqa: BLE001
                print(f"  ! bad XML {PurePosixPath(xml).name}: {ex}")
        if start is None and ffmpeg and p.suffix.lower() in {".mp4", ".mov", ".m4v"}:
            try:
                link = with_retries(lambda: dbx.files_get_temporary_link(v["path"]).link)
                start, dur = probe(ffmpeg, link)
                source = "container" if start else "none"
            except Exception:  # noqa: BLE001
                source = "error"
        return {"clip": p.stem, "path": v["path"], "start": start, "dur": dur,
                "bytes": v["bytes"], "source": source}

    from concurrent.futures import ThreadPoolExecutor

    vids = [(folder, v) for folder, vs in dep["videos"].items() for v in vs]
    with ThreadPoolExecutor(max_workers=4) as pool:  # 4, not 8: gentler on weak connections
        results = pool.map(one, vids)
        clips = list(progress(results, desc=f"  clip times {dep['name']}", total=len(vids))
                     if progress else results)
    clips.sort(key=lambda c: (c["start"] is None, c["start"] or datetime.min.replace(tzinfo=timezone.utc)))
    n_err = sum(1 for c in clips if c["source"] == "error")
    if n_err:
        print(f"\n  ! {dep['name']}: {n_err} clip(s) could not be read because of network errors. "
              f"NOT caching this table - re-run the same command to retry them.")
        return clips
    # don't cache a table that is missing times only because ffmpeg wasn't available to probe
    if not ffmpeg and any(c["source"] == "none" for c in clips):
        return clips
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "path", "start_utc", "dur_s", "bytes", "source"])
        for c in clips:
            w.writerow([c["clip"], c["path"], c["start"].strftime("%Y-%m-%dT%H:%M:%SZ") if c["start"] else "",
                        f"{c['dur']:.3f}" if c["dur"] else "", c["bytes"], c["source"]])
    return clips


# ── matching ──────────────────────────────────────────────────────────────────
def match_time(clips, starts, t):
    """Return (clip, offset_s, None) if t is inside a clip, else (None, None, (gap_before_s, gap_after_s))."""
    i = bisect.bisect_right(starts, t) - 1
    if i >= 0:
        c = clips[i]
        off = (t - c["start"]).total_seconds()
        if off < c["dur"] - 0.1:
            return c, off, None
    before = None
    if i >= 0:
        before = (t - (clips[i]["start"] + timedelta(seconds=clips[i]["dur"]))).total_seconds()
    after = (clips[i + 1]["start"] - t).total_seconds() if i + 1 < len(clips) else None
    return None, None, (before, after)


def clock_check(clips, starts, times):
    """Return (rows inside a clip, best whole-year shift or None, rows inside a clip after that shift).

    Only looks for a shift when no row lands in a clip as-is: that is the signature of the
    sheet and the camera disagreeing about the year (e.g. sheet corrected to 2024, camera 2028).
    """
    hit = sum(1 for t in times if match_time(clips, starts, t)[0])
    if hit or not times or not clips:
        return hit, None, 0
    best, best_hit = None, 0
    for years in (-1, 1, -2, 2, -3, 3, -4, 4, -5, 5):
        n = sum(1 for t in times if match_time(clips, starts, year_shift(t, years))[0])
        if n > best_hit:
            best, best_hit = years, n
    return hit, best, best_hit


def mean_brightness(path):
    """Mean grey level 0-255 of an image (on a downscaled copy, so it's fast)."""
    from PIL import Image, ImageStat

    with Image.open(path) as im:
        im = im.convert("L")
        im.thumbnail((320, 320))
        return ImageStat.Stat(im).mean[0]


def extract_lit(ffmpeg, src, off, dur, out_path, max_width, dark, step=0.5, max_nudge=4.0):
    """Extract the frame at `off`; if it is dark (lights not on yet), step forward until it isn't.

    Dropcam clips start with a few seconds of black while the lights warm up, and the sheet's
    whole-second timestamps often land in it. Returns (ok, err, nudge_s, brightness).
    """
    ok, err = run_ffmpeg(ffmpeg, src, off, out_path, max_width)
    if not ok or dark is None:
        return ok, err, 0.0, None
    b = mean_brightness(out_path)
    nudge = 0.0
    while b < dark and nudge + step <= max_nudge and off + nudge + step < (dur or 1e9) - 0.1:
        nudge += step
        ok2, _ = run_ffmpeg(ffmpeg, src, off + nudge, out_path, max_width)
        if not ok2:
            break
        b = mean_brightness(out_path)
    if b < dark:  # never lit up: keep the frame at the original time rather than a random later one
        run_ffmpeg(ffmpeg, src, off, out_path, max_width)
        b, nudge = mean_brightness(out_path), 0.0
    return ok, err, nudge, b


def fmt_gap(sec):
    if sec is None:
        return "n/a"
    sec = abs(sec)
    return f"{sec/3600:.1f} h" if sec >= 3600 else f"{sec/60:.1f} min" if sec >= 60 else f"{sec:.0f} s"


# ── main ──────────────────────────────────────────────────────────────────────
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheet", required=True, help="spreadsheet (.xlsx, .csv or .tsv)")
    ap.add_argument("--out", default="extracted_frames", help="output folder")
    ap.add_argument("--expedition", help="expedition folder name or part of it, e.g. DOEX0102 "
                                         "(default: the DOEX#### in the sheet name, else search all)")
    ap.add_argument("--offsets", default="0",
                    help="seconds relative to the timestamp, comma-separated, e.g. -2,0,2 (default: 0). "
                         "Use --offsets=-2,0,2 with an equals sign when starting with a minus.")
    ap.add_argument("--fix-year", help="remap years in the sheet, e.g. 2028:2024. Usually NOT needed: "
                                       "the sheet and the clips share the same camera clock")
    ap.add_argument("--no-auto-year-shift", action="store_true",
                    help="don't auto-correct deployments whose sheet year differs from the camera's "
                         "(applied only when the shift puts >=90%% of that deployment's rows inside clips)")
    ap.add_argument("--max-width", type=int, help="downscale frames wider than this (e.g. 1920)")
    ap.add_argument("--dark-threshold", type=float, default=20,
                    help="frames darker than this mean grey level (0-255) are nudged forward up to 4 s "
                         "to get past the lights-off start of a clip; 0 disables (default: 20)")
    ap.add_argument("--limit", type=int, help="only process the first N sheet rows (for testing)")
    ap.add_argument("--download", action="store_true", help="skip streaming; always download clips")
    ap.add_argument("--dry-run", action="store_true", help="match rows and report; extract nothing")
    ap.add_argument("--cache-dir", default="clip_tables", help="where per-deployment clip tables are cached")
    ap.add_argument("--listing-cache", default="dropbox_survey/listings",
                    help="where recursive expedition listings are cached (shared with dropbox_survey.py)")
    ap.add_argument("--refresh", action="store_true", help="re-list Dropbox and rebuild clip tables")
    args = ap.parse_args(argv)

    ffmpeg = find_ffmpeg()
    if not args.dry_run and not ffmpeg:
        sys.exit("ffmpeg not found. Install it (brew install ffmpeg) or: python -m pip install imageio-ffmpeg")

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None
    progress = (lambda it, **kw: tqdm(it, leave=False, **kw)) if tqdm else None

    rows = read_sheet(args.sheet)
    if args.limit:
        rows = rows[:args.limit]
    year_fix = {}
    if args.fix_year:
        for pair in args.fix_year.split(","):
            a, b = pair.split(":")
            year_fix[int(a)] = int(b)
    offsets = [float(x) for x in args.offsets.split(",")]

    dbx = connect()
    hint = args.expedition or (re.search(r"DOEX\d+", Path(args.sheet).name) or [None])[0]
    deps = sorted({str(r.get("deployment") or "").strip() for r in rows} - {""})
    print(f"{len(rows)} sheet rows, {len(deps)} deployments. "
          f"Searching {'expeditions matching ' + repr(hint) if hint else 'all expeditions'}...")
    if args.refresh:
        for f in Path(args.listing_cache).glob(f"*{hint or ''}*.json"):
            f.unlink()
    located = locate_deployments(dbx, deps, hint, args.listing_cache)

    # clip tables + per-deployment checks
    first_by_dep = {}
    for r in rows:
        first_by_dep.setdefault(str(r.get("deployment") or "").strip(), r)
    tables, dep_report, dep_shift = {}, [], {}
    for dep in deps:
        info = located.get(dep)
        if not info:
            print(f"  {dep}: deployment folder NOT FOUND")
            tables[dep] = None
            dep_report.append({"deployment": dep, "status": "folder_not_found"})
            continue
        cache_csv = Path(args.cache_dir) / info["expedition"] / f"{canon(info['name'])}.csv"
        if args.refresh and cache_csv.exists():
            cache_csv.unlink()
        clips_all = build_clip_table(dbx, info, cache_csv, ffmpeg=ffmpeg, progress=progress)
        clips = [c for c in clips_all if c["start"] and c["dur"]]
        starts = [c["start"] for c in clips]
        tables[dep] = (clips, starts)
        sources = defaultdict(int)
        for c in clips_all:
            sources[c["source"]] += 1
        arrival = parse_ts(first_by_dep[dep].get("cameraSeafloorArrival"))
        obs = [t for t in (parse_ts(r.get("observationTimestamp")) for r in rows
                           if str(r.get("deployment") or "").strip() == dep) if t]
        hit, shift, shifted_hit = clock_check(clips, starts, obs)
        if shift and not args.no_auto_year_shift and shifted_hit >= 0.9 * len(obs):
            dep_shift[dep] = shift
            hit = shifted_hit
            status = f"year_shifted_{shift:+d}y"
        else:
            status = ("ok" if hit else
                      "no_clip_times" if not clips else
                      f"clock_off_by_{shift:+d}y" if shift else "no_rows_in_clips")
        if arrival and dep in dep_shift:
            arrival = year_shift(arrival, dep_shift[dep])
        arrival_ok = bool(arrival and clips and match_time(clips, starts, arrival)[0])
        flag = "" if status == "ok" else f"   <-- {status}"
        print(f"\r  {dep}: {info['path']}  [{len(clips_all)} clips, times from "
              f"{dict(sources)}; {hit}/{len(obs)} rows inside a clip]{flag}")
        dep_report.append({
            "deployment": dep, "status": status, "folder": info["path"], "expedition": info["expedition"],
            "n_clips": len(clips_all), "clips_with_time": len(clips),
            "time_sources": " ".join(f"{k}:{v}" for k, v in sorted(sources.items())),
            "first_clip_utc": starts[0].strftime("%Y-%m-%d %H:%M:%S") if starts else "",
            "last_clip_utc": starts[-1].strftime("%Y-%m-%d %H:%M:%S") if starts else "",
            "seafloor_arrival_in_clip": arrival_ok, "rows": len(obs), "rows_in_clip": hit,
            "suggested_year_shift": shift or "", "applied_year_shift": dep_shift.get(dep, ""),
        })

    # match every (row, offset)
    jobs = []
    for i, row in enumerate(rows, start=1):
        dep = str(row.get("deployment") or "").strip()
        t0 = parse_ts(row.get("observationTimestamp"))
        base = {"row": i, "dep": dep, "taxon": row.get("scientificName", ""), "src": row}
        if t0 is None:
            jobs.append({**base, "offset_s": "", "t": None, "status": "bad_or_missing_timestamp"})
            continue
        if t0.year in year_fix:
            t0 = t0.replace(year=year_fix[t0.year])
        if dep in dep_shift:
            t0 = year_shift(t0, dep_shift[dep])  # sheet time -> camera time
        if not tables.get(dep):
            jobs.append({**base, "offset_s": "", "t": t0, "status": "no_deployment_folder"})
            continue
        clips, starts = tables[dep]
        if not clips:
            jobs.append({**base, "offset_s": "", "t": t0, "status": "no_clip_times"})
            continue
        for o in offsets:
            t = t0 + timedelta(seconds=o)
            clip, off, gap = match_time(clips, starts, t)
            if clip:
                jobs.append({**base, "offset_s": o, "t": t, "status": "matched", "clip": clip, "off": off})
            else:
                jobs.append({**base, "offset_s": o, "t": t, "status": "in_gap",
                             "gap_before": gap[0], "gap_after": gap[1]})

    # reports
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dep_cols = ["deployment", "status", "expedition", "folder", "n_clips", "clips_with_time", "time_sources",
                "first_clip_utc", "last_clip_utc", "seafloor_arrival_in_clip", "rows", "rows_in_clip",
                "suggested_year_shift", "applied_year_shift"]
    with open(out / "deployment_report.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=dep_cols)
        w.writeheader()
        w.writerows(dep_report)
    with open(out / "match_report.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "deployment", "scientificName", "target_time_utc", "offset_s", "status",
                    "clip", "time_source", "offset_in_clip_s", "gap_before_s", "gap_after_s"])
        for j in jobs:
            c = j.get("clip")
            w.writerow([j["row"], j["dep"], j["taxon"],
                        j["t"].strftime("%Y-%m-%d %H:%M:%S") if j["t"] else "", j["offset_s"], j["status"],
                        c["clip"] if c else "", c["source"] if c else "", f"{j['off']:.2f}" if c else "",
                        f"{j['gap_before']:.0f}" if j.get("gap_before") is not None else "",
                        f"{j['gap_after']:.0f}" if j.get("gap_after") is not None else ""])

    n = defaultdict(int)
    for j in jobs:
        n[j["status"]] += 1
    matched = [j for j in jobs if j["status"] == "matched"]
    uniq = {j["clip"]["path"]: j["clip"] for j in matched}
    gb = sum(c["bytes"] for c in uniq.values()) / 1e9
    print("\n── Match summary ─────────────────────────────")
    print(f"sheet rows: {len(rows)}   frames requested: {len(jobs)}")
    for k, v in sorted(n.items()):
        print(f"  {k:28s} {v}")
    print(f"clips needed: {len(uniq)}   (whole-clip download would be ~{gb:.1f} GB)")
    for j in [j for j in jobs if j["status"] == "in_gap"][:5]:
        print(f"  e.g. row {j['row']} {j['dep']} {j['t']:%Y-%m-%d %H:%M:%S}: nearest clip ended "
              f"{fmt_gap(j['gap_before'])} before / next starts {fmt_gap(j['gap_after'])} after")
    bad = [d for d in dep_report if d["status"] != "ok"]
    if bad:
        print(f"deployments needing attention: {', '.join(d['deployment'] + ' (' + d['status'] + ')' for d in bad)}")
    print(f"Reports: {out / 'match_report.csv'}, {out / 'deployment_report.csv'}")
    if args.dry_run:
        print("Dry run: nothing extracted.")
        return jobs

    # extract
    def out_path(j):
        o = j["offset_s"]
        tag = f"{'m' if o < 0 else 'p'}{abs(o):g}s"
        return out / j["dep"] / f"{j['row']:04d}_{slug(j['taxon'])}_{j['t']:%Y%m%dT%H%M%S}_{tag}.jpg"

    dark = args.dark_threshold or None
    by_clip = defaultdict(list)
    for j in matched:
        by_clip[j["clip"]["path"]].append(j)
    results = {}
    with tempfile.TemporaryDirectory(prefix="frames_") as tmp:
        it = by_clip.items()
        if tqdm:
            it = tqdm(list(it), desc="clips", unit="clip")
        for path, js in it:
            link, link_t, local, streaming = None, 0.0, None, not args.download
            for j in js:
                p = out_path(j)
                p.parent.mkdir(parents=True, exist_ok=True)
                if p.exists() and p.stat().st_size > 0:
                    results[id(j)] = (p, "exists", "", "")
                    continue
                ok, err, nudge, bright = False, "", 0.0, None
                if streaming:
                    for attempt in range(3):  # retry blips with a fresh link before a full download
                        if link is None or attempt > 0 or time.time() - link_t > 3 * 3600:
                            link = with_retries(lambda: dbx.files_get_temporary_link(path).link)
                            link_t = time.time()
                        ok, err, nudge, bright = extract_lit(ffmpeg, link, j["off"], j["clip"]["dur"], p,
                                                             args.max_width, dark)
                        if ok:
                            break
                        time.sleep(5 * (attempt + 1))
                    if not ok:
                        streaming = False
                        print(f"\n  streaming failed for {j['clip']['clip']} ({err}); downloading the clip instead")
                if not ok:
                    if local is None:
                        local = Path(tmp) / PurePosixPath(path).name
                        with_retries(lambda: dbx.files_download_to_file(str(local), path))
                    ok, err, nudge, bright = extract_lit(ffmpeg, str(local), j["off"], j["clip"]["dur"], p,
                                                         args.max_width, dark)
                status = "failed: " + err if not ok else "ok_dark" if bright is not None and bright < dark else "ok"
                results[id(j)] = (p, status, f"{nudge:g}" if ok else "",
                                  f"{bright:.0f}" if bright is not None else "")
            if local is not None:
                local.unlink(missing_ok=True)

    # manifest
    with open(out / "frames_manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_path", "status", "row", "deployment", "observationTimestamp", "target_time_utc",
                    "offset_s", "clip", "time_source", "offset_in_clip_s", "dark_nudge_s", "brightness"]
                   + LINEAGE_COLS)
        for j in jobs:
            if j["status"] != "matched":
                continue
            p, st, nudge, bright = results.get(id(j), ("", "not_run", "", ""))
            src = j["src"]
            w.writerow([str(p), st, j["row"], j["dep"], src.get("observationTimestamp", ""),
                        j["t"].strftime("%Y-%m-%d %H:%M:%S"), j["offset_s"], j["clip"]["clip"],
                        j["clip"]["source"], f"{j['off']:.2f}", nudge, bright]
                       + [src.get(c, "") for c in LINEAGE_COLS])
    vals = list(results.values())
    ok_n = sum(1 for r in vals if r[1] in ("ok", "ok_dark", "exists"))
    nudged = sum(1 for r in vals if r[2] not in ("", "0"))
    still_dark = sum(1 for r in vals if r[1] == "ok_dark")
    print(f"\nDone: {ok_n}/{len(matched)} frames written ({nudged} nudged past a dark clip start, "
          f"{still_dark} still dark). Manifest: {out / 'frames_manifest.csv'}")
    for r in [r for r in vals if r[1].startswith("failed")][:5]:
        print(f"  failed: {r[0].name}: {r[1]}")
    return jobs


if __name__ == "__main__":
    main()