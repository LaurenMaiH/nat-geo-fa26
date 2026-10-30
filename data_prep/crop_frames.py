"""
Crop each extracted frame to the animal its spreadsheet row is about.

Input:  the frames_manifest.csv written by frame_extraction.py (one full 4K frame per row).
Output: <out>/crops/<deployment>/<frame name>.jpg and <out>/crops_manifest.csv.

The aim is only to tie each crop to its spreadsheet row, so precision is favoured over coverage:
a row gets a crop only when the pipeline is confident, otherwise it gets none (with a reason).

1. Three detectors propose boxes, which are pooled (near-duplicates merged):
     JAMSTEC        YOLO-World fine-tuned on deep-sea footage
     MBARI-315k     FathomNet's YOLOv8 (its non-animal classes - marine snow, equipment... - dropped)
     prompted       general YOLO-World told what the row's animal is ("chimaera", "eel"...), which
                    finds large animals the other two miss (skip with --no-prompted; it is the slowest)
   Tiny boxes, boxes over half the frame and the bait arm are filtered out.

2. BioCLIP 2 scores every box against the row's taxon, written as its WoRMS lineage, among all taxa
   in the sheet (--sheet) plus a few non-animal prompts. A box's score is the probability on labels
   that agree with the row: its taxon, a taxon inside it, or a Family-or-deeper ancestor of it.

3. The best box is cropped if
     bioclip_match    its score is >= --min-p (0.2), or
     unique_in_class  BioCLIP is unsure of the exact (often rare) taxon, but one box is clearly of
                      the row's class (>= --class-min, e.g. "a shark") and every other box is not
                      (<= --class-max-other) - enough to tie the crop to the row.
   Rows identified coarser than Family (--min-row-rank; "Caridea", "Decapoda", "Amphipoda") never get
   a crop: those labels cover very different-looking animals and caused the confident mistakes in
   testing.

Detections are cached per frame (detections_cache.json), so changing the selection settings or crop
size and re-running is quick; only changing detection settings re-runs the detectors.

Examples
  python crop_frames.py --manifest extracted_frames/frames_manifest.csv --sheet SLB_DOEX0102_Data_Summary.csv
  python crop_frames.py --manifest ... --sheet ... --limit 20 --debug        # frames with boxes drawn
  python crop_frames.py --manifest ... --sheet ... --audit-sample 50         # crops for a person to check

Requires: pip install ultralytics opencv-python huggingface_hub requests gdown open_clip_torch
"""

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from worms_taxonomy import Worms, clean_label, rank_index

FRAME_OK = {"ok", "ok_dark", "exists"}
REF_WIDTH = 3840  # --min-side, --min-crop and the scoring crop size were tuned on 4K (3840 px wide) frames


def px(value, w):
    """A pixel setting tuned at 4K, scaled to a frame w pixels wide (e.g. 20 px at 4K -> 10 px at 1920)."""
    return value * w / REF_WIDTH

# ── detectors ─────────────────────────────────────────────────────────────────
YOLO_CACHE_DIR = Path.home() / ".config" / "ultralytics"
MODELS = {
    "jamstec": {"file": "yolov8x-jamstec.pt", "gdrive_id": "1rCyq4GZcG5UCqrl2SNNjmaHkLOZbWgyv"},
    "worldv2": {"file": "yolov8x-worldv2.pt",  # same weights as overnight_pipeline_optimized.py
                "gdrive_id": "1hh576zOzpUqgWSjIdSkR4EwgYQ8kH406"},
    "mbari":   {"hf": ("FathomNet/MBARI-315k-yolov8", "mbari_315k_yolov8.pt")},  # CC-BY-4.0
}
DETECTORS = ["jamstec", "mbari"]  # always run; "prompted" (worldv2 with per-row prompts) is added on top

# MBARI classes that are animals even though they are not a single WoRMS taxon - kept, not dropped as non-animal
MBARI_VAGUE_ANIMALS = {"marine organism", "Animalia", "Worm", "Medusae", "Hydromedusae", "Teuthoidea",
                       "LRJ complex", "Funiculina-Balticina complex", "Neptunea-Buccinum Complex"}

# Plain-English prompts for the row's animal. The row's WoRMS lineage is walked from its deepest taxon
# up; the first name found here gives the prompts.
TAXON_PROMPTS = {
    # fish
    "Chimaeriformes": ["chimaera", "ghost shark"], "Holocephali": ["chimaera", "ghost shark"],
    "Myliobatiformes": ["stingray", "ray"], "Rajiformes": ["skate", "ray"], "Torpediniformes": ["ray"],
    "Batoidea": ["ray"], "Elasmobranchii": ["shark"],
    "Anguilliformes": ["eel"], "Ophidiiformes": ["eel-like fish", "fish"], "Macrouridae": ["rattail fish", "fish"],
    "Myxini": ["hagfish", "eel"], "Teleostei": ["fish"], "Actinopteri": ["fish"], "Chordata": ["fish"],
    # crustaceans
    "Lithodidae": ["king crab", "crab"], "Paguroidea": ["hermit crab"], "Munididae": ["squat lobster"],
    "Galatheoidea": ["squat lobster"], "Chirostyloidea": ["squat lobster"], "Polychelida": ["lobster"],
    "Brachyura": ["crab"], "Caridea": ["shrimp"], "Dendrobranchiata": ["shrimp"], "Penaeoidea": ["shrimp"],
    "Mysida": ["shrimp"], "Isopoda": ["isopod"], "Amphipoda": ["amphipod"], "Copepoda": ["copepod"],
    "Decapoda": ["shrimp", "crab"], "Arthropoda": ["crustacean"],
    # echinoderms
    "Euryalida": ["basket star"], "Ophiuroidea": ["brittle star"], "Echinoidea": ["sea urchin"],
    "Asteroidea": ["sea star", "starfish"], "Holothuroidea": ["sea cucumber"], "Crinoidea": ["crinoid", "sea lily"],
    # molluscs
    "Nautilida": ["nautilus"], "Octopoda": ["octopus"], "Decapodiformes": ["squid"], "Gastropoda": ["snail"],
    # cnidarians, ctenophores, sponges, worms
    "Siphonophorae": ["siphonophore", "jellyfish"], "Hydrozoa": ["jellyfish"], "Scyphozoa": ["jellyfish"],
    "Ctenophora": ["comb jelly"], "Pennatuloidea": ["sea pen"], "Actiniaria": ["sea anemone"],
    "Octocorallia": ["coral", "sea fan"], "Antipatharia": ["black coral"], "Porifera": ["sponge"],
    "Polychaeta": ["worm"], "Annelida": ["worm"], "Chaetognatha": ["arrow worm"], "Nemertea": ["worm"],
}


def row_prompts(lineage):
    for t in reversed(lineage or []):
        if t["name"] in TAXON_PROMPTS:
            return TAXON_PROMPTS[t["name"]]
    return ["animal"]


def pick_device(requested):
    if requested:
        return requested
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(name):
    import gdown
    from ultralytics import YOLO, YOLOWorld, settings

    spec = MODELS[name]
    if "hf" in spec:  # Hugging Face download, cached in ~/.cache/huggingface
        from huggingface_hub import hf_hub_download
        model = YOLO(hf_hub_download(*spec["hf"]))
    else:
        YOLO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        settings.update({"weights_dir": str(YOLO_CACHE_DIR), "runs_dir": str(YOLO_CACHE_DIR / "runs")})
        path = YOLO_CACHE_DIR / spec["file"]
        if path.exists() and path.stat().st_size < 1_000_000:  # an HTML error page saved by an old download
            print(f"{path} is too small to be model weights - deleting and re-downloading")
            path.unlink()
        if not path.exists():
            print(f"Downloading {name} model to {path}...")
            # download by file id: works on gdown 4.x-6.x (gdown 6 dropped fuzzy=, and a /view URL
            # without it saves Google Drive's HTML page instead of the weights)
            tmp = path.with_suffix(".part")
            gdown.download(id=spec["gdrive_id"], output=str(tmp), quiet=False)
            if not tmp.exists() or tmp.stat().st_size < 1_000_000:
                tmp.unlink(missing_ok=True)
                sys.exit(f"Downloading the {name} weights from Google Drive failed (file missing or not a model). "
                         f"Check your connection; if Drive says 'too many users', try again later.")
            tmp.replace(path)
        model = YOLOWorld(str(path))
    model.model.eval()
    return model


def detect(model, img, args, device, conf):
    r = model.predict(img, imgsz=args.imgsz, conf=conf, iou=args.iou, agnostic_nms=True,
                      device=device, verbose=False)[0]
    return [{"label": model.names[c], "conf": round(float(s), 4),
             "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]}
            for (x1, y1, x2, y2), c, s in zip(r.boxes.xyxy.cpu().numpy().tolist(),
                                              r.boxes.cls.int().cpu().tolist(), r.boxes.conf.cpu().tolist())]


def cache_key(name, args, conf, prompts=None):
    # the False is a retired contrast-boost flag, kept so existing detection caches stay valid
    return hashlib.md5(json.dumps([name, args.imgsz, conf, args.iou, False, prompts]).encode()).hexdigest()[:10]


def run_detector(model, jobs, args, device, cache, cache_path, desc, prompts_of=None):
    """Detect on every (frame path, cache key) job; prompts_of(job) switches YOLO-World's prompts."""
    try:
        from tqdm import tqdm
        jobs = tqdm(jobs, desc=desc, unit="frame")
    except ImportError:
        pass
    current = None
    for n, job in enumerate(jobs):
        fp, key = job[0], job[1]
        if prompts_of and prompts_of(job) != current:
            current = prompts_of(job)
            model.set_classes(current)
        img = cv2.imread(str(fp))
        conf = args.prompt_conf if prompts_of else args.conf
        try:
            dets = detect(model, img, args, device, conf)
        except Exception as ex:  # noqa: BLE001  (e.g. an op not supported on mps)
            if device == "cpu":
                raise
            print(f"\n  detection failed on {device} ({ex}); switching to cpu")
            device = "cpu"
            dets = detect(model, img, args, device, conf)
        cache[f"{fp.name}|{key}"] = {"w": img.shape[1], "h": img.shape[0], "dets": dets}
        if n % 20 == 0:
            cache_path.write_text(json.dumps(cache))
    cache_path.write_text(json.dumps(cache))
    return device


# ── box filtering & pooling ──────────────────────────────────────────────────
def filter_boxes(dets, w, h, args, src):
    """Split one detector's boxes into kept and dropped (with the reason)."""
    kept, dropped = [], []
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        bw, bh = x2 - x1, y2 - y1
        reason = None
        if min(bw, bh) < px(args.min_side, w):
            reason = "too_small"
        elif (not args.keep_arm and y2 > 0.95 * h and bh > 0.2 * h and bh > 1.5 * bw
              and 0.3 * w < (x1 + x2) / 2 < 0.65 * w):
            reason = "bait_arm"  # tall box from the bottom edge near the centre = the arm + canister
        elif bw * bh > args.max_area * w * h:
            reason = "too_large"
        elif src == "mbari" and d["label"] not in MBARI_VAGUE_ANIMALS and clean_label(d["label"]) is None:
            reason = "not_animal"  # marine snow, equipment, sand, eggs, detritus...
        d = {**d, "src": src}
        (dropped if reason else kept).append({**d, "dropped": reason} if reason else d)
    return kept, dropped


def iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - iw * ih
    return iw * ih / u if u else 0.0


def pool_boxes(entries, args):
    """Filter each detector's boxes, then merge them; a box overlapping an earlier one (IoU >= 0.5) is a duplicate."""
    kept, dropped = [], []
    for src, entry in entries:
        k, d = filter_boxes(entry["dets"], entry["w"], entry["h"], args, src)
        for b in k:
            if any(iou(b["box"], o["box"]) >= 0.5 for o in kept):
                dropped.append({**b, "dropped": "duplicate"})
            else:
                kept.append(b)
        dropped += d
    return kept, dropped


def square_crop(box, w, h, scale, min_side):
    x1, y1, x2, y2 = box
    side = min(max(max(x2 - x1, y2 - y1) * scale, min_side), w, h)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    X1 = int(round(min(max(0, cx - side / 2), w - side)))
    Y1 = int(round(min(max(0, cy - side / 2), h - side)))
    return X1, Y1, X1 + int(side), Y1 + int(side)


# ── BioCLIP box scoring & selection ──────────────────────────────────────────
DEEP_RANKS = {"Family", "Subfamily", "Tribe", "Genus", "Subgenus", "Species", "Subspecies"}
BIOCLIP_RANKS = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
NEGATIVE_PROMPTS = ["a photo of rock", "a photo of sandy sediment", "a photo of marine snow particles",
                    "a photo of a metal camera frame", "a photo of dark empty water"]


def taxon_text(lin):
    """BioCLIP's taxonomic prompt, e.g. 'a photo of Animalia Arthropoda Malacostraca ... Heterocarpus'."""
    parts = [t["name"] for t in lin if t["rank"] in BIOCLIP_RANKS]
    if lin and lin[-1]["rank"] == "Species":  # WoRMS gives "Genus species"; BioCLIP wants the epithet
        parts[-1] = lin[-1]["name"].split()[-1]
    return "a photo of " + " ".join(parts)


class BioclipScorer:
    """Scores crops against a fixed label set: every taxon lineage given + NEGATIVE_PROMPTS."""

    def __init__(self, lineages, device):
        import open_clip
        import torch
        self.torch = torch
        self.lin = {taxon_text(l): l for l in lineages if l}
        self.labels = sorted(self.lin) + NEGATIVE_PROMPTS
        self.device = device
        self.model, _, self.prep = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip-2")
        self.model = self.model.to(device).eval()
        tok = open_clip.get_tokenizer("hf-hub:imageomics/bioclip-2")
        with torch.no_grad():
            t = self.model.encode_text(tok(self.labels).to(device))
        self.text = t / t.norm(dim=-1, keepdim=True)
        self._masks = {}

    def _mask(self, key, keep):
        if key not in self._masks:
            self._masks[key] = np.array([bool(self.lin.get(lab)) and keep(self.lin[lab]) for lab in self.labels])
        return self._masks[key]

    def row_mask(self, row_lin):
        """Labels that agree with the row: its taxon, inside it, or a Family-or-deeper ancestor of it."""
        row_names = {t["name"] for t in row_lin}
        return self._mask(("row", taxon_text(row_lin)), lambda lin: (
            row_lin[-1]["name"] in {t["name"] for t in lin}
            or (lin[-1]["name"] in row_names and lin[-1]["rank"] in DEEP_RANKS)))

    def class_mask(self, row_lin):
        """Labels in the same class as the row (e.g. every shark, ray and chimaera for a shark row)."""
        cls = next((t["name"] for t in row_lin if t["rank"] == "Class"), None)
        return self._mask(("class", cls), lambda lin: bool(cls) and cls in {t["name"] for t in lin})

    def score(self, img, boxes, row_lin):
        """Per box: (probability it is the row's taxon, probability it is of the row's class, BioCLIP's top label)."""
        from PIL import Image
        h, w = img.shape[:2]
        crops = []
        for b in boxes:
            x1, y1, x2, y2 = square_crop(b, w, h, 1.2, px(96, w))
            crops.append(self.prep(Image.fromarray(cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))))
        with self.torch.no_grad():
            f = self.model.encode_image(self.torch.stack(crops).to(self.device))
            f = f / f.norm(dim=-1, keepdim=True)
            P = (100.0 * f @ self.text.T).softmax(dim=-1).cpu().numpy()
        p_row = P[:, self.row_mask(row_lin)].sum(axis=1)
        p_cls = P[:, self.class_mask(row_lin)].sum(axis=1)
        return [(float(p), float(c), self.labels[int(q.argmax())].replace("a photo of ", ""))
                for p, c, q in zip(p_row, p_cls, P)]


def select_box(kept, row_lin, img, args, scorer):
    """Return (index into kept or None, selection or reason); annotates every kept box with its scores."""
    if not kept:
        return None, "no_detection"
    if not row_lin:
        return None, "unknown_taxon"
    for d, (p, c, top) in zip(kept, scorer.score(img, [d["box"] for d in kept], row_lin)):
        d.update({"p_row": round(p, 4), "p_class": round(c, 4), "bioclip_top": top})
    if rank_index(row_lin[-1]["rank"]) < rank_index(args.min_row_rank):
        return None, "coarse_taxon"
    best = max(range(len(kept)), key=lambda i: kept[i]["p_row"])
    if kept[best]["p_row"] >= args.min_p:
        return best, "bioclip_match"
    by_cls = sorted(range(len(kept)), key=lambda i: -kept[i]["p_class"])
    if (kept[by_cls[0]]["p_class"] >= args.class_min
            and all(kept[i]["p_class"] <= args.class_max_other for i in by_cls[1:])):
        return by_cls[0], "unique_in_class"
    return None, "low_score"


# ── drawing ───────────────────────────────────────────────────────────────────
def draw_boxes(img, kept, dropped, chosen, title, width=1920):
    s = width / img.shape[1]
    out = cv2.resize(img, (width, int(img.shape[0] * s)))
    for d in dropped:
        x1, y1, x2, y2 = (int(v * s) for v in d["box"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (128, 128, 128), 1)
    for i, d in enumerate(kept):
        x1, y1, x2, y2 = (int(v * s) for v in d["box"])
        col, th = ((0, 255, 0), 3) if i == chosen else ((0, 220, 255), 2)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, th)
        score = f" p={d['p_row']:.2f} c={d['p_class']:.2f} {d['bioclip_top'].split()[-1]}" if "p_row" in d else ""
        cv2.putText(out, f"{i + 1} {d['src'][0].upper()} {d['label']} {d['conf']:.2f}{score}",
                    (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    cv2.rectangle(out, (0, 0), (width, 34), (0, 0, 0), -1)
    cv2.putText(out, title[:150], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


# ── main ──────────────────────────────────────────────────────────────────────
KNOWN_COLS = ["scientificName", "taxonRank", "aphiaId", "phylum", "class", "subclass", "order", "suborder",
              "family", "subfamily", "genus", "subgenus", "species", "subspecies", "morphospecies",
              "tentativeId", "deployment", "observationTimestamp", "identificationQualifier"]


def normalise_headers(rows):
    """Rename columns to the spelling used here, ignoring case ('AphiaID' -> 'aphiaId'), so older sheets work."""
    canon = {c.lower(): c for c in KNOWN_COLS}
    return [{canon.get(k.strip().lower(), k) if k else k: v for k, v in r.items()} for r in rows]


def resolve_frame(row, manifest_dir):
    p = Path(row.get("frame_path") or "")
    if p.exists():
        return p
    alt = manifest_dir / (row.get("deployment") or "") / p.name
    return alt if alt.exists() else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="frames_manifest.csv from frame_extraction.py")
    ap.add_argument("--sheet", required=True,
                    help="full data-summary sheet (.csv); all its taxa compete in BioCLIP's label set")
    ap.add_argument("--out", help="output folder (default: the manifest's folder)")
    ap.add_argument("--limit", type=int, help="only the first N rows")
    # detection
    ap.add_argument("--imgsz", type=int, default=1280, help="YOLO input size (default 1280)")
    ap.add_argument("--conf", type=float, default=0.15, help="JAMSTEC / MBARI confidence threshold (default 0.15)")
    ap.add_argument("--prompt-conf", type=float, default=0.05,
                    help="prompted YOLO-World confidence threshold (default 0.05)")
    ap.add_argument("--no-prompted", action="store_true", help="skip the prompted YOLO-World detector (faster)")
    ap.add_argument("--iou", type=float, default=0.7, help="duplicate-box overlap threshold within a detector")
    ap.add_argument("--min-side", type=float, default=20, help="drop boxes with a side shorter than this (px at 4K; scaled to the frame's width)")
    ap.add_argument("--max-area", type=float, default=0.5, help="drop boxes covering more than this fraction")
    ap.add_argument("--keep-arm", action="store_true", help="don't drop tall boxes on the bait arm")
    # selection
    ap.add_argument("--min-p", type=float, default=0.2,
                    help="bioclip_match: score the best box needs (default 0.2)")
    ap.add_argument("--class-min", type=float, default=0.75,
                    help="unique_in_class: probability the box is of the row's class (default 0.75)")
    ap.add_argument("--class-max-other", type=float, default=0.1,
                    help="unique_in_class: every other box must be below this (default 0.1)")
    ap.add_argument("--min-row-rank", default="Family",
                    help="rows identified coarser than this get no crop (default Family)")
    ap.add_argument("--worms-cache", default=str(Path(__file__).resolve().parent / "worms_cache.json"),
                    help="where WoRMS lineages are cached (default: next to this script)")
    # output
    ap.add_argument("--pad", type=float, default=1.3, help="crop side = pad x the box's longer side")
    ap.add_argument("--min-crop", type=int, default=224, help="smallest crop side (px at 4K; scaled to the frame's width)")
    ap.add_argument("--device", help="cuda / mps / cpu (default: best available)")
    ap.add_argument("--debug", action="store_true", help="save each frame with its boxes drawn to <out>/debug/")
    ap.add_argument("--audit-sample", type=int, default=0, help="write N random crops for a person to check")
    ap.add_argument("--overwrite", action="store_true", help="re-write crops that already exist")
    args = ap.parse_args(argv)

    manifest = Path(args.manifest)
    mdir = manifest.parent
    out = Path(args.out) if args.out else mdir
    (out / "crops").mkdir(parents=True, exist_ok=True)
    with open(manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[:args.limit]
    src_cols = list(rows[0].keys()) if rows else []
    with open(args.sheet, newline="", encoding="utf-8-sig") as f:
        sheet = normalise_headers(list(csv.DictReader(f)))

    # taxonomy (cached; only new aphiaIds hit the network)
    worms = Worms(args.worms_cache)
    ids = {int(float(r["aphiaId"])) for r in rows + sheet if (r.get("aphiaId") or "").strip().replace(".0", "").isdigit()}
    if worms.prefetch(aphia_ids=ids):
        print(f"Looked up new taxa in WoRMS (cached in {args.worms_cache})")
    lineages = [worms.row_lineage(r) for r in rows]

    # detection, cached per frame file + detector + settings
    keys = {m: cache_key(m, args, args.conf) for m in DETECTORS}
    prompts = {} if args.no_prompted else {i: row_prompts(lin) for i, lin in enumerate(lineages)}
    pkeys = {i: cache_key("worldv2-prompted", args, args.prompt_conf, pr) for i, pr in prompts.items()}
    cache_path = out / "detections_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    wanted = set(keys.values()) | set(pkeys.values())
    cache = {k: v for k, v in cache.items() if k.split("|")[-1] in wanted}

    frames = {i: resolve_frame(r, mdir) for i, r in enumerate(rows) if r.get("status") in FRAME_OK}
    device = None
    for m, key in keys.items():
        jobs = [(fp, key) for fp in dict.fromkeys(fp for fp in frames.values() if fp) if f"{fp.name}|{key}" not in cache]
        if jobs:
            device = device or pick_device(args.device)
            print(f"Loading detector '{m}' on {device}...")
            device = run_detector(load_model(m), jobs, args, device, cache, cache_path, f"detecting ({m})")
    jobs = sorted({(fp, pkeys[i], tuple(prompts[i])) for i, fp in frames.items()
                   if fp and i in pkeys and f"{fp.name}|{pkeys[i]}" not in cache}, key=lambda j: j[2])
    if jobs:  # sorted by prompts, so YOLO-World's text prompts change as rarely as possible
        device = device or pick_device(args.device)
        print(f"Loading YOLO-World (prompted per row) on {device}...")
        device = run_detector(load_model("worldv2"), jobs, args, device, cache, cache_path,
                              "detecting (prompted)", prompts_of=lambda j: list(j[2]))

    device = device or pick_device(args.device)
    scorer = BioclipScorer([worms.row_lineage(r) for r in rows + sheet], device)
    print(f"BioCLIP 2 on {device}: {len(scorer.labels) - len(NEGATIVE_PROMPTS)} taxa + "
          f"{len(NEGATIVE_PROMPTS)} non-animal labels")

    # selection
    results = []
    picks_by_frame = defaultdict(list)  # (deployment, time, offset) -> [(result index, box)]
    for i, r in enumerate(rows):
        res = {"crop_path": "", "crop_status": "", "selection": "", "reason": "", "det_source": "",
               "det_label": "", "det_conf": "", "bioclip_p": "", "bioclip_class_p": "", "bioclip_top": "",
               "box_x1": "", "box_y1": "", "box_x2": "", "box_y2": "", "n_boxes": "", "all_boxes": ""}
        results.append(res)
        if i not in frames:
            res["crop_status"] = "no_frame"
            continue
        fp = frames[i]
        name = Path(r["frame_path"]).name
        entries = [(m, cache.get(f"{name}|{k}")) for m, k in keys.items()]
        if i in pkeys:
            entries.append(("prompted", cache.get(f"{name}|{pkeys[i]}")))
        if fp is None or any(e is None for _, e in entries):
            res["crop_status"] = "frame_missing"
            continue
        kept, dropped = pool_boxes(entries, args)
        idx, why = select_box(kept, lineages[i], cv2.imread(str(fp)) if kept else None, args, scorer)
        res.update({"n_boxes": len(kept), "all_boxes": json.dumps(kept + dropped)})
        res["_kept"], res["_dropped"], res["_idx"], res["_fp"] = kept, dropped, idx, fp
        res["_wh"] = (entries[0][1]["w"], entries[0][1]["h"])
        if kept and "p_row" in kept[0]:
            top = max(kept, key=lambda d: d["p_row"])
            res.update({"bioclip_p": top["p_row"], "bioclip_top": top["bioclip_top"]})
        if idx is None:
            res.update({"crop_status": "no_detection" if why == "no_detection" else "no_crop", "reason": why})
            continue
        d = kept[idx]
        res.update({"crop_status": "cropped", "selection": why, "det_source": d["src"], "det_label": d["label"],
                    "det_conf": d["conf"], "bioclip_p": d["p_row"], "bioclip_class_p": d["p_class"],
                    "bioclip_top": d["bioclip_top"], "box_x1": round(d["box"][0]), "box_y1": round(d["box"][1]),
                    "box_x2": round(d["box"][2]), "box_y2": round(d["box"][3])})
        picks_by_frame[(r.get("deployment"), r.get("target_time_utc"), r.get("offset_s"))].append((i, d["box"]))

    # rows sharing a frame must not claim the same box
    for picks in picks_by_frame.values():
        boxes = Counter(tuple(b) for _, b in picks)
        for ri, b in picks:
            if boxes[tuple(b)] > 1:
                results[ri].update({"crop_status": "no_crop", "reason": "shared_frame_conflict", "selection": ""})

    # crops + debug images
    for r, res in zip(rows, results):
        if "_fp" not in res:
            continue
        fp, (w, h) = res["_fp"], res["_wh"]
        crop_rel = Path("crops") / (r.get("deployment") or "") / fp.name
        img = None
        if res["crop_status"] == "cropped":
            res["crop_path"] = str(crop_rel)
            if args.overwrite or not (out / crop_rel).exists():
                img = cv2.imread(str(fp))
                X1, Y1, X2, Y2 = square_crop(res["_kept"][res["_idx"]]["box"], w, h, args.pad,
                                             px(args.min_crop, w))
                (out / crop_rel).parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out / crop_rel), img[Y1:Y2, X1:X2])
        if args.debug:
            img = img if img is not None else cv2.imread(str(fp))
            title = (f"row {r.get('row')} {r.get('scientificName')} ({r.get('taxonRank')}) "
                     f"-> {res['crop_status']} {res['selection'] or res['reason']}")
            dbg = draw_boxes(img, res["_kept"], res["_dropped"],
                             res["_idx"] if res["crop_status"] == "cropped" else None, title)
            dp = out / "debug" / (r.get("deployment") or "") / fp.name
            dp.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dp), dbg, [cv2.IMWRITE_JPEG_QUALITY, 80])

    # manifest
    new_cols = ["crop_path", "crop_status", "selection", "reason", "det_source", "det_label", "det_conf",
                "bioclip_p", "bioclip_class_p", "bioclip_top", "box_x1", "box_y1", "box_x2", "box_y2",
                "n_boxes", "all_boxes"]
    with open(out / "crops_manifest.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(src_cols + new_cols)
        for r, res in zip(rows, results):
            wr.writerow([r.get(c, "") for c in src_cols] + [res.get(c, "") for c in new_cols])

    if args.audit_sample:
        write_audit(rows, results, out, args.audit_sample)

    st = Counter(res["crop_status"] for res in results)
    sel = Counter(res["selection"] for res in results if res["selection"])
    why = Counter(res["reason"] for res in results if res["crop_status"] == "no_crop")
    print(f"\n{len(rows)} rows: " + ", ".join(f"{k} {v}" for k, v in st.most_common()))
    if sel:
        print("  cropped by: " + ", ".join(f"{k} {v}" for k, v in sel.most_common()))
    if why:
        print("  not cropped because: " + ", ".join(f"{k} {v}" for k, v in why.most_common()))
    print(f"Manifest: {out / 'crops_manifest.csv'}")


def write_audit(rows, results, out, n):
    """Side-by-side images (crop | frame with box) for a person to mark correct / wrong."""
    adir = out / "audit"
    if adir.exists():
        shutil.rmtree(adir)
    adir.mkdir(parents=True)
    cropped = [(r, res) for r, res in zip(rows, results) if res["crop_status"] == "cropped"]
    picked = random.Random(0).sample(cropped, min(n, len(cropped)))
    with open(adir / "audit.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["audit_image", "row", "deployment", "scientificName", "selection", "det_source",
                     "bioclip_p", "crop_path", "correct", "notes"])
        for k, (r, res) in enumerate(picked, 1):
            crop = cv2.imread(str(out / res["crop_path"]))
            frame = cv2.imread(str(res["_fp"]))
            d = res["_kept"][res["_idx"]]
            s = 1080 / frame.shape[0]
            fr = cv2.resize(frame, (int(frame.shape[1] * s), 1080))
            x1, y1, x2, y2 = (int(v * s) for v in d["box"])
            cv2.rectangle(fr, (x1, y1), (x2, y2), (0, 255, 0), 3)
            panel = np.hstack([cv2.resize(crop, (1080, 1080)), fr])
            cv2.rectangle(panel, (0, 0), (panel.shape[1], 40), (0, 0, 0), -1)
            cv2.putText(panel, f"{r.get('scientificName')}  ({r.get('class') or r.get('phylum')})  "
                        f"{res['selection']}  p={res['bioclip_p']}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            name = f"{k:03d}_row{r.get('row')}.jpg"
            cv2.imwrite(str(adir / name), panel, [cv2.IMWRITE_JPEG_QUALITY, 85])
            wr.writerow([name, r.get("row"), r.get("deployment"), r.get("scientificName"), res["selection"],
                         res["det_source"], res["bioclip_p"], res["crop_path"], "", ""])
    print(f"Audit: {len(picked)} images in {adir}/ - fill in the 'correct' column of audit.csv")


if __name__ == "__main__":
    sys.exit(main())
