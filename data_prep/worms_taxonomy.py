"""
Taxonomic lineages from WoRMS (World Register of Marine Species), cached on disk.

A lineage is a list of {"rank", "name"} from the top of the tree down to the taxon itself,
e.g. Biota > Animalia > Arthropoda > Crustacea > ... > Decapoda > Caridea > Pandalidae > Heterocarpus.

Used by crop_frames.py to write each row's taxon as a full lineage for BioCLIP, to decide which
labels agree with a row, and to tell MBARI-315k's animal classes from its non-animal ones.

Lookups go to https://www.marinespecies.org/rest and are cached in a JSON file, so each taxon is
fetched once; later runs need no network.
"""

import json
import re
import threading
import time
from pathlib import Path

WORMS = "https://www.marinespecies.org/rest"

# Rank order used to compare depths (WoRMS rank names). Unknown ranks fall back to their position.
RANKS = [
    "Superdomain", "Kingdom", "Subkingdom", "Infrakingdom", "Phylum", "Phylum (Division)", "Subphylum",
    "Subphylum (Subdivision)", "Infraphylum", "Parvphylum", "Gigaclass", "Megaclass", "Superclass",
    "Class", "Subclass", "Infraclass", "Subterclass", "Superorder", "Order", "Suborder", "Infraorder",
    "Parvorder", "Section", "Subsection", "Superfamily", "Epifamily", "Family", "Subfamily",
    "Supertribe", "Tribe", "Subtribe", "Genus", "Subgenus", "Species", "Subspecies", "Variety", "Forma",
]
RANK_INDEX = {r.lower(): i for i, r in enumerate(RANKS)}

# Sheet columns, in order, for building a lineage when a row has no usable aphiaId.
SHEET_RANK_COLS = [("phylum", "Phylum"), ("class", "Class"), ("subclass", "Subclass"), ("order", "Order"),
                   ("suborder", "Suborder"), ("family", "Family"), ("subfamily", "Subfamily"),
                   ("genus", "Genus"), ("subgenus", "Subgenus"), ("species", "Species")]


def rank_index(rank):
    return RANK_INDEX.get((rank or "").strip().lower(), -1)


def clean_label(name):
    """Turn a detector class name into a searchable taxon name, or None if it isn't an organism.

    'Opisthoteuthis cf. californiana' -> 'Opisthoteuthis', 'Cladorhizidae sp. A' -> 'Cladorhizidae',
    'Oneirophanta mutabilis complex' -> 'Oneirophanta mutabilis'. Eggs, egg cases, molts, detritus,
    equipment and other lower-case classes ('marine snow', 'sand') are not organisms -> None.
    """
    n = (name or "").strip()
    if not n or n[0].islower():
        return None
    low = n.lower()
    if any(w in low for w in ("egg", "molt", "detritus", "sampler", "equipment", "carcass")):
        return None
    n = re.sub(r"\s+(cf\.|aff\.)\s.*$", "", n)
    n = re.sub(r"\s+sp\.\s*\S*$", "", n)
    n = re.sub(r"\s+complex$", "", n, flags=re.I)
    n = re.sub(r"\s*\(.*?\)\s*", " ", n).strip()  # 'Latrunculia (Latrunculia) austini'
    if "-" in n.split()[0] or n.upper() == n:      # 'Neptunea-Buccinum', 'LRJ'
        return None
    return n


class Worms:
    def __init__(self, cache_path):
        self.path = Path(cache_path)
        self.cache = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.lock = threading.Lock()
        self._session = None
        self.calls = 0

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.cache, indent=0, sort_keys=True))

    def _get(self, url):
        import requests

        if self._session is None:
            self._session = requests.Session()
        for attempt in range(4):  # retry network hiccups: wait 2, 4, 8 s
            self.calls += 1
            try:
                r = self._session.get(url, timeout=30)
                if r.status_code == 204:  # WoRMS returns 204 No Content for "not found"
                    return None
                r.raise_for_status()
                return r.json()
            except Exception:  # noqa: BLE001
                if attempt == 3:
                    raise
                time.sleep(2 * 2 ** attempt)

    def _classification(self, aphia_id):
        node = self._get(f"{WORMS}/AphiaClassificationByAphiaID/{aphia_id}")
        out = []
        while node:
            out.append({"rank": node.get("rank"), "name": node.get("scientificname")})
            node = node.get("child")
        return out

    def lineage_by_id(self, aphia_id):
        key = f"id:{int(aphia_id)}"
        if key not in self.cache:
            rec = self._get(f"{WORMS}/AphiaRecordByAphiaID/{int(aphia_id)}")
            valid = (rec or {}).get("valid_AphiaID") or int(aphia_id)
            self.cache[key] = self._classification(valid) if rec else None
        return self.cache[key]

    def row_lineage(self, row):
        """Lineage for a spreadsheet row: WoRMS by aphiaId, else built from the sheet's rank columns."""
        try:
            aid = int(float(row.get("aphiaId") or 0))
        except ValueError:
            aid = 0
        if aid > 0:
            try:
                new = f"id:{aid}" not in self.cache
                lin = self.lineage_by_id(aid)
                if new:
                    self.save()
            except Exception as ex:  # noqa: BLE001  network down: use the sheet's columns, retry next run
                print(f"  ! WoRMS lookup failed for {aid} ({ex}); using the sheet's rank columns for this row")
                lin = None
            if lin:
                return lin
        lin = [{"rank": "Kingdom", "name": "Animalia"}]
        lin += [{"rank": r, "name": row[c].strip()} for c, r in SHEET_RANK_COLS if (row.get(c) or "").strip()]
        return lin

    def prefetch(self, aphia_ids=(), workers=4):
        """Look up every aphiaId not cached yet, a few at a time, then save the cache."""
        from concurrent.futures import ThreadPoolExecutor

        todo = [int(a) for a in set(aphia_ids) if f"id:{int(a)}" not in self.cache]
        if not todo:
            return 0

        def one(aid):
            try:
                return self.lineage_by_id(aid)
            except Exception as ex:  # noqa: BLE001  (network hiccup: leave it uncached, retry next run)
                print(f"  ! WoRMS lookup failed for {aid}: {ex}")

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(one, todo))
        self.save()
        return len(todo)

