"""
Rename mission 11 crop files to match the naming convention of missions 03 and 33.

Changes applied to each file:
  - Track ID zero-padded to 4 digits:   id18  → id0018
  - Frame number zero-padded to 5 digits: f662 → f00662
  - Label 'best' renamed to 'mid':      _best → _mid

Run as dry-run (default) to preview changes, then pass --apply to rename:
    python rename_mission11_crops.py
    python rename_mission11_crops.py --apply
"""

import re
import sys
from pathlib import Path

CROPS_DIR = Path(__file__).parent / "processed_missions" / "mission_11" / "mission_11_crops"
PATTERN = re.compile(r"^id(\d+)_(.+?)_f(\d+)_(early|mid|late|best)\.jpg$")


def main(dry_run: bool) -> None:
    if not CROPS_DIR.exists():
        print(f"ERROR: crops directory not found: {CROPS_DIR}")
        sys.exit(1)

    to_rename = []
    unmatched = []

    for jpg in sorted(CROPS_DIR.rglob("*.jpg")):
        m = PATTERN.match(jpg.name)
        if not m:
            unmatched.append(jpg)
            continue

        tid, cls, frame, label = m.groups()
        new_label = "mid" if label == "best" else label
        new_name = f"id{int(tid):04d}_{cls}_f{int(frame):05d}_{new_label}.jpg"

        if new_name != jpg.name:
            to_rename.append((jpg, jpg.parent / new_name))

    if unmatched:
        print("Files that did not match expected pattern (skipped):")
        for p in unmatched:
            print(f"  {p.relative_to(CROPS_DIR)}")
        print()

    if not to_rename:
        print("Nothing to rename — all files already match the target format.")
        return

    print(f"{'DRY RUN — ' if dry_run else ''}Renaming {len(to_rename)} files:\n")
    for old, new in to_rename:
        print(f"  {old.name}  ->  {new.name}")
        if not dry_run:
            old.rename(new)

    if dry_run:
        print(f"\nDry run complete. Pass --apply to perform the {len(to_rename)} renames.")
    else:
        print(f"\nDone. Renamed {len(to_rename)} files.")


if __name__ == "__main__":
    main(dry_run="--apply" not in sys.argv)
