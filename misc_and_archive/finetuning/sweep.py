"""
Hyperparameter sweep over finetune_bioclip.py.

Runs one finetuning session per combination of hyperparameters,
names each log descriptively, and writes a ranked summary CSV at the end.

Resumable: if a training_logs/{log_name}.csv already exists for a combo,
that run is skipped and its best loss is read from the existing file.

Usage:
    python sweep.py
"""

import itertools
import csv
from pathlib import Path
from finetune_bioclip import train

# ============================================================
# SWEEP CONFIG
# ============================================================

# Directory of species-labelled crop folders to train on.
# Use crops_augmented/ with USE_TRANSFORMS=False for a controlled sweep.
CROPS_DIR      = Path(__file__).parent / "crops_augmented"
USE_TRANSFORMS = False

# Set to True to randomly sample instead of exhaustive grid search.
RANDOM_SEARCH    = False
RANDOM_N_SAMPLES = 10

# ============================================================
# SEARCH SPACE — edit these to define your sweep
# ============================================================

SEARCH_SPACE = {
    "lr"          : [2e-5, 5e-5],
    "temperature" : [0.05, 0.07],
    "batch_size"  : [32],
    "proj_dim"    : [128],
}

# ============================================================
# HELPERS
# ============================================================

LOG_DIR      = Path(__file__).parent / "narrow_train_logs"
SUMMARY_PATH = LOG_DIR / "sweep_summary.csv"


def make_log_name(combo: dict) -> str:
    """
    Build a human-readable log filename from the hyperparameter combo.
    e.g. lr2e-05_temp0.07_bs32_proj128
    """
    parts = [
        f"lr{combo['lr']:.0e}",
        f"temp{combo['temperature']}",
        f"bs{combo['batch_size']}",
        f"proj{combo['proj_dim']}",
    ]
    return "_".join(parts)


def generate_combos(search_space: dict, random_search: bool, n_samples: int) -> list[dict]:
    keys       = list(search_space.keys())
    values     = list(search_space.values())
    all_combos = [dict(zip(keys, v)) for v in itertools.product(*values)]

    if random_search:
        import random
        random.shuffle(all_combos)
        return all_combos[:n_samples]

    return all_combos


def read_best_loss_from_csv(log_path: Path) -> float:
    """
    Read the minimum loss value from an existing training log CSV.
    Returns inf if the file is missing or malformed.
    """
    try:
        with open(log_path, newline="") as f:
            reader = csv.DictReader(f)
            losses = [float(row["loss"]) for row in reader if row["loss"]]
        return min(losses) if losses else float("inf")
    except Exception:
        return float("inf")


# ============================================================
# SWEEP
# ============================================================

def run_sweep():
    LOG_DIR.mkdir(exist_ok=True)
    combos = generate_combos(SEARCH_SPACE, RANDOM_SEARCH, RANDOM_N_SAMPLES)
    total  = len(combos)

    # --- check which combos are already done ---
    pending  = []
    skipped  = []
    for combo in combos:
        log_path = LOG_DIR / f"{make_log_name(combo)}.csv"
        if log_path.exists():
            skipped.append(combo)
        else:
            pending.append(combo)

    print(f"\n{'='*60}")
    print(f"  Sweep: {total} total runs")
    print(f"  Already done: {len(skipped)}  |  Remaining: {len(pending)}")
    print(f"  Mode: {'random search' if RANDOM_SEARCH else 'grid search'}")
    print(f"  Crops dir: {CROPS_DIR}")
    print(f"  Transforms: {'enabled' if USE_TRANSFORMS else 'disabled'}")
    print(f"{'='*60}\n")

    if not pending:
        print("  All runs already complete — jumping straight to summary.\n")

    results = []  # list of (log_name, best_loss, combo)

    # --- load results from already-completed runs ---
    for combo in skipped:
        log_name  = make_log_name(combo)
        log_path  = LOG_DIR / f"{log_name}.csv"
        best_loss = read_best_loss_from_csv(log_path)
        results.append((log_name, best_loss, combo))
        print(f"  [skipped] {log_name}  (best loss from file: {best_loss:.4f})")

    if skipped:
        print()

    # --- run pending combos ---
    for i, combo in enumerate(pending, 1):
        log_name = make_log_name(combo)
        print(f"\n[Run {i}/{len(pending)}] {log_name}")
        print(f"  Hyperparameters: {combo}")
        print(f"{'-'*60}")

        try:
            best_loss = train(
                log_name       = log_name,
                crops_dir      = CROPS_DIR,
                use_transforms = USE_TRANSFORMS,
                **combo,
            )
            results.append((log_name, best_loss, combo))
            print(f"  ✓ Finished — best loss: {best_loss:.4f}")
        except Exception as e:
            print(f"  ✗ Run failed: {e}")
            results.append((log_name, float("inf"), combo))

    # --- write summary ---
    results.sort(key=lambda x: x[1])  # rank by best_loss ascending

    with open(SUMMARY_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["rank", "best_loss", "log_name"] + list(SEARCH_SPACE.keys())
        writer.writerow(header)
        for rank, (log_name, best_loss, combo) in enumerate(results, 1):
            row = [rank, f"{best_loss:.6f}", log_name] + [combo[k] for k in SEARCH_SPACE.keys()]
            writer.writerow(row)

    print(f"\n{'='*60}")
    print(f"  Sweep complete!")
    print(f"  Summary saved to {SUMMARY_PATH}")
    print(f"\n  Top 3 runs:")
    for rank, (log_name, best_loss, combo) in enumerate(results[:3], 1):
        print(f"    #{rank}  loss={best_loss:.4f}  {log_name}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_sweep()