"""Prepare ALFWorld data: run alfworld-download to populate data/alfworld/raw.

The pinned split manifests live in data/alfworld/splits/{train,val,test}/items.json
(checked into git). Each manifest entry has {id, gamefile, task_type} where gamefile
is relative to data/alfworld/raw/. ALFWorldAdapter reads these manifests to load
exactly the games specified.

Run this script once to download the raw game files (it is a no-op if already on disk).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ALF_ROOT = ROOT / "data" / "alfworld"
RAW = ALF_ROOT / "raw"


def run_download() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["ALFWORLD_DATA"] = str(RAW.resolve())
    json_dir = RAW / "json_2.1.1"
    if json_dir.exists() and any(json_dir.iterdir()):
        print(f"[ok] ALFWorld games already at {json_dir}")
        return
    print(f"[alfworld] downloading to {RAW}")
    cli = Path(sys.executable).with_name("alfworld-download")
    cmd = str(cli) if cli.exists() else "alfworld-download"
    subprocess.check_call([cmd, "--data-dir", str(RAW.resolve())], env=env)


def find_json_dir() -> Path:
    """Locate the json_2.1.1 dir, regardless of intermediate nesting."""
    candidates = list(RAW.rglob("json_2.1.1"))
    if not candidates:
        raise FileNotFoundError(f"json_2.1.1 not found under {RAW}")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    if not args.skip_download:
        run_download()
    json_dir = find_json_dir()
    print(f"[ok] json dir: {json_dir}")
    for split in ("train", "valid_seen", "valid_unseen"):
        n = sum(1 for _ in (json_dir / split).rglob("game.tw-pddl"))
        print(f"  {split}: {n} game.tw-pddl files")

    # Verify pinned manifests reference valid games.
    import json
    splits_dir = ALF_ROOT / "splits"
    for split_name in ("train", "val", "test"):
        manifest = splits_dir / split_name / "items.json"
        if not manifest.exists():
            print(f"[warn] manifest not found: {manifest}")
            continue
        entries = json.loads(manifest.read_text())
        found = sum(1 for e in entries if (RAW / e["gamefile"]).exists())
        print(f"  manifest {split_name}: {found}/{len(entries)} games found on disk")

    print("[done] ALFWorld data ready")


if __name__ == "__main__":
    main()
