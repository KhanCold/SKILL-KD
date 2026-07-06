"""Prepare SpreadsheetBench data: download tarball, extract, build splits from pinned manifest.

Layout produced:
  data/spreadsheetbench/raw/
    dataset.json           (912 items — full SpreadsheetBench dataset)
    spreadsheet/<id>/{N}_{id}_input.xlsx, {N}_{id}_answer.xlsx
  data/spreadsheetbench/train.jsonl    (80 rows — from pinned manifest)
  data/spreadsheetbench/val.jsonl      (40 rows — from pinned manifest)
  data/spreadsheetbench/test.jsonl     (280 rows — from pinned manifest)

The pinned split manifests live in data/spreadsheetbench/splits/{train,val,test}/items.json
(checked into git). Each entry has {id, spreadsheet_path, instruction_type}. This script
filters the 912-row raw dataset down to the 400 items in the manifest and writes the
split JSONLs.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "spreadsheetbench" / "raw"
SPLITS_DIR = ROOT / "data" / "spreadsheetbench" / "splits"
OUT_DIR = ROOT / "data" / "spreadsheetbench"
EXTERNAL = ROOT / "external" / "SpreadsheetBench"

TARBALL_CANDIDATES = [
    "spreadsheetbench_912_v0.1.tar.gz",
    "all_data_912.tar.gz",
]
TARBALL_URL_PREFIX = "https://github.com/RUCKBReasoning/SpreadsheetBench/raw/main/data/"

SPLITS = {
    "train": "train.jsonl",
    "val": "val.jsonl",
    "test": "test.jsonl",
}


def clone_repo() -> None:
    if EXTERNAL.exists():
        print(f"[ok] external repo already at {EXTERNAL}")
        return
    EXTERNAL.parent.mkdir(parents=True, exist_ok=True)
    print(f"[git] cloning SpreadsheetBench -> {EXTERNAL}")
    subprocess.check_call(
        ["git", "clone", "--depth", "1",
         "https://github.com/RUCKBReasoning/SpreadsheetBench.git", str(EXTERNAL)]
    )


def fetch_tarball() -> Path:
    """Prefer the tarball already shipped inside the cloned repo; else download."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    def is_real_gzip(path: Path) -> bool:
        with path.open("rb") as f:
            return f.read(2) == b"\x1f\x8b"

    for name in TARBALL_CANDIDATES:
        in_repo = EXTERNAL / "data" / name
        if in_repo.exists() and is_real_gzip(in_repo):
            print(f"[ok] using tarball from cloned repo: {in_repo}")
            return in_repo

    for name in TARBALL_CANDIDATES:
        out = RAW_DIR / name
        if out.exists() and is_real_gzip(out):
            print(f"[ok] tarball already downloaded: {out}")
            return out

    for name in TARBALL_CANDIDATES:
        url = TARBALL_URL_PREFIX + name
        out = RAW_DIR / name
        print(f"[curl] trying {url}")
        rc = subprocess.call(["curl", "-fL", "-o", str(out), url])
        if rc == 0 and is_real_gzip(out):
            return out
        if out.exists():
            out.unlink()
    raise RuntimeError(
        "Could not obtain SpreadsheetBench tarball. Tried: "
        + ", ".join(TARBALL_CANDIDATES)
    )


def extract(tarball: Path) -> None:
    marker = RAW_DIR / ".extracted"
    if marker.exists():
        print(f"[ok] already extracted -> {RAW_DIR}")
        return
    print(f"[tar] extracting {tarball} -> {RAW_DIR}")
    with tarfile.open(tarball, "r:gz") as tar:
        tar.extractall(path=RAW_DIR)
    top = [p for p in RAW_DIR.iterdir() if p.is_dir() and p.name != "spreadsheet"]
    if len(top) == 1 and (top[0] / "spreadsheet").exists():
        nested = top[0]
        print(f"[tar] hoisting {nested} -> {RAW_DIR}")
        for child in nested.iterdir():
            shutil.move(str(child), str(RAW_DIR / child.name))
        nested.rmdir()
    marker.touch()


def find_dataset_file() -> Path:
    for name in ("dataset.json", "all_data_912.json"):
        p = RAW_DIR / name
        if p.exists() and p.stat().st_size > 1024:
            return p
    candidates = list(RAW_DIR.glob("*.jsonl"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No dataset.json or *.jsonl in {RAW_DIR}")


def load_rows(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(l) for l in text.splitlines() if l.strip()]
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON list")
    return data


def build_splits() -> None:
    """Build train/val/test JSONLs using the pinned manifest."""
    src = find_dataset_file()
    rows = load_rows(src)
    print(f"[data] loaded {len(rows)} rows from {src.name}")

    by_id = {str(r["id"]): r for r in rows}
    dataset_root = str(RAW_DIR.resolve())

    for split, fname in SPLITS.items():
        manifest_file = SPLITS_DIR / split / "items.json"
        if not manifest_file.exists():
            print(f"[warn] manifest not found: {manifest_file}, skipping {split}")
            continue

        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        out_file = OUT_DIR / fname
        out_file.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        missing = []
        with out_file.open("w", encoding="utf-8") as f:
            for entry in manifest:
                eid = str(entry["id"])
                row = by_id.get(eid)
                if row is None:
                    missing.append(eid)
                    continue
                row = dict(row)
                row["dataset_root"] = dataset_root
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
        if missing:
            print(f"[warn] {split}: {len(missing)} IDs not found in raw (first 5: {missing[:5]})")
        print(f"[write] {out_file.relative_to(ROOT)} ({written} rows)")


def remove_legacy_files() -> None:
    for name in ("train_200.jsonl", "test_200.jsonl"):
        legacy = OUT_DIR / name
        if legacy.exists():
            legacy.unlink()
            print(f"[clean] removed legacy {legacy.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()

    clone_repo()
    tarball = fetch_tarball()
    extract(tarball)
    build_splits()
    remove_legacy_files()
    print("[done] SpreadsheetBench ready")


if __name__ == "__main__":
    main()
