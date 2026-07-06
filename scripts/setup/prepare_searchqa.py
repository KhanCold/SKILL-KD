"""Prepare SearchQA data by transcoding the pinned split manifests.

Source of truth: data/searchqa/splits/{train,val,test}/items.json (checked into git).
Each is a JSON array of items with id/question/context/answers.

Output: data/searchqa/{train,val,test}.jsonl (one item per line) — the schema
the SearchQAAdapter expects.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "data" / "searchqa" / "splits"
OUT_DIR = ROOT / "data" / "searchqa"


SPLITS = {
    "train": "train.jsonl",
    "val": "val.jsonl",
    "test": "test.jsonl",
}


def transcode_split(source_dir: Path, split: str, out_file: Path) -> int:
    items_file = source_dir / split / "items.json"
    if not items_file.exists():
        raise FileNotFoundError(f"Missing source split: {items_file}")
    rows = json.loads(items_file.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{items_file} did not contain a JSON array")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        for row in rows:
            # Defensive normalisation: guarantee answers is a list of strings.
            if "answers" in row and not isinstance(row["answers"], list):
                row["answers"] = [str(row["answers"])]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Path to the searchqa splits directory.",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    if not source.exists():
        sys.exit(f"[error] Source not found: {source}")

    print(f"[searchqa] source = {source}")
    print(f"[searchqa] dest   = {OUT_DIR}")
    for split, fname in SPLITS.items():
        out_file = OUT_DIR / fname
        n = transcode_split(source, split, out_file)
        print(f"[searchqa] {split:>5}: {n:>5} items -> {out_file.relative_to(ROOT)}")
    print("[done] SearchQA ready")


if __name__ == "__main__":
    main()
