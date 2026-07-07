"""Prepare SearchQA data from public HF data and pinned split manifests.

Source of truth: data/searchqa/splits/{train,val,test}/items.json (checked into git).
Each is a JSON array of fixed public SearchQA keys. The script downloads
lucadiliello/searchqa and materialises the exact rows referenced by those keys.

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
HF_DATASET = "lucadiliello/searchqa"
HF_SPLITS = ("train", "validation")
REQUIRED_FIELDS = {"id", "question", "context", "answers"}


def _normalise_row(row: dict) -> dict:
    item_id = str(row.get("id") or row.get("key") or "").strip()
    answers = row.get("answers", [])
    if not isinstance(answers, list):
        answers = [str(answers)]
    return {
        "id": item_id,
        "question": str(row.get("question") or "").strip(),
        "context": str(row.get("context") or "").strip(),
        "answers": [str(answer) for answer in answers],
    }


def _manifest_needs_lookup(source_dir: Path) -> bool:
    for split in SPLITS:
        items_file = source_dir / split / "items.json"
        if not items_file.exists():
            continue
        rows = json.loads(items_file.read_text(encoding="utf-8"))
        for row in rows:
            if not REQUIRED_FIELDS.issubset(row):
                return True
    return False


def load_public_searchqa() -> dict[str, dict]:
    from datasets import load_dataset

    by_id: dict[str, dict] = {}
    for hf_split in HF_SPLITS:
        print(f"[searchqa] loading {HF_DATASET}:{hf_split}")
        dataset = load_dataset(HF_DATASET, split=hf_split)
        for row in dataset:
            item = _normalise_row(row)
            if item["id"]:
                by_id[item["id"]] = item
    return by_id


def transcode_split(
    source_dir: Path,
    split: str,
    out_file: Path,
    by_id: dict[str, dict] | None = None,
) -> int:
    items_file = source_dir / split / "items.json"
    if not items_file.exists():
        raise FileNotFoundError(f"Missing source split: {items_file}")
    rows = json.loads(items_file.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{items_file} did not contain a JSON array")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        for row in rows:
            item = row
            if not REQUIRED_FIELDS.issubset(item):
                if by_id is None:
                    raise ValueError(f"{items_file} contains ID-only rows but no public SearchQA index was loaded")
                item_id = str(item.get("id") or item.get("key") or "").strip()
                if item_id not in by_id:
                    raise KeyError(f"{items_file} references SearchQA key not found in public data: {item_id}")
                item = by_id[item_id]
            f.write(json.dumps(_normalise_row(item), ensure_ascii=False) + "\n")
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
    by_id = load_public_searchqa() if _manifest_needs_lookup(source) else None
    for split, fname in SPLITS.items():
        out_file = OUT_DIR / fname
        n = transcode_split(source, split, out_file, by_id=by_id)
        print(f"[searchqa] {split:>5}: {n:>5} items -> {out_file.relative_to(ROOT)}")
    print("[done] SearchQA ready")


if __name__ == "__main__":
    main()
