"""Prepare LiveMathematicianBench data.

Strategy:
1. Download 4 monthly `qa_<month>_final.json` files from HuggingFace at the
   pinned revision.
2. Use the pinned split manifests at data/livemathc/splits/{train,val,test}/items.json
   (checked into git) to slice the raw items into the same train/val/test split
   (35/18/124).
3. Normalize each item to the schema the LiveMathCAdapter expects.

Output: data/livemathc/{train,val,test}.jsonl. Adapter then applies
`_shuffle_item_choices` at load time using seed 42.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# HF mirror by default (override with HF_ENDPOINT if you have direct access).
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "data" / "livemathc" / "splits"
RAW_DIR = ROOT / "data" / "livemathc" / "raw"
OUT_DIR = ROOT / "data" / "livemathc"

HF_REPO = "LiveMathematicianBench/LiveMathematicianBench"
HF_REVISION = "b72450f6ce96c26158d64d945a5d31ef7727be41"
MONTHS = ["202511", "202512", "202601", "202602"]

SPLITS = {
    "train": "train.jsonl",
    "val": "val.jsonl",
    "test": "test.jsonl",
}

_CHOICE_LABELS = ["A", "B", "C", "D", "E", "F", "G"]


def _normalize_label(text: str) -> str:
    return str(text).strip().upper().rstrip(".):")


def _coerce_choices(raw_choices) -> list[dict]:
    """Coerce raw choice data into a list of {label, text} dicts."""
    if isinstance(raw_choices, list):
        choices: list[dict] = []
        for idx, item in enumerate(raw_choices):
            if isinstance(item, dict):
                label = str(item.get("label") or _CHOICE_LABELS[idx]).strip()
                text = str(item.get("text") or item.get("content") or "").strip()
            else:
                label = _CHOICE_LABELS[idx]
                text = str(item).strip()
            if text:
                choices.append({"label": label, "text": text})
        return choices
    if isinstance(raw_choices, dict):
        labels = sorted(raw_choices.keys())
        return [
            {"label": str(label).strip(), "text": str(raw_choices[label]).strip()}
            for label in labels
            if str(raw_choices[label]).strip()
        ]
    return []


def _coerce_theorem_types(raw) -> list[str]:
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if raw is None:
        return []
    text = str(raw).strip()
    return [text] if text else []


def _normalize_item(item: dict, row_idx: int, source_path: str) -> dict:
    """Normalize a raw HF item to the schema LiveMathCAdapter expects."""
    mcq = item.get("mcq", {}) if isinstance(item.get("mcq"), dict) else {}
    question = str(mcq.get("question") or item.get("question") or "").strip()
    choices = _coerce_choices(mcq.get("choices") or item.get("choices") or [])
    correct = mcq.get("correct_choice") or item.get("correct_choice") or {}

    if isinstance(correct, dict):
        correct_label = _normalize_label(correct.get("label", ""))
        correct_text = str(correct.get("text") or "").strip()
    else:
        correct_label = _normalize_label(correct)
        correct_text = ""

    choice_by_label = {_normalize_label(c["label"]): c["text"] for c in choices}
    if correct_label and not correct_text:
        correct_text = choice_by_label.get(correct_label, "")
    if correct_label and correct_text and correct_label not in choice_by_label:
        choices.append({"label": correct_label, "text": correct_text})
        choices.sort(
            key=lambda c: _CHOICE_LABELS.index(c["label"])
            if c["label"] in _CHOICE_LABELS
            else len(_CHOICE_LABELS)
        )
        choice_by_label[correct_label] = correct_text

    month = str(item.get("month") or "").strip()
    item_no = item.get("no", row_idx + 1)
    item_id = f"{month}:{item_no}" if month else str(item_no)

    return {
        "id": item_id,
        "month": month,
        "no": item_no,
        "paper_link": str(item.get("paper_link") or "").strip(),
        "theorem": str(item.get("theorem") or "").strip(),
        "sketch": str(item.get("sketch") or "").strip(),
        "theorem_type": _coerce_theorem_types(item.get("theorem_type")),
        "question": question,
        "choices": choices,
        "correct_choice": {"label": correct_label, "text": correct_text},
        "source_path": source_path,
    }


def download_monthly_files() -> dict[str, Path]:
    """Pull the 4 monthly qa_*_final.json files at the pinned revision."""
    from huggingface_hub import hf_hub_download

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for month in MONTHS:
        remote = f"data/{month}/qa_{month}_final.json"
        local = RAW_DIR / f"qa_{month}_final.json"
        if local.exists():
            print(f"[ok] {local.relative_to(ROOT)} already cached, skipping")
        else:
            print(f"[hf] downloading {remote} @ {HF_REVISION[:8]}")
            downloaded = hf_hub_download(
                repo_id=HF_REPO,
                repo_type="dataset",
                revision=HF_REVISION,
                filename=remote,
                local_dir=str(RAW_DIR / "_hf_cache"),
            )
            # Move the downloaded file to a stable path the rest of the script can use.
            local.write_bytes(Path(downloaded).read_bytes())
        paths[month] = local
    return paths


def load_all_items(monthly_files: dict[str, Path]) -> dict[str, dict]:
    """Read raw items and normalise. Return {item_id: normalised_item}."""
    by_id: dict[str, dict] = {}
    for month, path in monthly_files.items():
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError(f"Expected JSON array in {path}, got {type(raw).__name__}")
        for row_idx, item in enumerate(raw):
            # Some upstream items lack the "month" field; backfill from the file name
            # so item_id keys align with the split manifest entries.
            if not item.get("month"):
                item["month"] = month
            norm = _normalize_item(item, row_idx=row_idx, source_path=str(path))
            if norm["question"] and norm["choices"] and norm["correct_choice"]["label"]:
                by_id[norm["id"]] = norm
    return by_id


def write_split(source_id_dir: Path, split: str, by_id: dict[str, dict], out_file: Path) -> int:
    id_file = source_id_dir / split / "items.json"
    if not id_file.exists():
        raise FileNotFoundError(f"Missing ID manifest: {id_file}")
    id_rows = json.loads(id_file.read_text(encoding="utf-8"))
    out_file.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    missing = []
    with out_file.open("w", encoding="utf-8") as f:
        for row in id_rows:
            iid = row["id"]
            item = by_id.get(iid)
            if item is None:
                missing.append(iid)
                continue
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            written += 1
    if missing:
        print(
            f"[warn] {split}: {len(missing)} IDs in manifest not found in HF data "
            f"(first 5: {missing[:5]})"
        )
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Path to the livemathc splits directory.",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    if not source.exists():
        sys.exit(f"[error] Source not found: {source}")

    print(f"[livemathc] HF repo  = {HF_REPO} @ {HF_REVISION[:8]}")
    print(f"[livemathc] months   = {MONTHS}")
    print(f"[livemathc] ID split = {source}")
    print(f"[livemathc] dest     = {OUT_DIR}")

    monthly = download_monthly_files()
    by_id = load_all_items(monthly)
    print(f"[livemathc] loaded {len(by_id)} normalised items from HF")

    for split, fname in SPLITS.items():
        out_file = OUT_DIR / fname
        n = write_split(source, split, by_id, out_file)
        print(f"[livemathc] {split:>5}: {n:>3} items -> {out_file.relative_to(ROOT)}")
    print("[done] LiveMathematicianBench ready")


if __name__ == "__main__":
    main()
