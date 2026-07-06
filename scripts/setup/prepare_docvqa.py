"""Prepare DocVQA data.

Strategy:
1. Download the 6 validation-split parquet files from `lmms-lab/DocVQA`
   at the pinned revision.
2. Scan each parquet for the rows whose questionId appears in the pinned
   split manifests at data/docvqa/splits/{train,val,test}/items.json
   (~534 rows out of the full 5349-row validation split).
3. For each kept row:
   - Decode the image bytes and save as
     `data/docvqa_images/q{questionId}_d{docId}.png`
   - Write a JSONL line containing question / answers / image_path / docId
     etc. (the schema DocVQAAdapter expects).

The 6 parquet files total ~1 GB. They land in `data/docvqa/raw/` and can
be safely deleted after this script completes. Per-row image PNG output is
~534 files for the full 10% subset, dominated by image bytes (not parquet).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

# HF mirror by default (override with HF_ENDPOINT=https://huggingface.co).
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "data" / "docvqa" / "splits"
RAW_DIR = ROOT / "data" / "docvqa" / "raw"
OUT_DIR = ROOT / "data" / "docvqa"
IMAGE_DIR = ROOT / "data" / "docvqa_images"

HF_REPO = "lmms-lab/DocVQA"
HF_REVISION = "539088ef8a8ada01ac8e2e6d4e372586748a265e"
HF_CONFIG = "DocVQA"
HF_SPLIT = "validation"
NUM_SHARDS = 6  # validation-00000-of-00006 ... validation-00005-of-00006

SPLITS = {
    "train": "train.jsonl",
    "val": "val.jsonl",
    "test": "test.jsonl",
}


def load_id_manifest(source_dir: Path) -> dict[str, dict]:
    """Return {questionId: manifest_row} across all 3 splits, tagging which split each ID belongs to."""
    by_qid: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        items_file = source_dir / split / "items.json"
        if not items_file.exists():
            raise FileNotFoundError(f"Missing ID manifest: {items_file}")
        rows = json.loads(items_file.read_text(encoding="utf-8"))
        for row in rows:
            qid = str(row.get("questionId") or row.get("id") or "").strip()
            if not qid:
                continue
            row = dict(row)
            row["_split"] = split
            by_qid[qid] = row
    return by_qid


def download_parquets() -> list[Path]:
    """Pull all 6 validation parquet shards at the pinned revision."""
    from huggingface_hub import hf_hub_download

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i in range(NUM_SHARDS):
        remote = f"{HF_CONFIG}/{HF_SPLIT}-{i:05d}-of-{NUM_SHARDS:05d}.parquet"
        local = RAW_DIR / Path(remote).name
        if local.exists() and local.stat().st_size > 0:
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
            local.write_bytes(Path(downloaded).read_bytes())
        paths.append(local)
    return paths


def _coerce_answers(raw) -> list[str]:
    """Force answers into a list[str] for the JSONL row."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(a).strip() for a in raw if str(a).strip()]
    if isinstance(raw, (bytes, bytearray)):
        try:
            return _coerce_answers(json.loads(raw.decode("utf-8")))
        except Exception:
            return [raw.decode("utf-8", errors="ignore").strip()]
    return [str(raw).strip()]


def _decode_image_bytes(image_field) -> bytes:
    """HF parquet image columns may be dict({'bytes': b'...', 'path': ...}) or raw bytes."""
    if isinstance(image_field, dict):
        b = image_field.get("bytes")
        if b is not None:
            return b
        path = image_field.get("path")
        if path:
            return Path(path).read_bytes()
        raise ValueError(f"image field dict has neither 'bytes' nor 'path': {list(image_field.keys())}")
    if isinstance(image_field, (bytes, bytearray)):
        return bytes(image_field)
    raise TypeError(f"Unsupported image field type: {type(image_field).__name__}")


def process_parquet_files(parquet_paths: list[Path], wanted_qids: dict[str, dict]) -> dict[str, dict]:
    """Scan parquet shards, materialise the wanted rows. Returns {qid: row}."""
    import pyarrow.parquet as pq
    from PIL import Image

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    found: dict[str, dict] = {}
    for shard_idx, path in enumerate(parquet_paths):
        print(f"[docvqa] scanning shard {shard_idx + 1}/{len(parquet_paths)}: {path.name}")
        pf = pq.ParquetFile(str(path))
        for batch in pf.iter_batches(batch_size=64):
            batch_dict = batch.to_pydict()
            n = len(next(iter(batch_dict.values())))
            for i in range(n):
                qid = str(batch_dict.get("questionId", [None] * n)[i] or "").strip()
                if not qid or qid not in wanted_qids:
                    continue
                manifest = wanted_qids[qid]
                did = str(batch_dict.get("docId", [None] * n)[i] or manifest.get("docId") or "").strip()
                question = str(batch_dict.get("question", [""] * n)[i] or "").strip()
                answers = _coerce_answers(batch_dict.get("answers", [None] * n)[i])
                image_field = batch_dict.get("image", [None] * n)[i]
                if image_field is None:
                    print(f"[warn] qid={qid} missing image field, skipping")
                    continue
                # Decode + re-encode as PNG to normalise format and confirm validity.
                raw = _decode_image_bytes(image_field)
                img = Image.open(io.BytesIO(raw))
                img.load()
                image_filename = f"q{qid}_d{did}.png" if did else f"q{qid}.png"
                image_out = IMAGE_DIR / image_filename
                if not image_out.exists():
                    img.save(image_out, format="PNG")
                image_rel = str(image_out.relative_to(ROOT))
                topic = manifest.get("topic", "docvqa")
                found[qid] = {
                    "id": qid,
                    "questionId": qid,
                    "docId": did,
                    "question": question,
                    "answer": answers[0] if answers else "",
                    "answers": answers,
                    "task_type": topic,
                    "subtask": topic,
                    "image_paths": [image_rel],
                    "image_path": image_rel,
                    "ucsf_document_id": manifest.get("ucsf_document_id", ""),
                    "ucsf_document_page_no": manifest.get("ucsf_document_page_no", ""),
                    "source_split": manifest.get("source_split", HF_SPLIT),
                }
        # Early exit if we've already found everything we want.
        if len(found) >= len(wanted_qids):
            print(f"[docvqa] all {len(wanted_qids)} wanted rows found, stopping shard scan early")
            break
    return found


def write_splits(source_dir: Path, found: dict[str, dict]) -> None:
    for split, fname in SPLITS.items():
        id_file = source_dir / split / "items.json"
        id_rows = json.loads(id_file.read_text(encoding="utf-8"))
        out_file = OUT_DIR / fname
        out_file.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        missing = []
        with out_file.open("w", encoding="utf-8") as f:
            for row in id_rows:
                qid = str(row.get("questionId") or row.get("id") or "").strip()
                item = found.get(qid)
                if item is None:
                    missing.append(qid)
                    continue
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                written += 1
        if missing:
            print(f"[warn] {split}: {len(missing)} qids missing (first 5: {missing[:5]})")
        print(f"[docvqa] {split:>5}: {written:>4} items -> {out_file.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Path to the docvqa splits directory.",
    )
    parser.add_argument(
        "--keep-parquet",
        action="store_true",
        help="Keep the downloaded parquet files in data/docvqa/raw/ after extraction.",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    if not source.exists():
        sys.exit(f"[error] Source not found: {source}")

    print(f"[docvqa] HF repo  = {HF_REPO} @ {HF_REVISION[:8]} ({HF_CONFIG}/{HF_SPLIT})")
    print(f"[docvqa] ID split= {source}")
    print(f"[docvqa] images -> {IMAGE_DIR}")
    print(f"[docvqa] splits -> {OUT_DIR}")

    wanted = load_id_manifest(source)
    print(f"[docvqa] {len(wanted)} unique questionIds requested across train+val+test")

    parquets = download_parquets()
    found = process_parquet_files(parquets, wanted)
    print(f"[docvqa] materialised {len(found)} / {len(wanted)} rows")

    write_splits(source, found)

    if not args.keep_parquet:
        for p in parquets:
            try:
                p.unlink()
            except OSError:
                pass
        cache = RAW_DIR / "_hf_cache"
        if cache.exists():
            import shutil
            shutil.rmtree(cache, ignore_errors=True)
        print(f"[docvqa] removed raw parquet files (use --keep-parquet to retain)")

    print("[done] DocVQA ready")


if __name__ == "__main__":
    main()
