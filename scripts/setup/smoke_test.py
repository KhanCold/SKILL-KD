"""Smoke test: every benchmark adapter can load all its splits."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pact.adapters import load_adapter  # noqa: E402


CASES = [
    ("alfworld", ROOT / "configs/benchmarks/alfworld.yaml",
     ["train", "test_seen", "test_unseen"]),
    ("spreadsheetbench", ROOT / "configs/benchmarks/spreadsheetbench.yaml",
     ["train", "test"]),
    ("searchqa", ROOT / "configs/benchmarks/searchqa.yaml",
     ["train", "test"]),
    ("livemathc", ROOT / "configs/benchmarks/livemathc.yaml",
     ["train", "test"]),
    ("docvqa", ROOT / "configs/benchmarks/docvqa.yaml",
     ["train", "test"]),
]


def main() -> int:
    failures = 0
    for name, cfg, splits in CASES:
        print(f"\n=== {name} ({cfg.name}) ===")
        try:
            adapter = load_adapter(cfg)
        except Exception as e:
            print(f"  [FAIL] load_adapter: {e}")
            failures += 1
            continue
        for split in splits:
            try:
                tasks = adapter.load_tasks(split, limit=3)
                print(f"  [ok] {split}: loaded {len(tasks)} tasks")
                if tasks:
                    first = tasks[0]
                    keys = list(first.keys())[:8]
                    print(f"        first keys: {keys}")
            except Exception as e:
                print(f"  [FAIL] {split}: {e}")
                failures += 1
    print(f"\n{'='*40}")
    print(f"failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
