from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_public_benchmark_configs_match_release_scope():
    benchmark_names = {path.stem for path in (ROOT / "configs" / "benchmarks").glob("*.yaml")}

    assert benchmark_names == {
        "alfworld",
        "docvqa",
        "livemathc",
        "searchqa",
        "spreadsheetbench",
    }
