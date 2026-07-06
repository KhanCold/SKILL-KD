from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_public_benchmark_configs_exclude_wikitq():
    benchmark_names = {path.stem for path in (ROOT / "configs" / "benchmarks").glob("*.yaml")}

    assert "wikitq" not in benchmark_names
