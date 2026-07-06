import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.argv = ["viz_server.py"]
import viz_server


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_scan_results_slims_trajectories_for_initial_payload(tmp_path, monkeypatch):
    results = tmp_path / "results"
    run = results / "demo_20260622_120000"
    bench = run / "alfworld"
    group = bench / "evolution" / "groups" / "0000_task"

    _write_json(run / "run_config.json", {"student_model": "student"})
    _write_json(group / "task.json", {"task_type": "pick_and_place", "turk_annotations": {"anns": [{"task_desc": "do it"}]}})
    _write_json(group / "summary.json", {"success": False})
    _write_json(
        group / "student_initial.json",
        {
            "raw": "x" * 100_000,
            "messages": [{"role": "assistant", "content": "y" * 100_000}],
            "evaluation": {"success": False, "num_actions": 3, "steps": [{"action": "look"}]},
        },
    )
    _write_json(
        group / "round_00" / "critic.json",
        {"accepted": True, "raw": "z" * 100_000, "ops": [{"op": "add", "rule": {"title": "Tip"}}]},
    )

    monkeypatch.setattr(viz_server, "RESULTS_DIR", results)

    data = viz_server.scan_results()

    slim_group = data[0]["benchmarks"][0]["groups"][0]
    assert slim_group["student_initial"]["evaluation"] == {"success": False, "num_actions": 3}
    assert "raw" not in slim_group["student_initial"]
    assert "messages" not in slim_group["student_initial"]
    assert slim_group["rounds"][0]["critic"] == {"accepted": True}


def test_scan_group_detail_keeps_full_trajectory(tmp_path):
    group = tmp_path / "0000_task"
    _write_json(group / "task.json", {"question": "q", "table": {"header": []}})
    _write_json(group / "student_initial.json", {"raw": "full", "evaluation": {"success": False}})

    detail = viz_server.scan_group_detail(group)

    assert detail["student_initial"]["raw"] == "full"
