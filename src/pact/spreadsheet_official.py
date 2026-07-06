from __future__ import annotations

import datetime
import glob as _glob
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import openpyxl


# ---------- value transform / compare (official port) ----------

def _datetime_to_float(dt: datetime.datetime) -> float:
    excel_start_date = datetime.datetime(1899, 12, 30)
    delta = dt - excel_start_date
    return delta.days + delta.seconds / 86400.0


def _transform_value(v):
    if isinstance(v, bool):
        # openpyxl can return Python bool; official code doesn't special-case
        # bools, but round(float(True), 2) == 1.0 which breaks 1 vs True. Keep
        # parity with the official transform by promoting bool -> float.
        return round(float(v), 2)
    if isinstance(v, (int, float)):
        return round(float(v), 2)
    if isinstance(v, datetime.time):
        return str(v)[:-3]
    if isinstance(v, datetime.datetime):
        return round(_datetime_to_float(v), 0)
    if isinstance(v, str):
        try:
            return round(float(v), 2)
        except ValueError:
            return v
    return v


def _compare_cell_value(v1, v2) -> bool:
    v1 = _transform_value(v1)
    v2 = _transform_value(v2)
    if (v1 == "" and v2 is None) or (v1 is None and v2 == ""):
        return True
    if (v1 == "" and v2 == "") or (v1 is None and v2 is None):
        return True
    if type(v1) is not type(v2):
        return False
    return v1 == v2


# ---------- range parsing (official port) ----------

def _col_num2name(n: int) -> str:
    name = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        name = chr(65 + r) + name
    return name


def _col_name2num(name: str) -> int:
    num = 0
    for c in name:
        num = num * 26 + (ord(c) - ord("A") + 1)
    return num


def _parse_range(range_str: str):
    start_cell, end_cell = range_str.split(":")
    sc = "".join(ch for ch in start_cell if ch.isalpha())
    sr = "".join(ch for ch in start_cell if ch.isdigit())
    ec = "".join(ch for ch in end_cell if ch.isalpha())
    er = "".join(ch for ch in end_cell if ch.isdigit())
    return (_col_name2num(sc), int(sr)), (_col_name2num(ec), int(er))


def _generate_cell_names(range_str: str):
    if ":" not in range_str:
        return [range_str]
    (sc, sr), (ec, er) = _parse_range(range_str)
    cols = [_col_num2name(i) for i in range(sc, ec + 1)]
    return [f"{c}{r}" for c in cols for r in range(sr, er + 1)]


def _cell_level_compare(wb_gt, wb_proc, sheet_name: str, cell_range: str):
    if sheet_name not in wb_proc.sheetnames:
        return False, f"worksheet not found: {sheet_name}"
    ws_gt = wb_gt[sheet_name]
    ws_proc = wb_proc[sheet_name]
    for cn in _generate_cell_names(cell_range):
        cg = ws_gt[cn]
        cp = ws_proc[cn]
        if not _compare_cell_value(cg.value, cp.value):
            return False, f"value@{sheet_name}!{cn}: gt={cg.value!r} pred={cp.value!r}"
    return True, ""


# ---------- public API ----------

def compare_workbooks(gt_file: str, proc_file: str, answer_position: str) -> tuple[bool, str]:
    if not os.path.exists(proc_file):
        return False, "file not exist"
    try:
        wb_gt = openpyxl.load_workbook(filename=gt_file, data_only=True)
        wb_proc = openpyxl.load_workbook(filename=proc_file, data_only=True)
    except Exception as e:  # noqa: BLE001
        return False, f"load error: {e}"
    try:
        ok_all = True
        msg_first = ""
        for scr in (answer_position or "").split(","):
            scr = scr.strip()
            if not scr:
                continue
            if "!" in scr:
                sheet_name, cell_range = scr.split("!", 1)
                sheet_name = sheet_name.strip().strip("'\"")
            else:
                sheet_name = wb_gt.sheetnames[0]
                cell_range = scr
            cell_range = cell_range.strip().strip("'\"")
            ok, msg = _cell_level_compare(wb_gt, wb_proc, sheet_name, cell_range)
            if not ok:
                ok_all = False
                if not msg_first:
                    msg_first = msg
        return ok_all, msg_first
    finally:
        wb_gt.close()
        wb_proc.close()


def evaluate(pred_path: str, gold_path: str,
             instruction_type: str, answer_position: str) -> dict:
    ok, msg = compare_workbooks(gold_path, pred_path, answer_position)
    return {
        "ok": ok,
        "reason": msg,
        "instruction_type": instruction_type,
    }


# ---------- SKILL-KD-specific: code extraction, execution, recalc ----------

def extract_code(response: str) -> str:
    """Extract the first fenced Python code block from text.

    Aligned with SkillOpt's extract_code in codegen_agent.py — takes the
    FIRST code block, not the last.
    """
    if "```" not in response:
        return response.strip()
    start = response.find("```")
    nl = response.find("\n", start)
    end = response.find("```", nl + 1)
    if nl == -1 or end == -1:
        return response.strip()
    return response[nl + 1 : end].strip()


def _find_test_cases(spreadsheet_dir: Path) -> list[tuple[str, Path, Path]]:
    """Discover test cases dynamically, aligned with SkillOpt's _find_test_cases.

    Supports naming conventions:
      * ``{no}_{id}_input.xlsx``  + ``{no}_{id}_answer.xlsx``  (original)
      * ``{no}_{id}_init.xlsx``   + ``{no}_{id}_golden.xlsx``  (verified_400)
      * ``initial.xlsx``          + ``golden.xlsx``             (verified_400, no prefix)

    Returns:
        Sorted list of (case_number, input_path, answer_path) tuples.
    """
    cases: list[tuple[str, Path, Path]] = []
    task_dir = str(spreadsheet_dir)

    # Try *_input.xlsx + *_answer.xlsx
    for ip_str in sorted(_glob.glob(os.path.join(task_dir, "*_input.xlsx"))):
        ip = Path(ip_str)
        no = ip.name.split("_", 1)[0]
        ap = Path(ip_str.replace("_input.xlsx", "_answer.xlsx"))
        if ap.exists():
            cases.append((no, ip, ap))

    # Try *_init.xlsx + *_golden.xlsx
    for ip_str in sorted(_glob.glob(os.path.join(task_dir, "*_init.xlsx"))):
        ip = Path(ip_str)
        no = ip.name.split("_", 1)[0]
        ap = Path(ip_str.replace("_init.xlsx", "_golden.xlsx"))
        if ap.exists():
            cases.append((no, ip, ap))

    # Fallback: bare initial.xlsx + golden.xlsx
    if not cases:
        bare_init = spreadsheet_dir / "initial.xlsx"
        bare_gold = spreadsheet_dir / "golden.xlsx"
        if bare_init.exists() and bare_gold.exists():
            cases.append(("1", bare_init, bare_gold))

    return cases


def spreadsheet_preview(input_file: Path, max_rows: int = 5, max_cols: int = 20) -> str:
    try:
        wb = openpyxl.load_workbook(input_file, data_only=False)
    except Exception as exc:
        return f"[preview unavailable: {type(exc).__name__}: {exc}]"
    chunks: list[str] = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        chunks.append(
            f"## Sheet: {sheet_name}  "
            f"(dim={ws.dimensions}, max_row={ws.max_row}, max_col={ws.max_column})"
        )
        for row in ws.iter_rows(
            min_row=1,
            max_row=min(ws.max_row, max_rows),
            max_col=min(ws.max_column, max_cols),
            values_only=False,
        ):
            cells = []
            for cell in row:
                v = cell.value
                if v is None:
                    cells.append(f"{cell.coordinate}=")
                else:
                    s = str(v)
                    if len(s) > 40:
                        s = s[:37] + "..."
                    cells.append(f"{cell.coordinate}={s}")
            chunks.append(" | ".join(cells))
        if ws.max_row > max_rows:
            chunks.append(f"... ({ws.max_row - max_rows} more rows)")
        chunks.append("")
    wb.close()
    return "\n".join(chunks)


def run_solution_on_cases(
    task: dict[str, Any],
    response: str,
    output_root: Path,
    timeout_s: int = 120,
) -> dict[str, Any]:
    """Execute extracted code against all test cases and score.

    Aligned with SkillOpt's process_one_codegen evaluation pipeline:
      - Same code is reused for all cases (no string replacement of paths).
      - Path injection via INPUT_PATH/OUTPUT_PATH in execute_python_code.
      - No LibreOffice recalculation — compare with data_only=True directly.
      - answer_sheet qualification for multi-sheet tasks.
    """
    code = extract_code(response)
    task_id = str(task["id"])
    dataset_root = Path(task["dataset_root"])
    spreadsheet_dir = dataset_root / task["spreadsheet_path"]
    output_dir = output_root / task_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── answer_sheet qualification (aligned with SkillOpt rollout.py:232-236) ──
    answer_position = task.get("answer_position", "")
    answer_sheet = task.get("answer_sheet", "")
    if answer_position and answer_sheet and "!" not in answer_position:
        answer_position_eval = f"{answer_sheet}!{answer_position}"
    else:
        answer_position_eval = answer_position

    # ── dynamic test case discovery (aligned with SkillOpt rollout.py:58-87) ──
    cases = _find_test_cases(spreadsheet_dir)

    case_results = []
    for case_idx, input_path, answer_path in cases:
        output_file = f"{case_idx}_{task_id}_output.xlsx"
        output_path = output_dir / output_file

        input_preview = spreadsheet_preview(input_path, max_rows=5) if input_path.exists() else ""
        answer_preview = spreadsheet_preview(answer_path, max_rows=5) if answer_path.exists() else ""

        # Execute the SAME code for all cases.  Path injection is handled by
        # execute_python_code (strips any model-defined INPUT_PATH/OUTPUT_PATH
        # and injects the correct per-case paths).  No string replacement of
        # filenames — aligned with SkillOpt's run_generated_code approach.
        exec_result = execute_python_code(
            code,
            output_dir,
            timeout_s,
            input_path=str(input_path),
            output_path=str(output_path),
        )
        if exec_result["returncode"] != 0:
            case_results.append(
                {
                    "case": case_idx,
                    "success": False,
                    "error": "execution_failed",
                    "execution": exec_result,
                    "input_path": str(input_path),
                    "answer_path": str(answer_path),
                    "output_path": str(output_path),
                    "input_preview": input_preview,
                    "answer_preview": answer_preview,
                }
            )
            continue

        alt_output = spreadsheet_dir / output_file
        if alt_output.exists():
            if not output_path.exists():
                shutil.move(str(alt_output), str(output_path))
            else:
                alt_output.unlink()

        # No LibreOffice recalculation — aligned with SkillOpt which reads
        # workbooks with data_only=True and relies on cached values.  The
        # system prompt tells the model to compute values in Python, never
        # write Excel formulas.
        comparison, msg = compare_workbooks(
            str(answer_path),
            str(output_path),
            answer_position_eval,
        )
        case_results.append(
            {
                "case": case_idx,
                "success": bool(comparison),
                "error": msg,
                "execution": exec_result,
                "recalc_output": {"available": False, "success": False, "backend": None},
                "input_path": str(input_path),
                "answer_path": str(answer_path),
                "output_path": str(output_path),
                "input_preview": input_preview,
                "answer_preview": answer_preview,
            }
        )

    hard_success = bool(case_results) and all(item["success"] for item in case_results)
    soft_score = (
        sum(1 for item in case_results if item["success"]) / len(case_results)
        if case_results
        else 0.0
    )
    return {
        "success": hard_success,
        "mode": "official_spreadsheetbench",
        "hard_restriction": int(hard_success),
        "soft_restriction": soft_score,
        "test_case_results": [int(item["success"]) for item in case_results],
        "cases": case_results,
        "solution_code": code,
        "requires_recalculation_backend": False,
        "recalculation_backend_available": bool(find_libreoffice_executable()),
    }


def execute_python_code(
    code: str,
    cwd: Path,
    timeout_s: int,
    input_path: str = "",
    output_path: str = "",
) -> dict[str, Any]:
    """Execute Python code with path injection, aligned with SkillOpt's run_generated_code.

    Strips user-defined INPUT_PATH/OUTPUT_PATH assignments and injects correct paths.
    Wraps user code in try/except to catch exceptions and exit with code 2.
    Uses sys.executable for consistent Python interpreter.
    """
    import sys
    import textwrap

    # Ensure output directory exists
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Strip model-defined INPUT_PATH/OUTPUT_PATH at module level (aligned with SkillOpt)
    code = re.sub(
        r'^\s*(INPUT_PATH|OUTPUT_PATH)\s*=\s*.+$',
        "",
        code,
        flags=re.MULTILINE,
    )

    # Build runner script with try/except wrapper (aligned with SkillOpt's RUNNER_TEMPLATE)
    indented_code = textwrap.indent(code, "    ")
    script_content = f"""
import os, sys, traceback
INPUT_PATH = {repr(input_path)}
OUTPUT_PATH = {repr(output_path)}
try:
{indented_code}
except Exception:
    traceback.print_exc()
    sys.exit(2)
"""

    with tempfile.NamedTemporaryFile("w", suffix=".py", encoding="utf-8", delete=False) as f:
        f.write(script_content)
        script = Path(f.name)
    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            text=True,
            capture_output=True,
            timeout=timeout_s,
        )
        if proc.returncode != 0:
            error_output = (proc.stdout + proc.stderr)[-5000:]
            return {
                "returncode": proc.returncode,
                "stdout": proc.stdout[-5000:],
                "stderr": proc.stderr[-5000:],
                "error": error_output,
            }
        # Check if output file was created (aligned with SkillOpt)
        if output_path and not os.path.exists(output_path):
            error_msg = "output file was not created"
            return {
                "returncode": 1,
                "stdout": proc.stdout[-5000:],
                "stderr": proc.stderr[-5000:],
                "error": error_msg,
            }
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout[-5000:],
            "stderr": proc.stderr[-5000:],
            "error": "",
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": -1,
            "stdout": (exc.stdout or "")[-5000:] if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "")[-5000:] if isinstance(exc.stderr, str) else "",
            "error": "timeout",
        }
    finally:
        script.unlink(missing_ok=True)


def find_libreoffice_executable() -> str | None:
    for name in ("libreoffice", "soffice"):
        exe = shutil.which(name)
        if exe:
            return exe
    for p in (
        "/Applications/LibreOffice.app/Contents/MacOS/soffice",
        "/Applications/LibreOffice.app/Contents/MacOS/libreoffice",
    ):
        if Path(p).exists():
            return p
    for p in (
        "/usr/bin/soffice",
        "/usr/bin/libreoffice",
        "/usr/local/bin/soffice",
        "/usr/local/bin/libreoffice",
        "/opt/libreoffice/program/soffice",
        Path.home() / "local/libreoffice/usr/bin/soffice",
        Path.home() / "local/libreoffice/usr/bin/libreoffice",
    ):
        if Path(p).exists():
            return str(p) if isinstance(p, Path) else p
    return None


def recalculate_workbook(path: Path) -> dict[str, Any]:
    executable = find_libreoffice_executable()
    if not executable or not path.exists():
        return {"available": False, "success": False, "backend": None}

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        profile_dir = tmp / "uno_profile"
        out_dir = tmp / "out"
        profile_dir.mkdir()
        out_dir.mkdir()
        proc = subprocess.run(
            [
                executable,
                f"-env:UserInstallation=file://{profile_dir}",
                "--headless",
                "--calc",
                "--convert-to",
                "xlsx:Calc MS Excel 2007 XML",
                "--outdir",
                str(out_dir),
                str(path),
            ],
            text=True,
            capture_output=True,
            timeout=120,
        )
        if proc.returncode == 0:
            converted = out_dir / path.name
            if converted.exists():
                shutil.move(str(converted), str(path))
                return {
                    "available": True,
                    "success": True,
                    "backend": executable,
                }
        return {
            "available": True,
            "success": False,
            "backend": executable,
            "stdout": proc.stdout[-1000:],
            "stderr": proc.stderr[-1000:],
        }
