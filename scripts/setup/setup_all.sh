#!/usr/bin/env bash
# ===========================================================================
#  SKILL-KD one-shot local setup
# ===========================================================================
#  Prereqs (one-time, manual):
#    - brew install python@3.11
#    - brew install --cask libreoffice   (only needed for SpreadsheetBench
#                                         official evaluation = formula recalc)
#  Then:
#    bash scripts/setup/setup_all.sh
#
#  Skip individual benchmarks with env flags:
#    SKIP_ALF=1 bash scripts/setup/setup_all.sh
#    SKIP_SSB=1 bash scripts/setup/setup_all.sh
#    SKIP_SEARCHQA=1 bash scripts/setup/setup_all.sh
#    SKIP_LIVEMATHC=1 bash scripts/setup/setup_all.sh
#    SKIP_DOCVQA=1 bash scripts/setup/setup_all.sh
# ===========================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV="${ROOT}/.venv"

echo "[setup] root=${ROOT}"
echo "[setup] python=${PYTHON_BIN}"

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "[venv] creating ${VENV}"
  "${PYTHON_BIN}" -m venv "${VENV}"
  "${VENV}/bin/pip" install -U pip wheel setuptools
fi

echo "[pip] installing skill-kd[alfworld,dev]"
"${VENV}/bin/pip" install -e '.[alfworld,dev]'

if [[ -z "${SKIP_SSB:-}" ]]; then
  echo "[data] SpreadsheetBench"
  "${VENV}/bin/python" scripts/setup/prepare_spreadsheetbench.py
fi

if [[ -z "${SKIP_ALF:-}" ]]; then
  echo "[data] ALFWorld"
  "${VENV}/bin/python" scripts/setup/prepare_alfworld.py
fi

if [[ -z "${SKIP_SEARCHQA:-}" ]]; then
  echo "[data] SearchQA"
  "${VENV}/bin/python" scripts/setup/prepare_searchqa.py
fi

if [[ -z "${SKIP_LIVEMATHC:-}" ]]; then
  echo "[data] LiveMathematicianBench"
  "${VENV}/bin/python" scripts/setup/prepare_livemathc.py
fi

if [[ -z "${SKIP_DOCVQA:-}" ]]; then
  echo "[data] DocVQA"
  "${VENV}/bin/python" scripts/setup/prepare_docvqa.py
fi

echo "[done] SKILL-KD environment ready. Activate with: source .venv/bin/activate"
