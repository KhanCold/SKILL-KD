#!/usr/bin/env bash
# ============================================================================
#  SKILL-KD Unified Experiment Runner
# ============================================================================
#
#  Usage:
#    bash scripts/run.sh
#
#  Modes:
#    MODE=test        — evaluate only (SKILL optional; uses initial_skill if omitted)
#    MODE=evolve      — train skill only
#    MODE=evolve+test — train then evaluate (default)
#
#  Examples:
#    # Evaluate a saved skill
#    EXP_NAME="baseline" MODE=test SKILL=results/xxx/skills/final.md bash scripts/run.sh
#
#    # Evaluate per-benchmark evolved skills from a previous run (directory mode)
#    EXP_NAME="retest" MODE=test SKILL=results/qwen3-4b_qwen3.6-plus_20260611_151825 bash scripts/run.sh
#
#    # Evolve + test on two benchmarks
#    EXP_NAME='main' \
#      BENCHMARKS="alfworld spreadsheetbench" \
#      STUDENT_MODEL=qwen3-4b \
#      TEACHER_MODEL=qwen3.6-plus \
#      bash scripts/run.sh
#
#    # Smoke test across all benchmarks (small limits)
#    EXP_NAME='smoke' \
#      BENCHMARKS="alfworld spreadsheetbench searchqa livemathc docvqa" \
#      STUDENT_MODEL=qwen3-4b \
#      TEACHER_MODEL=qwen3.6-plus \
#      TRAIN_LIMIT=10 EVAL_LIMIT=10 \
#      bash scripts/run.sh
#
#    # Resume a previous run
#    RESUME=1 EXP_NAME='main' bash scripts/run.sh
#
#  Note: DocVQA is multimodal (image + text). Pure-text models like qwen3-4b
#  will receive image_url parts but ignore them — use a qwen-vl-* model instead.
#
#  All parameters (override via environment variables):
#    MODE=evolve+test
#    BENCHMARKS="spreadsheetbench"
#    TRAIN_LIMIT=           # stratified subsampling when < pool size
#    EVAL_LIMIT=            # same; alfworld test_seen/test_unseen subsample independently
#    MAX_ITERS=3
#    STUDENT_MODEL=qwen3-8b
#    TEACHER_MODEL=         # required for evolve/evolve+test
#    SKILL=path/to/skill.md   # single file (all benchmarks) or run directory (per-benchmark auto-resolve)
#    EXP_NAME=my_experiment
#    SKILL_KD_API_KEY_ENV=DASHSCOPE_API_KEY
#    SKILL_KD_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
#
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

trim_env_value() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

# Load .env file
if [[ -f ".env" ]]; then
  while IFS='=' read -r key value; do
    key="$(trim_env_value "${key}")"
    value="$(trim_env_value "${value}")"
    value="${value%\"}"
    value="${value#\"}"
    value="${value%\'}"
    value="${value#\'}"
    [[ -z "${key}" || "${key}" == \#* || -z "${value}" ]] && continue
    if [[ -z "${!key:-}" ]]; then
      export "${key}=${value}"
    fi
  done < ".env"
fi

# Default parameters
MODE="${MODE:-evolve+test}"
if [[ -z "${BENCHMARKS+x}" ]]; then
  BENCHMARKS="spreadsheetbench"
fi
TRAIN_LIMIT="${TRAIN_LIMIT:-}"
EVAL_LIMIT="${EVAL_LIMIT:-}"
MAX_ITERS="${SKILL_KD_MAX_ITERS:-${MAX_ITERS:-3}}"
STUDENT_MODEL="${STUDENT_MODEL:-qwen3-8b}"
# Teacher is only meaningful in evolve / evolve+test. Leave it empty in test mode
# so it isn't passed to the runner — otherwise run_id and run_config.json would
# record a teacher that was never actually used.
if [[ -z "${TEACHER_MODEL:-}" && "${MODE}" != "test" ]]; then
  TEACHER_MODEL="qwen3.5-flash"
fi
TEACHER_MODEL="${TEACHER_MODEL:-}"
API_KEY_ENV="${SKILL_KD_API_KEY_ENV:-DASHSCOPE_API_KEY}"
BASE_URL="${SKILL_KD_BASE_URL:-${OPENAI_BASE_URL:-${DASHSCOPE_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}}}"
STUDENT_BASE_URL="${STUDENT_BASE_URL:-}"
TEACHER_BASE_URL="${TEACHER_BASE_URL:-}"
EXP_NAME="${EXP_NAME:-}"
SKILL="${SKILL:-}"
RUN_ID="${RUN_ID:-}"

RESUME_FLAG=""
if [[ "${1:-}" == "--resume" || "${RESUME:-}" == "1" || "${RESUME:-}" == "true" ]]; then
  RESUME_FLAG="--resume"
fi

if [[ -z "${!API_KEY_ENV:-}" ]]; then
  echo "[warning] ${API_KEY_ENV} not set — using fallback 'not-needed' (self-hosted model)." >&2
  export "${API_KEY_ENV}=not-needed"
fi

PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python"
fi

echo "========================================"
echo "  SKILL-KD Experiment: ${MODE}"
echo "========================================"
echo "Benchmarks  : ${BENCHMARKS}"
echo "Train limit : ${TRAIN_LIMIT:-all}"
echo "Eval limit  : ${EVAL_LIMIT:-all}"
echo "Max iters   : ${MAX_ITERS}"
echo "Student     : ${STUDENT_MODEL}"
if [[ "${MODE}" != "test" ]]; then
  echo "Teacher     : ${TEACHER_MODEL}"
fi
if [[ -n "${EXP_NAME}" ]]; then
  echo "Exp name    : ${EXP_NAME}"
fi
if [[ -n "${SKILL}" ]]; then
  echo "Skill       : ${SKILL}"
fi
echo "========================================"

# Build argument list safely
ARGS=(
  --mode "${MODE}"
  --max-iters "${MAX_ITERS}"
  --student-model "${STUDENT_MODEL}"
  --base-url "${BASE_URL}"
  --api-key-env "${API_KEY_ENV}"
)
if [[ -n "${BENCHMARKS}" ]]; then
  ARGS+=(--benchmarks ${BENCHMARKS})
fi
if [[ -n "${STUDENT_BASE_URL}" ]]; then
  ARGS+=(--student-base-url "${STUDENT_BASE_URL}")
fi
if [[ -n "${TEACHER_BASE_URL}" ]]; then
  ARGS+=(--teacher-base-url "${TEACHER_BASE_URL}")
fi
if [[ -n "${TRAIN_LIMIT}" ]]; then
  ARGS+=(--train-limit "${TRAIN_LIMIT}")
fi
if [[ -n "${EVAL_LIMIT}" ]]; then
  ARGS+=(--eval-limit "${EVAL_LIMIT}")
fi
if [[ -n "${TEACHER_MODEL}" ]]; then
  ARGS+=(--teacher-model "${TEACHER_MODEL}")
fi
if [[ -n "${EXP_NAME}" ]]; then
  ARGS+=(--exp-name "${EXP_NAME}")
fi
if [[ -n "${SKILL}" ]]; then
  ARGS+=(--skill "${SKILL}")
fi
if [[ -n "${RUN_ID}" ]]; then
  ARGS+=(--run-id "${RUN_ID}")
fi
if [[ -n "${RESUME_FLAG}" ]]; then
  ARGS+=("${RESUME_FLAG}")
fi

"${PYTHON_BIN}" -m pact.run_experiment "${ARGS[@]}"
