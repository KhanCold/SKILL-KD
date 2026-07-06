# SKILL-KD

Code for **SKILL-KD: Contrastive Skill Distillation for LLM Agents**.

SKILL-KD improves frozen student agents by converting teacher-student behavioral
gaps into reusable textual skills. For each benchmark, a student and teacher
share a `skill.md`; when the student fails, the teacher runs the same task, a
critic proposes skill edits, and the student retries with the patched skill.

## Setup

```bash
brew install python@3.11
brew install --cask libreoffice          # SpreadsheetBench formula recalculation

bash scripts/setup/setup_all.sh
cp .env.example .env
```

Set `DASHSCOPE_API_KEY` in `.env`, or configure any OpenAI-compatible endpoint
with `SKILL_KD_BASE_URL` and `SKILL_KD_API_KEY_ENV`.

Skip individual benchmark preparation with:
`SKIP_ALF=1`, `SKIP_SSB=1`, `SKIP_SEARCHQA=1`, `SKIP_LIVEMATHC=1`,
or `SKIP_DOCVQA=1`.

## Benchmarks

| Benchmark | Train | Val | Test | Metric |
|---|---:|---:|---:|---|
| SearchQA | 400 | 200 | 1400 | EM / F1 |
| SpreadsheetBench | 80 | 40 | 280 | Workbook diff |
| DocVQA | 107 | 53 | 374 | ANLS |
| LiveMathC | 35 | 18 | 124 | Exact match |
| ALFWorld | 39 | 18 | 134 | Environment success |

DocVQA is multimodal; use a vision-capable student model for meaningful scores.

## Run

Use `scripts/run.sh` as the single entry point.

```bash
# Evolve + test
EXP_NAME=my-run \
  BENCHMARKS="searchqa spreadsheetbench docvqa livemathc alfworld" \
  STUDENT_MODEL=qwen3-4b \
  TEACHER_MODEL=qwen3.6-plus \
  TRAIN_LIMIT=50 EVAL_LIMIT=50 \
  bash scripts/run.sh

# Test only with a saved skill
EXP_NAME=baseline MODE=test \
  BENCHMARKS="spreadsheetbench" \
  SKILL=results/<run>/spreadsheetbench/skills/final.md \
  STUDENT_MODEL=qwen3-4b \
  bash scripts/run.sh

# Resume
RESUME=1 EXP_NAME=my-run bash scripts/run.sh
```

Important environment variables:

| Variable | Default | Description |
|---|---|---|
| `MODE` | `evolve+test` | `evolve`, `test`, or `evolve+test` |
| `BENCHMARKS` | `spreadsheetbench` | Space-separated benchmark names |
| `STUDENT_MODEL` | `qwen3-8b` | Student model |
| `TEACHER_MODEL` | `qwen3.5-flash` | Teacher and critic model |
| `TRAIN_LIMIT` | all | Optional training subsample |
| `EVAL_LIMIT` | all | Optional eval subsample |
| `MAX_ITERS` | `3` | Max adaptive skill-edit rounds per task |

## Results

Runs are written to `results/<run_id>/`. This source release does not include
full experiment trajectories, metrics, or raw outputs. It only keeps the final
4B skill snapshots at:

```text
results/4b-skills/<benchmark>/skills/final.md
```

To inspect newly generated results:

```bash
python viz_server.py            # http://localhost:8080
python viz_server.py 9000
```

## Layout

```text
src/pact/                  Core algorithm, adapters, prompts, skill ops
configs/benchmarks/        Benchmark configs
configs/skills/            Initial skill files
scripts/setup/             Data preparation
scripts/run.sh             Experiment launcher
tests/                     Unit tests
```

## Tests

```bash
pytest -q
```

Tests mock LLM calls and do not require network access.
