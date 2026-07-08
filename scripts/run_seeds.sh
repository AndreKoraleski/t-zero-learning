#!/usr/bin/env bash
# Train one config for multiple seeds, sequentially or in parallel.
#
# Usage (from anywhere):
#   CONFIG=ppo_antdir_goal ./scripts/run_seeds.sh
#   CONFIG=ppo_metaworld_mt10 NUM_SEEDS=5 PARALLEL=3 FIRST_SEED=2000 ./scripts/run_seeds.sh
#
# Environment variables:
#   CONFIG          (required) Config stem under configs/ (e.g. ppo_antdir_goal)
#   NUM_SEEDS       Total number of runs (default: 5)
#   FIRST_SEED      Base seed; runs use FIRST_SEED, FIRST_SEED+1, ... (default: 1000)
#   PARALLEL        Max trainings running at once (default: 1 = sequential).
#                   WARNING: parallel trainings share the GPU — watch memory.
#   CONDA_ENV       If set (e.g. cleanrl), runs: conda run -n "$CONDA_ENV" --no-capture-output python ...
#                   Ignored when RUN_IN_CONTAINER=1 (Apptainer image supplies python).
#   EXTRA_OVERRIDE  Optional extra train.py --override arguments (space-separated)
#   LOG_DIR         Where per-seed logs go (default: logs/<CONFIG>)
#   WORKDIR         If set, use as repo root instead of inferring from this script's path (cluster / Slurm).
#   RUN_ROOT  Same as WORKDIR; either wins over auto-detection.
#
# Headless / cluster: export MUJOCO_GL=egl (or osmesa) before running so MuJoCo can render off-screen.
#
# Examples:
#   CONDA_ENV=cleanrl MUJOCO_GL=egl CONFIG=ppo_halfcheetahvel ./scripts/run_seeds.sh
#   NUM_SEEDS=3 FIRST_SEED=1001 CONFIG=ppo_antdir_goal ./scripts/run_seeds.sh
#   # Sweep env kwargs (one run per value):
#   for vel in -1.0 0.5 2.0; do
#     CONFIG=ppo_halfcheetahvel NUM_SEEDS=1 \
#       EXTRA_OVERRIDE='env_kwargs={"target_vel": '"$vel"'}' ./scripts/run_seeds.sh
#   done

set -euo pipefail

if [[ -n "${RUN_ROOT:-}" ]]; then
  ROOT="$RUN_ROOT"
elif [[ -n "${WORKDIR:-}" ]]; then
  ROOT="$WORKDIR"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$ROOT"

if [[ -z "${CONFIG:-}" ]]; then
  echo "ERROR: CONFIG is required (config stem under configs/), e.g.:" >&2
  echo "  CONFIG=ppo_antdir_goal $0" >&2
  exit 2
fi

NUM_SEEDS="${NUM_SEEDS:-5}"
FIRST_SEED="${FIRST_SEED:-1000}"
PARALLEL="${PARALLEL:-1}"
LOG_DIR="${LOG_DIR:-logs/$CONFIG}"

mkdir -p "$LOG_DIR"

if [[ -n "${CONDA_ENV:-}" ]] && [[ -z "${RUN_IN_CONTAINER:-}" ]]; then
  run_python() {
    conda run -n "$CONDA_ENV" --no-capture-output python "$@"
  }
else
  run_python() {
    python "$@"
  }
fi

extra_args=()
if [[ -n "${EXTRA_OVERRIDE:-}" ]]; then
  # shellcheck disable=SC2206
  extra_args=($EXTRA_OVERRIDE)
fi

echo "Repo: $ROOT"
echo "Config: $CONFIG | Seeds: $NUM_SEEDS starting at $FIRST_SEED | Parallel: $PARALLEL | CONDA_ENV=${CONDA_ENV:-<none>} | container=${RUN_IN_CONTAINER:-0}"

fail=0
pids=()

for ((i = 0; i < NUM_SEEDS; i++)); do
  SEED=$((FIRST_SEED + i))
  log="$LOG_DIR/${CONFIG}_seed${SEED}.log"
  echo "=== [$((i + 1))/$NUM_SEEDS] seed=$SEED -> $log ==="
  if (( PARALLEL <= 1 )); then
    run_python train.py \
      --config "$CONFIG" \
      --override "seed=${SEED}" \
      "${extra_args[@]}" \
      2>&1 | tee "$log"
  else
    run_python train.py \
      --config "$CONFIG" \
      --override "seed=${SEED}" \
      "${extra_args[@]}" \
      >"$log" 2>&1 &
    pids+=("$!")
    # Throttle: keep at most PARALLEL trainings running
    while (( $(jobs -rp | wc -l) >= PARALLEL )); do
      wait -n || fail=1
    done
  fi
done

for p in "${pids[@]:-}"; do
  [[ -n "$p" ]] && { wait "$p" || fail=1; }
done

if (( fail )); then
  echo "ERROR: one or more seed runs failed. Check logs under $LOG_DIR" >&2
  exit 1
fi

echo "Done. Logs under $LOG_DIR"
