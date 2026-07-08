#!/bin/bash
#
# One Slurm allocation (1 GPU) runs scripts/run_seeds.sh: NUM_SEEDS seeds for CONFIG,
# with at most PARALLEL trainings at once (each training gets seeds BASE_SEED,
# BASE_SEED+1, ...).
#
# By default the seeds script runs *inside* the Apptainer/Singularity image: no host conda — the container provides Python +
# deps. Override with USE_APPTAINER=0 only if train.py works directly on the host
# (e.g. conda on the node).
#
# Repo path on the compute node (Slurm may copy this script to spool). Set WORKDIR or
# RUN_ROOT to the host checkout:
#   sbatch --export=ALL,WORKDIR=/raid/you/t-zero,CONFIG=ppo_antdir_goal cluster/run_seeds_parallel.sh
#
# Other overrides:
#   sbatch --export=ALL,WORKDIR=/raid/...,CONFIG=ppo_metaworld_mt10,NUM_SEEDS=5,PARALLEL=3 cluster/run_seeds_parallel.sh
#   sbatch --export=ALL,WORKDIR=/raid/...,CONFIG=...,BASE_SEED=2000 cluster/run_seeds_parallel.sh
#   sbatch --export=ALL,WORKDIR=/raid/...,CONFIG=...,USE_APPTAINER=0,CONDA_ENV=cleanrl cluster/run_seeds_parallel.sh
#   sbatch --export=ALL,WORKDIR=/raid/...,CONFIG=...,IMAGE_FILE=/path/t-zero.sif cluster/run_seeds_parallel.sh
#
# Resource knobs (job name, CPUs, memory) are set per submission when the defaults below
# don't fit:
#   sbatch --job-name=mw-mt10 --cpus-per-task=32 --mem=20G --export=ALL,... cluster/run_seeds_parallel.sh
#
#SBATCH --job-name=run-seeds-parallel
#SBATCH --partition=h100n2
#SBATCH --nodelist=dgx-H100-02
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=40G
#SBATCH --output=slurm_run_seeds_parallel_%j.out
#SBATCH --error=slurm_run_seeds_parallel_%j.err
#SBATCH --time=72:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${RUN_ROOT:-}" ]]; then
  ROOT="$RUN_ROOT"
elif [[ -n "${WORKDIR:-}" ]]; then
  ROOT="$WORKDIR"
else
  ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
if [[ ! -f "$ROOT/train.py" ]]; then
  echo "ERROR: ROOT=$ROOT is not the t-zero repo (train.py missing). Set WORKDIR or RUN_ROOT." >&2
  exit 2
fi
cd "$ROOT"

if [[ -z "${CONFIG:-}" ]]; then
  echo "ERROR: CONFIG is required, e.g. sbatch --export=ALL,CONFIG=ppo_antdir_goal $0" >&2
  exit 2
fi

USE_APPTAINER="${USE_APPTAINER:-1}"

DEF_FILE="${DEF_FILE:-$ROOT/cluster/t-zero.def}"
IMAGE_FILE="${IMAGE_FILE:-$ROOT/t-zero.sif}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"
ENV_FILE="${ENV_FILE:-$ROOT/.env}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"

BASE_SEED="${BASE_SEED:-1000}"
NUM_SEEDS="${NUM_SEEDS:-5}"
PARALLEL="${PARALLEL:-3}"
LOG_DIR="${LOG_DIR:-logs/$CONFIG}/parallel_${SLURM_JOB_ID:-local}"

mkdir -p "$LOG_DIR" "$ROOT/runs" "$ROOT/wandb"

EXEC_BIN=""
if command -v apptainer >/dev/null 2>&1; then
  EXEC_BIN="apptainer"
elif command -v singularity >/dev/null 2>&1; then
  EXEC_BIN="singularity"
fi

ENV_FILE_ARGS=()
if [[ -f "$ENV_FILE" ]]; then
  ENV_FILE_ARGS+=(--env-file "$ENV_FILE")
fi

if [[ "$USE_APPTAINER" == "1" ]]; then
  if [[ -z "$EXEC_BIN" ]]; then
    echo "ERROR: USE_APPTAINER=1 but neither apptainer nor singularity is in PATH." >&2
    exit 2
  fi

  if [[ ! -f "$IMAGE_FILE" ]]; then
    echo "Building image: $IMAGE_FILE from $DEF_FILE"
    BUILD_FLAGS=(--fakeroot --ignore-fakeroot-command)
    "$EXEC_BIN" build "${BUILD_FLAGS[@]}" "$IMAGE_FILE" "$DEF_FILE"
  elif [[ "$FORCE_REBUILD" == "1" ]]; then
    echo "Force rebuilding image: $IMAGE_FILE from $DEF_FILE"
    BUILD_FLAGS=(--force --fakeroot --ignore-fakeroot-command)
    "$EXEC_BIN" build "${BUILD_FLAGS[@]}" "$IMAGE_FILE" "$DEF_FILE"
  else
    echo "Using existing image: $IMAGE_FILE"
  fi
fi

echo "Started: $(date)  host=$(hostname)  ROOT=$ROOT  SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "CONFIG=$CONFIG  BASE_SEED=$BASE_SEED  NUM_SEEDS=$NUM_SEEDS  PARALLEL=$PARALLEL  USE_APPTAINER=$USE_APPTAINER  MUJOCO_GL=$MUJOCO_GL"
echo "Logs under $LOG_DIR"

# In Apptainer mode: bind ROOT -> /t-zero and use the container python (ignore
# host CONDA_ENV). On host: optional CONDA_ENV is handled by run_seeds.sh itself.
if [[ "$USE_APPTAINER" == "1" ]]; then
  "$EXEC_BIN" exec --nv \
    --bind "$ROOT:/t-zero" \
    --pwd /t-zero \
    --env MUJOCO_GL="$MUJOCO_GL" \
    --env WANDB_DIR=/t-zero/wandb \
    --env RUN_IN_CONTAINER=1 \
    --env RUN_ROOT=/t-zero \
    --env CONFIG="$CONFIG" \
    --env NUM_SEEDS="$NUM_SEEDS" \
    --env FIRST_SEED="$BASE_SEED" \
    --env PARALLEL="$PARALLEL" \
    --env "LOG_DIR=$LOG_DIR" \
    --env "EXTRA_OVERRIDE=${EXTRA_OVERRIDE:-}" \
    "${ENV_FILE_ARGS[@]}" \
    "$IMAGE_FILE" \
    bash scripts/run_seeds.sh
else
  CONFIG="$CONFIG" NUM_SEEDS="$NUM_SEEDS" FIRST_SEED="$BASE_SEED" PARALLEL="$PARALLEL" \
    LOG_DIR="$LOG_DIR" EXTRA_OVERRIDE="${EXTRA_OVERRIDE:-}" RUN_ROOT="$ROOT" \
    bash "$ROOT/scripts/run_seeds.sh"
fi

echo "All seed runs finished OK at $(date)"
