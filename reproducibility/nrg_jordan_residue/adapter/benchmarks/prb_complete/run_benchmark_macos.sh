#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-pilot}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/ptdirac_complete_benchmark_matplotlib}"

JULIA_BIN="${JULIA:-$HOME/.juliaup/bin/julia}"
if [[ ! -x "$JULIA_BIN" ]]; then
  JULIA_BIN="$(command -v julia || true)"
fi
if [[ -z "$JULIA_BIN" ]]; then
  echo "Julia not found. Set JULIA=/path/to/julia."
  exit 2
fi

python3 benchmarks/prb_complete/scripts/run_complete_benchmark.py \
  --profile "$PROFILE" \
  --julia "$JULIA_BIN" \
  --resume
