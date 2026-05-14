#!/usr/bin/env bash
# Run the real-data experiment on all three fairlearn benchmarks.
# Usage:  bash experiments/run_all.sh             (defaults below)
#         bash experiments/run_all.sh --K 5 ...   (any extra flags forwarded)
#         FORCE=1 bash experiments/run_all.sh     (delete cached results first)

set -euo pipefail

DATASETS=(adult diabetes_hospital bank_marketing)
ROOT=$(cd "$(dirname "$0")/.." && pwd)

K=${K:-10}
LAMBDAS=${LAMBDAS:-"0,0.03,0.1,0.3,1,3,10,30,100"}
N_SEEDS=${N_SEEDS:-5}
MAX_SAMPLES=${MAX_SAMPLES:-5000}
FIT_N_SAMPLES=${FIT_N_SAMPLES:-1000}
DOWNSTREAM=${DOWNSTREAM:-logreg}

for ds in "${DATASETS[@]}"; do
    if [[ "${FORCE:-0}" == "1" ]]; then
        rm -rf "$ROOT/data/$ds"
        rm -rf "$ROOT/figures/$ds"
    fi
    echo "═══════════ $ds ═══════════"
    python "$ROOT/experiments/real_data.py" \
        --dataset "$ds" \
        --K "$K" \
        --lambdas "$LAMBDAS" \
        --n_seeds "$N_SEEDS" \
        --max_samples "$MAX_SAMPLES" \
        --fit_n_samples "$FIT_N_SAMPLES" \
        --downstream_model "$DOWNSTREAM" \
        "$@"
done

echo "All datasets done. Outputs: data/<ds>/  figures/<ds>/"
