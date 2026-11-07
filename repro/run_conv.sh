#!/usr/bin/env bash
# Run one LoCoMo conversation through the robust A-Mem harness.
#
#   ./repro/run_conv.sh 0                 # baseline (memory evolution ON)
#   ./repro/run_conv.sh 0 --no-me         # ablation (memory evolution OFF)
#   ./repro/run_conv.sh 0 --no-me --k 5 --temp 0.5
#
# Requires OPENAI_API_KEY (DeepSeek key) and OPENAI_BASE_URL, or a .env file
# exporting them; see REPRO.md.
set -euo pipefail

CONV="${1:?usage: run_conv.sh <conversation_index> [--no-me] [--k N] [--temp T]}"
shift || true

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${AMEM_MODEL:-deepseek-flash}"
K=10
TEMP=0.5
ABLATION=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-me) ABLATION="--ablate_memory_evolution"; shift ;;
    --k) K="$2"; shift 2 ;;
    --temp) TEMP="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

: "${OPENAI_API_KEY:?set OPENAI_API_KEY (DeepSeek API key)}"
: "${OPENAI_BASE_URL:=https://api.deepseek.com/v1}"

DATASET="$ROOT/repro/datasets/conv_${CONV}.json"
[[ -f "$DATASET" ]] || { echo "missing $DATASET — run: python repro/split_dataset.py" >&2; exit 2; }

TAG="$( [[ -n "$ABLATION" ]] && echo nome || echo full )"
OUTDIR="$ROOT/repro/out"
mkdir -p "$OUTDIR"

cd "$ROOT"
python test_advanced_robust.py \
  --dataset "$DATASET" \
  --backend openai \
  --model "$MODEL" \
  --retrieve_k "$K" \
  --temperature_c5 "$TEMP" \
  --output "$OUTDIR/conv${CONV}_${TAG}_${MODEL}_k${K}.json" \
  --seed 42 \
  $ABLATION 2>&1 | tee "$OUTDIR/conv${CONV}_${TAG}.log"
