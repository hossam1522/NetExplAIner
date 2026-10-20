#!/usr/bin/env bash
# Campaign: models x strategies x tool modes x question sets.
# Every cell is one `netexplainer run --resume` writing results/<cell>/results.jsonl,
# so a cut (ssh, reboot, OOM) loses at most one question: just launch it again.
#
#   scripts/run_campaign.sh --tmux          # detached tmux session "campaign"
#   MODELS="phi4-q4 cogito-8b" TOOLS="none callable declared" scripts/run_campaign.sh
#
# Environment (defaults in brackets):
#   MODELS      model aliases or Ollama names              [phi4-q4]
#   STRATEGIES  direct decomp mixed cot react              [direct decomp mixed cot react]
#   TOOLS       none callable declared                     [none callable]
#   QSETS       core extended all                          [core]
#   GEN_URL     Ollama of the evaluated models             [http://localhost:11437]
#   JUDGE_URL   Ollama of the judge; empty = no judge now  [http://localhost:11438]
#               (generate first, then `netexplainer rejudge` per cell)
#   JUDGE_MODEL                                            [gemma3:27b]
#   OUT         results directory                          [results]
#   EXTRA       more `run` options, e.g. "--max-packets 512 --num-ctx 32768"
#   RETRIES     attempts per cell                          [3]
set -u
cd "$(dirname "$0")/.."

if [ "${1:-}" = "--tmux" ]; then
    shift
    exec tmux new-session -d -s "${TMUX_SESSION:-campaign}" \
        "bash $0 $* 2>&1 | tee -a ${OUT:-results}.campaign.log; read -r -p 'done, press enter'"
fi

MODELS=${MODELS:-phi4-q4}
STRATEGIES=${STRATEGIES:-direct decomp mixed cot react}
TOOLS=${TOOLS:-none callable}
QSETS=${QSETS:-core}
GEN_URL=${GEN_URL:-http://localhost:11437}
JUDGE_URL=${JUDGE_URL-http://localhost:11438}
JUDGE_MODEL=${JUDGE_MODEL:-gemma3:27b}
OUT=${OUT:-results}
EXTRA=${EXTRA:-}
RETRIES=${RETRIES:-3}
RUN=${RUN:-uv run python -m netexplainer}

if [ -n "$JUDGE_URL" ]; then
    JUDGE=(--judge-backend ollama --judge-model "$JUDGE_MODEL" --judge-base-url "$JUDGE_URL")
else
    JUDGE=(--judge-backend none)
fi

failed=0
for model in $MODELS; do
  for strategy in $STRATEGIES; do
    for tools in $TOOLS; do
      for qset in $QSETS; do
        cell="${model//[:\/]/-}__${strategy}__${tools}__${qset}"
        dir="$OUT/$cell"
        [ -e "$dir/.done" ] && { echo "skip $cell (done)"; continue; }
        case $tools in
          callable) toolflag=(--tools) ;;
          declared) toolflag=(--tools-declared-not-callable) ;;
          *) toolflag=() ;;
        esac
        mkdir -p "$dir"
        for attempt in $(seq "$RETRIES"); do
          echo "$(date -Is) $cell attempt $attempt"
          # shellcheck disable=SC2086
          if $RUN --log-file "$dir/run.log" run --models "$model" \
              --base-url "$GEN_URL" --strategy "$strategy" "${toolflag[@]}" \
              --question-set "$qset" --output-dir "$dir" --resume \
              "${JUDGE[@]}" $EXTRA; then
            touch "$dir/.done"; break
          fi
          sleep 30
        done
        [ -e "$dir/.done" ] || { echo "FAILED $cell"; failed=1; }
      done
    done
  done
done
echo "$(date -Is) campaign finished (failed=$failed)"
exit $failed
