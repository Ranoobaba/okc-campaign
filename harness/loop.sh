#!/usr/bin/env bash
# loop.sh: run the campaign loop unattended on a Mac. One headless `claude -p` per experiment, state on
# disk (CAMPAIGN.md, LEARNINGS.md, results.tsv, git), so a crash costs one experiment, not the night.
#
#   harness/loop.sh                       # until STOP file or --hours
#   harness/loop.sh --hours 8 --problem muse_bf16_proj
#   touch STOP                            # finishes the current experiment, then exits
#
# Keep the lid open or on external power with display: caffeinate does not stop lid-close sleep.
set -uo pipefail
cd "$(dirname "$0")/.."
HOURS=10; PROBLEM=""; MAX=200; SESSION="okcloop"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --hours) HOURS="$2"; shift 2 ;;
    --problem) PROBLEM="$2"; shift 2 ;;
    --max) MAX="$2"; shift 2 ;;
    --inner) INNER=1; shift ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done
export PATH="$HOME/.okc/bin:$HOME/fib-moe-loop/.venv/bin:$PATH"
if [[ -z "${INNER:-}" ]]; then
  for t in claude modal tmux caffeinate; do command -v $t >/dev/null || { echo "missing $t"; exit 1; }; done
  rm -f STOP; mkdir -p logs
  tmux kill-session -t $SESSION 2>/dev/null
  tmux new-session -d -s $SESSION "caffeinate -dimsu bash '$0' --inner --hours $HOURS --max $MAX ${PROBLEM:+--problem $PROBLEM} 2>&1 | tee -a logs/loop.log"
  echo "loop running in tmux session $SESSION (tmux attach -t $SESSION). Stop with: touch STOP"
  exit 0
fi
END=$(( $(date +%s) + HOURS*3600 ))
read -r -d '' PROMPT <<'P'
Read program.md (authoritative), then LEARNINGS.md, then the last 120 lines of CAMPAIGN.md and results.tsv.
You are one session of the campaign loop: run exactly ONE experiment, record it, exit.
FOCUS_PROBLEM_LINE
Steps: (1) python3 harness/board.py and note rank changes in CAMPAIGN.md. (2) Pick one idea from the queue,
by evidence and leverage. (3) Change one thing in problems/<P>/submission.py (copy the previous version to
problems/<P>/candidates/ first). (4) Run the local judge: ~/fib-moe-loop/.venv/bin/modal run harness/judge.py
--problem <P> --submission problems/<P>/submission.py (use --no-bench first if correctness is in doubt).
(5) Apply the keep/discard rule; if discarded, restore the previous submission.py. (6) Append one row to
results.tsv and a dated entry to CAMPAIGN.md; durable facts to LEARNINGS.md; git commit. (7) If the local
geomean beats our current board entry for <P> by more than its interval and a token is available
(python3 harness/okc_api.py me works), run okc_api.py check first, then submit, and record the job id and result.
Never bend a rule in program.md. Never fabricate a measurement. Be concise in the ledger.
P
n=0
while [[ ! -f STOP && $(date +%s) -lt $END && $n -lt $MAX ]]; do
  n=$((n+1)); stamp=$(date +%Y%m%d-%H%M%S)
  echo "=== experiment $n at $stamp ==="
  focus=""; [[ -n "$PROBLEM" ]] && focus="Focus problem: $PROBLEM."
  claude -p "${PROMPT/FOCUS_PROBLEM_LINE/$focus}" --dangerously-skip-permissions --max-turns 80 > "logs/exp_${stamp}.log" 2>&1
  echo "exit $? ; $(tail -1 results.tsv)"
  sleep 20
done
echo "loop ended (n=$n)"
