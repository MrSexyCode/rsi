#!/usr/bin/env bash
# Train in chunks with an eval at every chunk boundary: train -> (checkpoint) -> merge to HF -> evals/run.py.
# Each chunk resumes from the latest checkpoint (trainer.resume_mode=auto: weights, optimizer, dataloader position);
# the LR is constant, so ending a chunk via total_training_steps changes nothing but where it stops. A chunk counts as
# done when its checkpoint exists (verl's teardown can print a spurious traceback, so exit codes are not trusted).
# In Pi mode the invariant monitor (pi_monitor.py --watch) runs alongside training.
#
# Usage: TASK=omanic|multihop MODE=direct|pi SEED=0 CHUNK=50 TOTAL=200 EVAL_SETS="..." ./chunked.sh RUN_NAME [hydra...]
#   EVAL_SETS: evals/run.py --sets arguments (default: multihop_grid dev:natural); EVAL_SAMPLES (default 4);
#   EVAL=0 skips the boundary evals. Logs: harness_rl/outputs/chunked/RUN_NAME.log
set -euo pipefail

RUN=${1:?run name}
shift
CHUNK=${CHUNK:-50}
TOTAL=${TOTAL:-200}
EVAL=${EVAL:-1}
EVAL_SETS=${EVAL_SETS:-"multihop_grid dev:natural"}
EVAL_SAMPLES=${EVAL_SAMPLES:-4}
RSI=/home/ubuntu/rsi
HR=$RSI/harness_rl
PY=/home/ubuntu/playground/verl/.venv/bin/python
CKPTS=/home/ubuntu/playground/outputs/checkpoints/$RUN
LOG=$HR/outputs/chunked/$RUN.log
mkdir -p "$(dirname "$LOG")"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

MON_PID=
if [ "${MODE:-pi}" = pi ]; then
    mkdir -p "$HR/outputs/traces/$RUN"
    (cd "$RSI" && exec "$PY" -m harness_rl.scripts.pi_monitor "$RUN" --watch >> "$HR/outputs/traces/$RUN/violations.log" 2>&1) &
    MON_PID=$!
fi
trap '[ -n "$MON_PID" ] && kill $MON_PID 2>/dev/null || true' EXIT

for END in $(seq "$CHUNK" "$CHUNK" "$TOTAL"); do
    if [ -d "$CKPTS/global_step_$END" ]; then
        log "chunk ending at step $END: checkpoint exists, skipping training"
    else
        log "chunk ending at step $END: training"
        # V1 also stops at total_epochs (epoch = step // (pool // batch); 78 steps on the 20k multihop pool) and the
        # launcher defaults to 1, so lift that cap and let total_training_steps decide; the dataloader wraps around.
        TOTAL_EPOCHS=${TOTAL_EPOCHS:-1000} SAVE_FREQ=$CHUNK TEST_FREQ=$CHUNK "$HR/scripts/run_train.sh" "$RUN" \
            trainer.total_training_steps=$END trainer.resume_mode=auto trainer.max_actor_ckpt_to_keep=2 \
            "$@" > /dev/null 2>&1 || true
        [ -d "$CKPTS/global_step_$END" ] || { log "no checkpoint at step $END; see outputs/logs/$RUN.log"; exit 1; }
        log "chunk ending at step $END: done"
    fi
    if [ "$EVAL" = 1 ]; then
        "$HR/scripts/merge_ckpt.sh" "$RUN" "$END" >> "$LOG" 2>&1
        log "eval at step $END: $EVAL_SETS"
        (cd "$RSI" && "$PY" -m harness_rl.evals.run --checkpoint "$HR/outputs/models/$RUN/step_$END" \
            --tokenizer Qwen/Qwen3-8B --launch --harness "${HARNESS:-vanilla}" --sets $EVAL_SETS \
            --samples "$EVAL_SAMPLES" --run-name "${RUN}_step$END" >> "$LOG" 2>&1)
        log "eval at step $END: done"
    fi
done
log "all chunks done"
