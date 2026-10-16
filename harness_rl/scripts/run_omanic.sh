#!/usr/bin/env bash
# Omanic DAPO through the unchanged playground launcher, direct or through Pi. Only the agent loop differs.
# Usage: MODE=direct|pi [HARNESS=vanilla] [SEED=0] ./run_omanic.sh RUN_NAME [extra hydra overrides...]
#   MODEL_PATH defaults to base Qwen/Qwen3-8B (0.7 parity runs). SEED sets data.seed (shuffle order) only;
#   TODO(0.7): find and set the rollout sampling seed too.
#   Pi mode writes ~5% rollout traces to harness_rl/outputs/traces/RUN_NAME.
# Outputs: the playground's outputs/{rollouts,val,tensorboard,checkpoints,metrics,logs}/RUN_NAME.
set -euo pipefail

RUN_NAME=${1:?run name}
shift
MODE=${MODE:-pi}
HARNESS=${HARNESS:-vanilla}
SEED=${SEED:-0}
RSI=/home/ubuntu/rsi
export MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}

COMMON=(
    data.seed=$SEED
    +ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=$RSI
)
case $MODE in
    direct) EXTRA=() ;;
    pi)
        EXTRA=(
            actor_rollout_ref.rollout.agent.default_agent_loop=pi_agent
            actor_rollout_ref.rollout.agent.agent_loop_config_path=$RSI/harness_rl/pi_loop/agent_loop.yaml
            +harness_rl.harness=$HARNESS
            +harness_rl.trace_dir=$RSI/harness_rl/outputs/traces/$RUN_NAME
            +harness_rl.trace_frac=0.05
            +harness_rl.seed=$SEED
            +harness_rl.checkpoint_id=$MODEL_PATH
        )
        ;;
    *) echo "MODE must be direct or pi" >&2; exit 1 ;;
esac

exec /home/ubuntu/playground/run_omanic_dapo.sh "$RUN_NAME" "${COMMON[@]}" "${EXTRA[@]}" "$@"
