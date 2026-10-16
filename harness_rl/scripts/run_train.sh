#!/usr/bin/env bash
# DAPO through the unchanged playground Omanic launcher (run_omanic_dapo.sh), direct or through Pi.
#   TASK=omanic    the existing environment, unchanged (Phase 0 A/B)
#   TASK=multihop  same recipe on the multihop adapter's pool: only data files, reward function and prompt length differ
# Usage: TASK=omanic|multihop MODE=direct|pi [HARNESS=vanilla] [SEED=0] ./run_train.sh RUN_NAME [hydra overrides...]
#   MODEL_PATH defaults to base Qwen/Qwen3-8B (0.7 parity runs). SEED offsets every verl seed by the same amount
#   (42 + SEED): data shuffle, vLLM sampling (per replica: replica_rank + seed), Megatron init, actor mini-batch shuffle.
#   SEED=0 reproduces verl's defaults.
#   Pi mode writes ~5% rollout traces to harness_rl/outputs/traces/RUN_NAME.
# Outputs: the playground's outputs/{rollouts,val,tensorboard,checkpoints,metrics,logs}/RUN_NAME.
set -euo pipefail

RUN_NAME=${1:?run name}
shift
TASK=${TASK:-omanic}
MODE=${MODE:-pi}
HARNESS=${HARNESS:-vanilla}
SEED=${SEED:-0}
RSI=/home/ubuntu/rsi
export MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}

S=$((42 + SEED))
COMMON=(
    data.seed=$S
    actor_rollout_ref.rollout.seed=$S
    actor_rollout_ref.actor.megatron.seed=$S
    actor_rollout_ref.actor.data_loader_seed=$S
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

case $TASK in
    omanic) TASK_ARGS=() ;;
    multihop)
        POOL=${POOL:-/home/ubuntu/data/harness_rl/multihop_pool}
        TASK_ARGS=(
            data.train_files=$POOL/train.parquet
            data.val_files=$POOL/val.parquet
            data.max_prompt_length=${MAX_PROMPT_LENGTH_MH:-6400}
            reward.custom_reward_function.path=$RSI/harness_rl/adapters/multihop/reward.py
            reward.custom_reward_function.name=compute_score
            trainer.project_name=harness_rl_multihop
        )
        ;;
    *) echo "TASK must be omanic or multihop" >&2; exit 1 ;;
esac

exec /home/ubuntu/playground/run_omanic_dapo.sh "$RUN_NAME" "${COMMON[@]}" "${EXTRA[@]}" "${TASK_ARGS[@]}" "$@"
