# 0.1 Inventory

Date: 2026-10-16. Source tree: `/home/ubuntu/playground` (not a git repo; verl inside it is).

## Decisions taken at 0.1 (confirmed with the user)

| Question | Decision |
|---|---|
| Which environment is "the existing environment" | **Omanic DAPO** (`run_omanic_dapo.sh`) |
| Init model for 0.7 parity runs | **Base `Qwen/Qwen3-8B`** (HF snapshot `b968826d9c46dd6066d109eabc6255188de91218`) |
| Location of new code | **`/home/ubuntu/rsi/harness_rl`** (git repo at `/home/ubuntu/rsi`) |

Consequences worth flagging:

- **Omanic is closed-book.** The prompt holds a question and four options, no context passages.
  The plan calls the environment "in-context multi-hop"; for Omanic the "context" is empty.
  The Pi wrapper does not care, but the natural eval sets in 0.6 (MuSiQue, 2Wiki, HotpotQA,
  BABILong, RULER) are all in-context, so they measure transfer, not the training distribution.
- **The existing algorithm is DAPO, not plain GRPO**: GRPO advantages + dynamic sampling
  (drop all-same-reward groups and refill), clip-higher 0.2/0.28, token-mean loss, soft overlong
  penalty, no KL loss. "The existing GRPO config" in 0.7.3 is read as this DAPO config.
- **No existing direct run starts from base Qwen3-8B on Omanic.** The one Omanic run
  (`omanic_dapo_from_musique`) starts from the GSM8K→MuSiQue step-77 model and was killed at step 13.
  0.7.3 therefore needs fresh direct runs (2 seeds) as well as the Pi runs.

## Data format

Preprocessing: `playground/omanic/preprocess.py`, sources `playground/data/omanic/{OmanicSynth,OmanicBench}.jsonl`
(HF `li-lab/Omanic`). Output parquet in `~/data/omanic/`:

| File | Rows | Source | Options | Use |
|---|---|---|---|---|
| `train.parquet` | 10,296 | OmanicSynth | shuffled per question (rng seed 0) | training |
| `test.parquet` | 967 | OmanicBench | original order (B correct 42.9%) | paper-comparable |
| `test_shuffled.parquet` | 967 | OmanicBench | shuffled (seed 1) | **held-out split** |
| `test_single_hop.parquet` | 3,868 | OmanicBench sub-questions | original | step-wise diagnosis |
| `val.parquet` | 300 (299 after filter) | stratified subset of `test_shuffled` by (motif, has_math_hop), `random_state=0` | shuffled | in-training validation |

Row schema (verl RLHF parquet):

```python
{
  "data_source": "omanic",
  "prompt": [{"role": "user", "content": <INSTRUCTION>}],          # single user message, no system message
  "ability": "multihop_mc",
  "reward_model": {"style": "rule", "ground_truth": '{"label": "B", "answer": "54 years"}'},
  "extra_info": {"split", "index", "id", "motif", "has_math_hop", "original_label"},
}
```

## Prompt template

Single user message, no system prompt:

```
Answer the following multiple-choice question. Answering it requires several reasoning steps.

Question: {question}

Options:
A. ...
B. ...
C. ...
D. ...

Reason step by step. Then end your response with exactly this line:
Answer: <the letter of the correct option>
```

(Single-hop variant replaces the intro with "Answer the following multiple-choice question.")

## Answer extraction and reward

`playground/omanic/reward.py::compute_score(data_source, solution_str, ground_truth, extra_info)`:

- Parse only the text after the **last `</think>`**.
- Scan lines bottom-up; the **last** line matching `^\s*\**\s*(?:final\s+)?answer\s*\**\s*:\s*\**\s*(.*?)\s*$` (case-insensitive) is the answer line.
- Hedges (`B or C`, `B, C`, `B/C`, `B and C`) → `None`. A lone capital letter, optionally wrapped (`(B)`, `**B**`, `\boxed{B}`, `B. text`) → that letter. Lower-case letters are not accepted.
- Returns `{"score": 1/0, "acc": 1/0, "format_ok": pred is not None, "has_math_hop": ...}`.
- With DAPO's reward manager the final reward is `score + overlong_penalty`, where the overlong penalty is linear in the last 1,024 tokens of the 8,192 budget (`penalty_factor=1.0`), so reward ∈ [-1, 1].

The reward function decodes the response with `tokenizer.decode(response_ids, skip_special_tokens=True)`
(`verl/experimental/reward_loop/reward_manager/dapo.py:65`). `<think>`/`</think>` are **not** special tokens in
Qwen3 (ids 151667/151668, `special=False`) so they survive decoding and the `</think>` split works; `<|im_end|>` is stripped.

## verl

- Checkout: `/home/ubuntu/playground/verl`, commit `1d367a4f` (2026-10-10, `git describe` = `v0.8.0-513-g1d367a4f`), `verl.__version__ = 0.10.0.dev`. One untracked dir (`demo/`).
- Venv: `verl/.venv` (Python 3.12) via `uv run --frozen --all-packages --extra vllm --extra megatron`. vLLM 0.29.0, torch 2.13.0+cu130, transformers 5.12.1.
- Trainer: `python -m verl.trainer.main_ppo` → `TaskRunnerV1` (the V1 trainer, `verl/trainer/ppo/v1/trainer_base.py`). DAPO dynamic sampling is implemented by the V1 ReplayBuffer; `max_num_gen_batches=4` is **ignored** (warning in the log); `max_inflight_gen_batches` is the knob.
- Rollout backend: **vLLM**, async server mode (`vLLMHttpServer` Ray actors), TP 2, `gpu_memory_utilization=0.6`, 8 agent-loop workers.
- Actor: Megatron TP2 × PP2 × DP2, full recompute, param + optimizer offload; ref model param offload.

### Launch chain

`playground/run_omanic_dapo.sh` → `verl/examples/grpo_trainer/run_qwen3_8b_megatron.sh` (sets the arrays below) → `uv run ... python3 -m verl.trainer.main_ppo`.

Effective config of the Omanic run (from `outputs/logs/omanic_dapo_from_musique.log`):

| Key | Value |
|---|---|
| `algorithm.adv_estimator` | grpo |
| `algorithm.filter_groups` | enable=True, metric=reward |
| `data.train_batch_size` / `ppo_mini_batch_size` | 256 / 64 |
| `data.max_prompt_length` / `max_response_length` | 1024 / 8192 |
| `data.apply_chat_template_kwargs.enable_thinking` | True |
| `rollout.n` | 5 |
| `rollout.temperature` / `top_p` / `top_k` (train) | 1.0 / 1.0 / -1 |
| `rollout.val_kwargs` | greedy (temperature 0, n=1, do_sample=False) |
| `rollout.calculate_log_probs` | True |
| `rollout.agent.default_agent_loop` | `single_turn_agent` |
| `rollout.agent.agent_loop_config_path` | None |
| actor `optim.lr` | 1e-6 |
| `clip_ratio_low/high/c` | 0.2 / 0.28 / 10.0 |
| `loss_agg_mode` | token-mean |
| `use_kl_loss` | False |
| `reward.reward_manager.name` | dapo (overlong buffer 1024, factor 1.0) |
| `ppo_max_token_len_per_gpu` | 24576 |
| `data.seed` | 42 |
| `trainer.total_epochs` | 1 (10,296 / 256 ≈ 40 steps) |

## Agent-loop API (installed verl)

File: `verl/experimental/agent_loop/agent_loop.py`. Names confirmed:

- `AgentLoopBase(ABC)`: constructor `(trainer_config: DictConfigWrap, server_manager: LLMServerClient, tokenizer, processor, dataset_cls, data_config: DictConfigWrap, hf_model_type=None, **kwargs)`; abstract `async def run(self, sampling_params: dict, **kwargs) -> AgentLoopOutput`. `kwargs` are the dataset row's non-tensor fields (`raw_prompt`, `data_source`, `reward_model`, `extra_info`, `index`, ...).
- `AgentLoopOutput(BaseModel)`: `prompt_ids`, `response_ids`, `response_mask`, `response_logprobs`, `routed_experts`, `multi_modal_data`, `reward_score`, `num_turns`, `metrics: AgentLoopMetrics`, `extra_fields`, `mm_processor_*`.
- Registration: `@register("name")` decorator, or YAML at `rollout.agent.agent_loop_config_path` with entries `{name, _target_}`. Selection per row via a `agent_name` column, else `rollout.agent.default_agent_loop`.
- Token-in/token-out generation: `self.server_manager.generate(request_id=..., prompt_ids=..., sampling_params=..., ...) -> TokenOutput` (`verl/workers/rollout/llm_server.py`, class `LLMServerClient`). `TokenOutput` (`verl/workers/rollout/replica.py`): `token_ids`, `log_probs`, `routed_experts`, `stop_reason` (`completed`/`aborted`/None), `num_preempted`, `extra_fields`. If `max_tokens` is absent it is pinned to `rollout.response_length`; generation resumes after preemption and concatenates.
- `sampling_params` built in `AgentLoopWorker.generate_sequences`: `temperature, top_p, top_k, repetition_penalty=1.0, logprobs=calculate_log_probs` (val_kwargs when validating).
- Reward: if the loop does not set `reward_score`, the reward loop computes it from the decoded response (path used today).
- Built-in tracing: `rollout.trace.*` (`rollout_trace_op`), not needed for our JSONL traces.

### Direct path today (`SingleTurnAgentLoop`)

1. `prompt_ids = apply_chat_template(raw_prompt, tokenize=True, add_generation_prompt=True, enable_thinking=True)` via `QwenContinuousTokenBuilder` (model type `qwen3`).
2. `server_manager.generate(prompt_ids, sampling_params)`.
3. `response_ids = output.token_ids` unchanged (CT `merge_assistant_tokens` is a plain concatenation), `response_mask` all ones, `response_logprobs = output.log_probs`, all truncated to `response_length`.
4. `num_turns = 2`.

`PiAgentLoop` must produce byte-identical `prompt_ids` for the same messages; the only difference allowed is who sends the request.

## Qwen3 chat template and thinking

- Tokenizer `Qwen2Tokenizer`, HF `Qwen/Qwen3-8B`. Special: `<|im_end|>`=151645, `<|endoftext|>`=151643. Not special: `<think>`=151667, `</think>`=151668, `<tool_call>`=151657.
- `enable_thinking=True` generation prompt: `<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n` — the model writes `<think>\n...\n</think>\n\n{answer}<|im_end|>` itself, so the thinking tokens are part of `response_ids` and are trained on.
- `enable_thinking=False` appends `<think>\n\n</think>\n\n` to the prompt.
- Earlier assistant turns are rendered **without** reasoning (both `<think>` inside `content` and `reasoning_content` are dropped for turns before the last user message). Irrelevant while one call per rollout holds.
- `generation_config.json`: temperature 0.6, top_p 0.95, top_k 20, eos `[151645, 151643]`. verl passes explicit sampling params, so training uses T=1.0, top_p=1.0.

## Baseline numbers from existing runs and evals

**Base Qwen3-8B, Omanic** (`outputs/eval/omanic/report_base_vs_musique.txt`; thinking on, T=0.6, top_p 0.95, top_k 20, 16k new tokens, 4 samples/question):

| Set | Base Qwen3-8B | GSM8K→MuSiQue step 77 |
|---|---|---|
| OmanicBench, original order (n=967) | 0.538 | 0.538 |
| OmanicBench, shuffled (n=967) | **0.546** | 0.531 |
| Single-hop sub-questions (n=3,868) | 0.754 | 0.754 |
| Format OK / truncated | 0.997 / 0.001 | 0.998 / 0.002 |
| Mean response tokens | 3,032 | 2,936 |

**Omanic DAPO run** `omanic_dapo_from_musique` (init GSM8K→MuSiQue step 77, killed at step 13 — the log ends there):

| Step | val acc (greedy, n=299) | train score | response len | gen s | step s |
|---|---|---|---|---|---|
| 0 | 0.505 | — | — | — | — |
| 5 | 0.498 | 0.442 | 3,374 | 268 | 426 |
| 10 | 0.495 | 0.444 | 3,354 | 256 | 575 (ckpt) |
| 13 | — | 0.431 | 3,098 | 261 | 409 |

Flat: no learning signal in 13 steps. Step time ≈ 410–475 s, generation ≈ 60–65% of it.

**MuSiQue GRPO** (for reference; init GSM8K step 58, 77 steps): val reward 0.606 → 0.701, EM 0.484 → 0.574, ≈ 430 s/step, gen ≈ 69 s.

## Other things found

- Real `~/.pi` exists with a managed Pi **1.2.0** install (`@earendil-works/pi-coding-agent`; the old `@mariozechner/pi-coding-agent` name stops at 0.73.1). harness_rl uses its own pinned copy at `harness_rl/pi_runtime/` (npm lockfile, `--ignore-scripts`) and never the one in `~/.pi`.
- Node v22.22.1 (Pi requires ≥ 22.19). Passwordless sudo is available; unprivileged user namespaces are blocked (`apparmor_restrict_unprivileged_userns=1`), which matters for network isolation in 0.2/0.4.
- Machine: 8× H100 80GB, 208 cores, 1.8 TB RAM.
- `playground/outputs/rollouts/<run>/<step>.jsonl` dump schema: `input, output, gts, score, step, uid` (decoded text only, no token ids).
