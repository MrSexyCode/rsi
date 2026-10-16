# harness_rl log

## 2026-10-16 — 0.1 Inventory

**Done**
- Inventoried `/home/ubuntu/playground` into `docs/inventory.md`.
- Confirmed with the user:
  - The existing environment is **Omanic DAPO**.
  - Parity runs start from **base Qwen3-8B**.
  - New code lives in `/home/ubuntu/rsi/harness_rl` (git repo at `/home/ubuntu/rsi`).
- Created the package skeleton.
- Added `tests/test_inventory.py` (6 tests). It pins the Qwen3 token ids and template, `</think>` surviving reward decoding, the Omanic reward contract, the presence of the parquet data, and the pinned Pi version.

**Numbers**

Base Qwen3-8B on OmanicBench (T=0.6, 4 samples):

| Variant | Accuracy |
|---|---|
| Shuffled options | 0.546 |
| Original order | 0.538 |
| Single-hop | 0.754 |

The existing Omanic DAPO run (init from the MuSiQue checkpoint) was flat at val ≈ 0.50 over 13 steps, at ≈ 425 s/step with generation ≈ 270 s.

**Open issues**
- Omanic is closed-book: the plan's "in-context" framing does not apply to it.
- The "GRPO baseline" is actually DAPO.
- There is no existing base-init Omanic run, so 0.7.3 needs fresh direct runs.
- verl V1 ignores `filter_groups.max_num_gen_batches`.

## 2026-10-16 — 0.2 Pi spike

**Done**
- Pinned `@earendil-works/pi-coding-agent@1.2.0` in `harness_rl/pi_runtime/` (lockfile).
- Answered all six spike questions with scripts in `harness_rl/spike/`.
- Added `tests/test_pi_spike.py` (14 tests against the real binary and a logging stub).
- Wrote the full report in `docs/pi_spike.md`.

**Numbers**
- With the lockdown flags, 0 of 14 planted leak markers reach the request; with default flags, 7 leak.
- Under lockdown Pi makes exactly one `connect()`, to the endpoint.
- Median direct rollout time is 101 s (vLLM TP2, 320 in flight, base Qwen3-8B, Omanic, T=1.0).
- Pi overhead per rollout at 1,280 in flight (one training step's worth), proxy-like stream:
  - JSON subprocess: median 8.3 s, p99 12.4 s, which is 8.2 % of the median rollout. Passes the 10 % rule, narrowly.
  - SDK gateway: median 0.36 s, p99 1.4 s.
- At 64 in flight: JSON 0.93 s, RPC 0.40 s, SDK 0.10 s. Full table in `docs/pi_spike.md` §6.

**Open issues**
- `baseUrl` is not env-interpolated, so each rollout gets a per-rollout HOME with a literal URL, plus an `X-Rollout-Id` header as a cross-check.
- The stdin prompt is whitespace-stripped by Pi.
- The network is not OS-isolated (no unprivileged netns on this host); this is acceptable while tools are off and must be fixed before Phase 4.
- Pi 1.3.0 came out on 2026-10-15; we stay on 1.2.0 until a deliberate bump.
- **Decided at the STOP (2026-10-16, user): JSON subprocess.** Revisit only if 0.7.5 shows Pi overhead > 10 % of step time; the SDK gateway (`spike/sdk_gateway.mjs`) is the fallback.
- Parity subtlety for 0.4: on truncated responses (no `</think>`), the direct reward parses the whole text, thinking included. Pi's `content` would be empty. So `PiAgentLoop` must score `decode(response_ids)` exactly as the direct path does, not Pi's content.

## 2026-10-16 — 0.5 Harness packages v0 (written ahead of the 0.2 review)

**Done**
- Wrote `harness/vanilla/` and `harness/pi-default/`. Each package holds:
  - `harness.json`: manifest
  - `settings.json`: thinking off; retry, compaction, cache warming and telemetry off; project trust never; `httpIdleTimeoutMs` 1 h
  - `provider.json`: everything except `baseUrl`
  - `extension.ts`: `setActiveTools([])`, plus system-prompt control as the manifest says
- **vanilla** sends no system message, because the Omanic environment has none, so the rendered prompt equals the direct path's.
- **pi-default** keeps Pi's default prompt with tools off (it renders as `<tools>(none)`) and pins the `<cwd>` section to `/workspace`, so the prompt is identical across rollouts.
- Added `core/versioning.py`: `content_hash` = sha256 over the sorted relative paths, lengths and bytes, skipping caches.
- Added `pi_loop/runner.py`:
  - `HarnessPackage.load`
  - `materialize` (per-rollout HOME with a literal per-rollout `baseUrl`)
  - `JsonSubprocessRunner.run(prompt, rollout_id, base_url) -> PiResult` (ok = final `stopReason` in {stop, length})
- Added `tests/test_harness.py` (12 tests).

**Numbers**
- Harness versions: vanilla `ffaf8d9bca6f…`, pi-default `be649fde6516…`.

**Open issues**
- Pi's default HTTP idle timeout is 300 s, while the median rollout is 101 s and the max observed was 181 s. It is raised to 1 h in the settings, and the proxy will also send SSE keepalives.

## 2026-10-16 — 0.3 Model proxy (written by a subagent, reviewed)

**Done**
- `proxy/` (`server.py`, `backends.py`, `render.py`, `parse.py`, `__main__`) and `core/trace.py`.
- Rendering uses verl's own continuous-token builder.
- Backends: `VerlBackend`, `FakeBackend` and `VLLMHTTPBackend`; the last is tested only against a stub so far.
- SSE responses include keepalives and error events.
- Records: lossless split `reasoning_raw + content_raw == decoded text`.
- Sampling comes from the registered rollout, not from Pi's request; what Pi asked for is logged.

**Numbers**
- 33 tests (`test_proxy.py` and `test_proxy_pi.py`, the latter against real Pi). They cover:
  - render equals both `apply_chat_template` and verl's builder
  - reconstruction is exact
  - streaming chunks reassemble to the non-streaming result
  - 200 concurrent rollouts
- With Pi's idle timeout set to 3 s, a 5 s generation succeeds with keepalives and fails without them.

**Open issues**
- Tool calls are parsed only when the request carries `tools`.
- Tokenization runs in one executor thread, because HF fast tokenizers are not thread-safe.
- `VLLMHTTPBackend` needs a real-GPU smoke test in 0.6.

## 2026-10-16 — 0.4 PiAgentLoop (unit-level; real-verl smoke pending)

**Done**
- `pi_loop/agent_loop.py`: `@register("pi_agent")`, registered through `agent_loop.yaml`.
  - One proxy and one runner per worker process.
  - Exactly one call per rollout is enforced. A violation is recorded as `pi_valid=0` with the reason, and the response becomes a single masked `<|im_end|>`: reward 0 through the unchanged reward path, no gradient.
  - Traces: a deterministic ≈5% sample by rollout-id hash, written to `+harness_rl.trace_dir`.
- `scripts/run_omanic.sh`: `MODE=direct|pi` on top of the unchanged `run_omanic_dapo.sh`, with base Qwen3-8B.

**Numbers**
- `tests/test_pi_agent_loop.py` runs real Pi and the real proxy against verl's `SingleTurnAgentLoop`, both on one deterministic fake engine. With the vanilla harness, `prompt_ids`, `response_ids`, `response_mask`, logprobs and reward are **identical**, for both normal and truncated responses.
- pi-default = Pi's system block + exactly the direct prompt.
- Full suite: 69 passed.

**Open issues**
- Global step is not visible inside `run()`, so traces carry a timestamp and the dataset index instead.
- Pi-specific metrics (valid rate, overhead) are in `extra_fields` but are not yet aggregated into the trainer's logger.

## 2026-10-16 — 0.6 Eval suite, data half (written by a subagent, reviewed)

**Done**
- 21 frozen sets in `evals/sets/<name>/{data.jsonl,manifest.json}`, all with seed 0:

  | Set | Size | Notes |
  |---|---|---|
  | omanic_heldout | 967 | all dev |
  | musique_ans_dev | 500 | |
  | 2wiki_dev | 500 | |
  | hotpotqa_distractor_dev | 500 | |
  | babilong_qa1–5 × {0k, 4k, 8k} | 100 each | 15 sets |
  | ruler_vt_{4k,8k} | 200 each | generated from RULER @ c3f5e3b4 |

- Natural sets are split 80/20 dev/final, stratified. `load_set` refuses `final` without `allow_final` and refuses a set whose hash has drifted from its manifest.
- Versioned prompt adapters. The answer line is shared ("Reason step by step. Then end your response with exactly this line: Answer: …") and every prompt is pre-stripped.
- Scorers:
  - QA: SQuAD F1 (primary) and EM, with the official yes/no rule.
  - Omanic: the existing reward, imported by path.
  - BABILong: the official `compare_answers`.
  - RULER VT: `string_match_all` recall.
- `python -m harness_rl.evals.build [--check]` rebuilds the sets or verifies them byte-identical.

**Numbers**
- Mean / max prompt tokens (chat-rendered):

  | Set | Mean | Max |
  |---|---|---|
  | MuSiQue | 2,645 | 4,773 |
  | 2Wiki | 1,094 | 4,452 |
  | HotpotQA | 1,496 | 2,852 |
  | BABILong 8k | ≈ 7.6k | 8,616 |
  | RULER 8k | 8,176 | 8,186 |

- Tests: 204 pass with `HARNESS_RL_SLOW=1`, including byte-identical rebuilds of every set.

**Open issues (for review)**
- The answer line says "entity, date, or number", but 45/500 2Wiki and 31/500 HotpotQA golds are yes/no. Proposal: add "or yes/no" and rebuild.
- F1 vs EM as the primary QA metric.
- RULER deviations: no answer prefill, because of thinking mode, and a per-sample haystack fit.
- `evals/run.py` is still to be written. It needs the vLLM backend smoke-tested on a GPU.

## 2026-10-16 — 0.4 smoke run in real verl (`hrl_smoke_pi2`)

**Done**
- Ran 2 DAPO steps through `PiAgentLoop` on 8×H100: base Qwen3-8B, batch 16×5, 2,048-token responses, vanilla harness.
- The first attempt (`hrl_smoke_pi`) generated fine, then crashed in the V1 trainer's metrics. verl's agent loops must carry the engine's `min/max_global_steps` tags, and the proxy dropped them: `_json_safe` threw away numpy scalars. Fixed in both places, and the parity test now asserts the tags.

**Numbers**
- Invalid rollouts: **0**. Every rollout made exactly 1 model call.
- Step times 83 s and 52 s (generation 43 s and 39 s).
- Trace records: 24, complete. Example: 133 prompt / 479 response tokens with 479 logprobs, Pi overhead 0.58 s.
- Rollout-vs-actor logprob mismatch at step 1, Pi path: `rollout_probs_diff_mean` 0.0057, `k3_kl` 0.00069, `log_ppl_abs_diff` 0.00091.
- The existing direct Omanic run (different init) shows 0.0052–0.0053, 0.00074–0.00076 and 0.00095–0.00100: the same scale.
- The formal 0.7.2 comparison needs matched runs.

**Open issues**
- Pi metrics (valid rate, overhead) still go only to traces and `extra_fields`, not to the trainer's logger.

## 2026-10-16 — 0.6 Eval runner (`evals/run.py`)

**Done**
- `python -m harness_rl.evals.run --checkpoint --harness --sets --samples 4 [--direct] [--final]`
  - Engine: `--vllm-url URL` or `--launch` (vLLM data-parallel over `--gpus`).
  - Sampling: T 0.6, top_p 0.95, top_k 20, 16,384 new tokens, with a per-(set, task, sample) seed.
- The Pi path and `--direct` share the same proxy (rendering), the same engine and the same scorer, which reads the verl-identical `decode(response_ids, skip_special_tokens=True)`. `--direct` posts the same single user message to the proxy without Pi.
- Output: `outputs/evals/<run>/{results.jsonl,summary.json}`, and one row per set in `docs/evals.md`.
- Added `tests/test_eval_prompt_parity.py`: on the first 2 tasks of all 21 sets, Pi-vanilla and direct prompt ids are identical to each other and to `apply_chat_template`.

**Numbers**
- Smoke run (6 sets × 4 tasks × 2 samples): 0 invalid in both modes.
- **Item-level outputs differ despite identical prompt ids and seeds.** vLLM sampling isn't batch-invariant, so parity can only be statistical.
- BABILong qa2_4k full dev (80 × 4): direct 0.669 ± 0.041 vs Pi 0.672 ± 0.039, 0 invalid. The 4-item smoke gap (0.75 vs 0.125) was noise.

**Running**
- 0.7.1 static parity: base Qwen3-8B, all dev sets, ×4, direct then Pi-vanilla.
