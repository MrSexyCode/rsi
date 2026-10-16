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

## 2026-10-16 — 1.1 Adapter interface (Phase 1 started; Phase 0 gate runs 0.7.2–0.7.5 still pending)

**Done**
- `core/adapter.py`: `TrustLevel`, `Cell`, `Task`, `Reward`, `ValidityReport`, `Twin`, `Probe`, the `Adapter` protocol, and the `UNDETERMINED` / `AMBIGUOUS` sentinels.
- Helpers:
  - `rng_for(*parts)`: hash-seeded, never the global RNG.
  - `rng_for_spec(spec)`: instantiate depends on the spec alone.
  - `make_task(adapter, cell, seed)`: the canonical draw.
  - `package_version`: adapter version = content hash of its package.
- Contract additions:
  - `task.meta["answer_text"]` and `Twin.expected` are final-answer strings that `verify` scores 1.0, so twin checks are domain-agnostic.
  - `meta["cell"]` holds the cell knobs.
- `core/registry.py`. `adapters/__init__.py` auto-imports every subpackage, and each registers itself, so core never imports an adapter (AST-checked in a test).
- `tests/test_adapter_contract.py` is parametrized over the registered adapters plus an in-test toy adapter. It checks:
  - determinism by seed
  - spec JSON round trip
  - validity, stripped prompts, and the gold answer scoring 1
  - twin expectations
  - probes within 5 pts + 3σ of chance

**Numbers**
- 7 contract tests pass (toy adapter).
- 1.2–1.4 (multihop) and 1.6 (progout) are being written by two agents in parallel.

## 2026-10-16 — 0.7.1 Static parity: PASS

**Done**
- Base Qwen3-8B on all 21 dev sets, mean@4 (T 0.6, top_p 0.95, top_k 20, 16k budget), direct vs Pi-vanilla.
- Shared vLLM DP8 server; identical prompt ids and per-sample seeds.
- Results: `outputs/evals/parity_base_{direct,pi_vanilla}/` and `docs/evals.md`.

**Numbers**
- All 21 sets are within 2 stderr; max |diff/combined-stderr| = 1.0. There are 0 invalid rollouts in 59k samples.

  | Set | Direct | Pi |
  |---|---|---|
  | omanic_heldout | 0.543 | 0.542 |
  | 2wiki | 0.799 | 0.801 |
  | hotpotqa | 0.748 | 0.748 |
  | musique | 0.614 | 0.621 |

- BABILong diffs range from −0.022 to +0.041. Truncation rates match (qa3_8k: 0.150 vs 0.144).
- The base-model Omanic number reproduces the playground's earlier eval (0.546).

**Open issues**
- RULER VT is saturated at 1.000 at 4k and 8k, so it is useless for discrimination. Either drop it, or move it to 16k–32k and/or more chains.
- Wall times (2,525 s direct vs 1,118 s Pi) are not comparable, because Pi ran second on a warm prefix cache.

## 2026-10-16 — 1.6 progout adapter stub (written by a subagent, reviewed)

**Done**
- `adapters/progout/`: generator, sandbox, adapter, reward. Trust level EXECUTABLE.
- **Grammar:** int and list ops, bounded `for` (range or list) and `if`/`else`, one final `print`.
  - Cells: statements {6, 9, 12} × loop depth {0, 1, 2} × list length {3, 5, 7} = 27.
  - Exact nesting depth and statement count by construction.
  - Values stay small by construction; dead or self-cancelling programs are filtered out in `sample_spec`.
- **Sandbox:**
  - AST whitelist of nodes, ops, int constants ≤ 1000 and 8 builtins called directly. Imports, attributes, strings, def/lambda, while, comprehensions and dunders are all rejected.
  - Execution: `python -I -S` subprocess with empty env, temp cwd, and rlimits set inside the child (CPU 2 s, AS 256 MB, FSIZE 0, NPROC 0), plus a 5 s wall timeout.
  - Results are cached in `task.meta`.
- **Twins:** renaming keeps the answer; constant perturbation changes it.
- **Probe:** last literal. chance = 1/(number of distinct int literals) when the output is one of them, else 0.

**Numbers**
- Tests: 51 pass, and the contract passes.
- 40 adversarial programs are rejected without being executed.
- The sandbox output equals in-process `exec` on 216 generated programs.
- `last_literal` is within 3.5 pts of chance on every cell, at 200 tasks/cell (chance ≈ 3–6%).

**Interface feedback acted on**
- Added `Twin.expect_valid`; the contract test no longer special-cases the "ablation" kind.
- Documented that instantiate's rng may be ignored, that leakage and answer-in-question rules are adapter-defined, and that EXECUTABLE adapters may execute in `sample_spec` / `instantiate` but must cache the results.

**Open issues**
- The probes-near-chance check has little power when chance is a few %.
- An execution timeout under heavy load could flip validity; there is no execution-status notion in the interface.

## 2026-10-16 — 1.2–1.4 multihop world compiler, probes, cells and pools (written by a subagent, reviewed)

**Done**
- `adapters/multihop/`:
  - names: syllable generator with a blocklist built from dwyl words, top first and last names (Census, baby-names) and GeoNames countries/cities
  - schema: 23 functional relations and 5 attributes; every relation has 5 fact templates, 2 noun-phrase and 2 verb-phrase question templates
  - world, dsl, executor (runs on the triples parsed back from the rendered text), distractors (4 types), render, twins, probes, adapter, reward
- **"Hops" means the longest dependency chain.**
  - chain: hops relations.
  - comparison: 2 × (hops − 1) relations plus 2 attribute facts.
  - intersection: 2 × (hops − 1) relations plus 2 join facts.
- **Decoy design keeps every probe at chance.** Every operator includes same-relation decoy chains. Comparison ranks are spread evenly, and intersection uses a 3×3 candidate grid. The answer type never appears mid-chain.
- **Cells and sets:**
  - The full grid has 135 cells.
  - Held-out eval grid: `evals/sets/multihop_grid`, 810 tasks, `scorer: multihop`.
  - Training pool: `/home/ubuntu/data/harness_rl/multihop_pool/{train,val}.parquet`, 20,000/500 rows, uniform over cells, byte-identical on rebuild.
  - Seed ranges are defined once in `dsl.py`: train `[0, 2^30)`, held-out `[2^30, 2^30 + 2^24)`.
- I added `ADAPTER_SETS` to `test_evals.py` (`multihop_grid` is built by the adapter, not by `evals/build.py`).

**Numbers**
- **Probe table:** 4,000 tasks per cell over the full grid. The max |acc − chance| per probe is 1.5–2.2 pts (largest at 2.9 σ over 810 checks), so the gate is passed: no cell is above 5 pts.
- Chance: chain 0.146, intersection 0.167, comparison 0.5.
- Prompt tokens: mean 665, max 1,368. 6.8 % of pool rows exceed 1,024 tokens, so the 1.7 smoke run needs `data.max_prompt_length` ≥ 1,400 (use 1,536).
- Tests: multihop plus contract tests pass in 28 s; the slow tests pass with `HARNESS_RL_SLOW=1`. The full suite passes.

**Deviations**
- The eval set is at `evals/sets/multihop_grid`, not `evals/synthetic_grid`, so the existing runner and loader work unchanged.
- Probes pick among answer-type candidates, so chance is 1/candidates (1/2 for comparison) as the plan defines it.
- High-level branch distractors are occasionally placed fewer times than asked (≈ 15 of 270 high tasks); every type is still present.

## 2026-10-16 — 1.5 calibration (running)

**Done**
- Wrote `adapters/multihop/calibration.py`, which builds three sets from held-out seed indices ≥ 1,000,000 (disjoint from training and from `multihop_grid`), stored in `outputs/calibration/sets/`:
  - `calib_grid`: 135 cells × 20 tasks
  - `calib_closedbook`: 300 tasks with the facts removed and no abstention offered ("give your best guess"), so the check isn't passed trivially by abstaining
  - `calib_twins`: 300 base tasks and their counterfactual twins
- Added `evals/run.py --sets-root`.
- Wrote `adapters/multihop/calibration_report.py`, which writes `docs/difficulty_map.{md,csv}` and computes the gate checks.

**Numbers** (base Qwen3-8B through Pi-vanilla; full map in `docs/difficulty_map.{md,csv}`)
- **Difficulty range: FAIL.** Cells span 0.61–1.00. The overall mean is 0.92, so no cell is ≤ 10 %.
  - Hardest cells: intersection, 6 hops, high distractors (0.61–0.78), and comparison, 6 hops, high (0.69–0.81).
- **Accuracy falls with hops, except for one anomaly.** Mean accuracy by hops:

  | Operator | 2 | 3 | 4 | 5 | 6 |
  |---|---|---|---|---|---|
  | intersection | 0.97 | 0.94 | 0.90 | 0.82 | 0.80 |
  | comparison | 0.99 | 0.97 | 0.96 | 0.94 | 0.85 |
  | chain | 0.92 | 0.98 | 0.95 | 0.93 | 0.88 |

  The anomaly is the 2-hop chain, which is low because of the template issue below.
- **Closed-book: PASS.** Accuracy 0.000 vs chance 0.270, with 1,200 samples, all of which answered.
- **Twin-consistency baseline: 0.946.** Base accuracy 0.927, twin 0.923, over 300 pairs × 4.
- Probe table: see 1.2–1.4 (CPU, 4,000 tasks per cell). All probes are within 2.2 pts of chance.

**Issues found**
- **Generator bug: loose fact templates.** For example, `studied_at` rendered as "X was trained at Y". The model reasonably reads it as a different relation and answers "cannot be determined". This is worst where two relations share types, i.e. exactly where the "other relation" distractors sit.
  - Abstentions on answerable tasks rise with hops (chain 25 → 144, intersection 47 → 222, out of 1,440 samples per hop level), so template ambiguity is part of the measured difficulty.
  - Fix: audit every template for unambiguous semantics, then bump the adapter version.
- **Question template redundancy:** "Which artifact is the artifact that is the flagship product of …".
- **74 / 21,600 grid samples were invalid** because the proxy↔vLLM HTTP keep-alive connection dropped (`ServerDisconnectedError`); these are eval-backend errors, not Pi. Fixed: `VLLMHTTPBackend` now retries connection-level failures (up to 3 times; same request and seed).
- **The plan's knob grid is too easy for base Qwen3-8B with thinking.** Reaching ≤ 10 % cells needs harder knobs than the plan's grid, which is a plan change and needs a decision.

## 2026-10-16 — Decision: replace the long 0.7.3 training-parity runs (user-approved plan change)

**Why**
- Token identity is established: identical prompt ids, response ids, logprobs and reward path in unit tests against `SingleTurnAgentLoop`, plus 59k-sample static parity.
- So Pi can change training only through invalid rollouts, edge cases under model drift, operational decay over long runs, or throughput. All of these are measured directly by per-rollout invariants.
- A 2-seed curve comparison is the weakest detector of them. Also, the only Omanic run so far was flat, so its curves might not move at all.

**New plan**
1. **Invariants:** per-rollout checks and Pi metrics in every Pi training run:
   - valid rate, calls per rollout
   - prompt ids equal the direct rendering
   - Pi overhead, process and disk counts
2. **Short A/B on Omanic:** 1 direct + 1 Pi seed, ~40 steps, from base Qwen3-8B, same seed. It covers 0.7.2 (logprob mismatch, Pi vs direct), 0.7.5 (step time), and a sanity check that the curves overlap.
3. **Soak and seed variance on multihop:** the 1.7 smoke run through Pi, as 2 seeds instead of 1, chunked at 50-step boundaries with evals between chunks. This gives 0.7.4 (seed variance) on the task that matters, plus long-run Pi stability.

**GPU order**
G1 re-calibration → Omanic A/B → multihop × 2 seeds.

## 2026-10-16 — Multihop adapter v2 (subagent, reviewed) and launch prep

**Done**
- **Templates:** each relation and attribute owns a set of key words. Every template uses only its own keys, and a test enforces it.
  - Worst v1 offenders: intersection "has as its X {y}" phrases (up to +9.5 pts excess error), "the hometown of" (+6.4), "spent their childhood in" (+5.1).
  - Renames: grew_up_in → lives_in, largest_employer → official_partner, founded_year → established_year.
- **Question wording:** relative-clause intersections; no "Which X is the X that…".
- **Knobs (approved plan change):**
  - hops ∈ {2, 3, 4, 5, 6, 8, 10, 12}.
  - New `very_high` distractor level: 3× high's near-miss count, plus more decoy chains and a 4×4 intersection grid.
  - The grid is now 288 cells.
- **Rebuilt sets:**
  - `multihop_grid`: hops {2, 4, 6, 8, 12} × 3 ops × 4 levels, shuffled, 30 per cell = 1,800 tasks.
  - Training pool: 20k/500 over 288 cells.
- **Bug fix:** validity now matches names as whole words.
- **Seeds:** `scripts/run_train.sh` (was `run_omanic.sh`; TASK=omanic|multihop) sets every verl seed to 42 + SEED: data shuffle, vLLM `rollout.seed`, Megatron, actor mini-batch shuffle.
- **Invariants:** `PiAgentLoop` writes one metrics row per rollout (valid, calls, prompt match, overhead, global step). For vanilla it asserts prompt ids equal the direct rendering on every rollout.
- **Monitor:** `scripts/pi_monitor.py` gives per-step invariant summaries; `--watch` adds Pi process, rollout-dir and disk counters.
- **Checkpoint export:** this verl writes actor weights in HF format (`actor/model/huggingface`, Megatron-Bridge), so `scripts/merge_ckpt.sh` just hard-links them. It is instant, and the export survives checkpoint rotation.
- **Chunked training:** `scripts/chunked.sh` runs train-to-boundary (resume=auto), then export, then `evals/run.py --launch` at each boundary, with the monitor alongside in Pi mode.

**Numbers**
- Probe table: 4,000 tasks per cell on 288 cells. The max |acc − chance| is 3.3 pts, so the gate is passed. Re-running the 8 cells with > 3σ checks at 10k tasks each gave ≤ 1.0 pt.
- Prompt tokens: pool mean 1,337, max 6,283. The multihop runs therefore use `data.max_prompt_length = 6400`.
- Tests: multihop + contract 47 passed (31 slow), `test_evals` 183, `test_pi_agent_loop` 5.

**Running**
- v2 calibration. The v1 results are kept as `outputs/evals/calib_*_v1` and `docs/difficulty_map_v1.*`.

## 2026-10-16 — 1.5 calibration on multihop v2: all calibration gates PASS

**Numbers** (base Qwen3-8B through Pi-vanilla; 288 cells × 20 tasks × 8 samples = 46,080 rollouts; map in `docs/difficulty_map.{md,csv}`)
- **Range: PASS.** Cells span 0.006–1.000; overall mean 0.744. Some cells are ≤ 10 % (e.g. comparison 12-hop very_high) and some are ≥ 90 % (most 2–4-hop cells).
- **Accuracy falls with hops: PASS**, monotone for every operator. Mean by hops 2, 3, 4, 5, 6, 8, 10, 12:

  | Operator | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 12 |
  |---|---|---|---|---|---|---|---|---|
  | chain | 0.99 | 0.98 | 0.95 | 0.91 | 0.89 | 0.72 | 0.60 | 0.42 |
  | comparison | 0.98 | 0.95 | 0.92 | 0.88 | 0.78 | 0.65 | 0.48 | 0.33 |
  | intersection | 0.97 | 0.94 | 0.86 | 0.81 | 0.71 | 0.52 | 0.34 | 0.24 |

  The v1 2-hop chain dip is gone (v2: 0.99).
- **Closed-book: PASS.** Accuracy 0.000 vs chance 0.254.
- **Twin-consistency baseline: 0.870.** Base accuracy 0.768, twin 0.754, from 922 correct base samples. (The v1 figure, 0.946, came from an easier grid.)
- Invalid rollouts: 0 of 47.9k.
- **The hardest cells are below chance**, e.g. comparison 12-hop very_high = 0.01 vs a chance of 0.5. The model abstains ("cannot be determined") on 70–87 % of samples and truncates on up to 39 %.
  - Spot check (comparison 12-hop, no distractors, reversed): the model claimed a link was missing that the context states verbatim. It mis-attributed fact 68 while tracking ~70 facts.
  - So these are genuine long-context failures, not generator bugs.
  - For RL, abstaining on an answerable task scores 0, like any wrong answer.

**Phase 1 gate status**
- Contract tests: pass for both adapters.
- Probes: within 5 pts on every cell (max 3.3).
- Difficulty range and monotone hops: pass.
- Closed-book: pass.
- Remaining: the smoke run (1.7), done as the 2-seed multihop run of the new parity plan.
