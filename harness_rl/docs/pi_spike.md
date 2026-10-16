# 0.2 Pi spike

Date: 2026-10-16.

**Pinned version: `@earendil-works/pi-coding-agent@1.2.0`**, installed at `harness_rl/pi_runtime/` from a committed
`package.json` + `package-lock.json` (`npm install --ignore-scripts`, Node 22.22.1). It is the same version as the
managed install in `~/.pi`, which harness_rl never uses.

- The package was renamed: the old `@mariozechner/pi-coding-agent` stops at 0.73.1 (deprecated).
- `1.3.0` was published on 2026-10-15, and https://pi.dev/docs/latest tracks it. All findings below were checked against the 1.2.0 binary, and the docs shipped inside the 1.2.0 package match. Upgrading is a deliberate bump that must rerun `tests/test_pi_spike.py`.

Every claim below comes from a script, not from the docs alone:

- Scripts: `harness_rl/spike/` — `stub_server.py`, `pi_iso.sh`, `sdk_gateway.mjs`, `bench_overhead.py`.
- Regression tests: `harness_rl/tests/test_pi_spike.py` (14 tests, all pass). They run the real Pi binary against a logging stub endpoint.

## 1. OpenAI-compatible endpoint with a per-rollout URL

Use a custom provider in `<agent-dir>/models.json`, where `<agent-dir>` is `$HOME/.pi/agent` (or `$PI_CODING_AGENT_DIR`):

```json
{"providers": {"rlproxy": {
  "baseUrl": "http://127.0.0.1:<port>/v1",
  "api": "openai-completions",
  "apiKey": "${ROLLOUT_ID}",
  "headers": {"X-Rollout-Id": "${ROLLOUT_ID}"},
  "compat": {"supportsDeveloperRole": false, "supportsStore": false},
  "models": [{"id": "qwen3-8b", "reasoning": true, "contextWindow": 40960, "maxTokens": 8192}]}}}
```

Then run `pi --model rlproxy/qwen3-8b ...`.

- **`baseUrl` is not env-interpolated.** `"${PROXY_BASE_URL}"` fails with `errorMessage: "Invalid URL"`. `apiKey` and `headers` do support `$VAR` / `${VAR}`.
- **Per-rollout routing options:**
  - Per-rollout HOME: write a per-rollout `models.json` with the literal URL `http://proxy/r/{rollout_id}/v1`.
  - Fixed URL plus header: route on `X-Rollout-Id` (or the bearer token), set from the `ROLLOUT_ID` env var. Verified: the header arrives as sent.
- **Proposal for 0.3/0.4:** the proxy serves both `/r/{rollout_id}/v1/chat/completions` (as the plan says) and accepts `X-Rollout-Id`. `PiAgentLoop` writes the literal per-rollout URL into the per-rollout HOME it already creates, which costs nothing. The header is a cross-check: the proxy rejects a request whose path id and header id disagree.
- **A missing `ROLLOUT_ID` fails loudly.** The key resolves to empty, Pi exits non-zero with "No API key found", and no request is sent.
- **JSON mode exits 0 even when the provider call fails.** The loop must read `stopReason` / `errorMessage` from the final assistant `message_end`, never the exit code alone.

## 2. Zero active tools

Three mechanisms work, and all of them remove the `tools` field from the request:

- `--no-tools` (`-nt`) disables built-in, extension and MCP tools.
- `pi.setActiveTools([])` in an extension's `session_start` handler.
- SDK `noTools: "all"`.

Without any of these, Pi sends `read, bash, edit, write`. We use `-nt` and also `-ne --no-mcp`, because the built-in MCP, codemode and tool-search extensions otherwise load.

## 3. System prompt control and isolation

- **`--system-prompt X` replaces only the preamble.** Pi still appends a `<cwd>…</cwd>` section, so it is not usable for exact control.
- **Exact control: a `before_agent_start` handler returning `{systemPrompt: text}`.** The text is sent byte-exact as the leading message.
- **`systemPrompt: ""` makes Pi send no system message at all.** This is what the existing Omanic environment does (a single user message, no system message), so `harness/vanilla` uses it.
- **Role:** with `reasoning: true`, Pi sends the prompt as role `developer` unless `compat.supportsDeveloperRole: false`, which gives `system`. We set it to false.

**Lockdown flags** used for every rollout:

```
-p --mode json --no-session -ne -e <harness>/extension.ts -ns -np --no-themes -nc -na --no-mcp -nt --offline
```

The environment is cleared (`env -i`) except for these variables:

```
PATH  HOME=<per-harness agent home>  PI_OFFLINE=1  PI_TELEMETRY=0  PI_SKIP_VERSION_CHECK=1  ROLLOUT_ID
```

**Leak test.** Markers were planted in every location Pi reads:

- In the parent directory and in cwd: `AGENTS.md`, `CLAUDE.md`, `.pi/SYSTEM.md`, `.pi/APPEND_SYSTEM.md`, `.pi/extensions/*.ts`, `.pi/skills/*/SKILL.md` and `.pi/settings.json`.
- In the agent dir: `SYSTEM.md`, `APPEND_SYSTEM.md`, `AGENTS.md`, `extensions/*.ts` and `skills/*`.

| Run | Markers in the request |
|---|---|
| Default flags | 7 leak: agent-dir SYSTEM/APPEND/AGENTS.md, agent-dir extension, agent-dir skill, AGENTS.md from cwd **and from the parent dir**. Project settings were also ignored by default (print mode does not trust projects). |
| Lockdown flags | **0 leak.** The request is exactly `[{"role":"user","content":[...prompt...]}]`, and project `settings.json` is ignored. |

The agent dir's own `settings.json` is still read. That is intended: it belongs to the harness package.

**Captured request (lockdown, vanilla):**

```json
{"model": "qwen3-8b", "stream": true, "stream_options": {"include_usage": true},
 "max_completion_tokens": 8192, "reasoning_effort": "medium",
 "messages": [{"role": "user", "content": [{"type": "text", "text": "<prompt>"}]}]}
```

The headers include `User-Agent: pi (linux …)`, `x-stainless-*`, `Authorization: Bearer <ROLLOUT_ID>` and `X-Rollout-Id`.

**Network.** With `--offline` / `PI_OFFLINE=1` and `PI_TELEMETRY=0`, the only connection observed is to the provider URL. The docs list these as Pi's own network calls, all disabled or inapplicable in this mode: the pi.dev version check, install telemetry, catalog refresh, MCP servers, and npm installs for packages listed in settings.

- **Not enforced at the OS level.** Unprivileged network namespaces are blocked on this host (`apparmor_restrict_unprivileged_userns=1`); sudo is available.
- **Proposal:** treat it as configuration-enforced for Phases 0–1, where tools are off and Pi runs no user code. Add an iptables owner-match (or a dedicated uid) that allows only the proxy port before Phase 4, when model-written code runs.

## 4. Streaming and request fields

- **Pi always streams** (`stream: true`), and there is no non-streaming switch. The proxy must implement SSE; non-streaming is only needed for our own tests and `--direct`.
- **Sent:**
  - `model`
  - `stream_options.include_usage`
  - `max_completion_tokens = min(model.maxTokens, contextWindow − est_input − 4096)`
  - `reasoning_effort` = thinking level (default `medium`; omitted with `--thinking off`)
  - `store: false` unless `supportsStore: false`
- **Not sent:** `temperature`, `top_p`, `top_k`, `seed`, `stop`. They are only sent if `samplingParams` is configured in models.json. The proxy takes sampling from the rollout config anyway and logs what Pi asked for.
- **User content arrives as a parts array:** `[{"type":"text","text":...}]`. The proxy must flatten text parts to a string before applying the Qwen3 template; the template does not concatenate parts the way we need.
- **Prompt delivery:**
  - argv is unusable: `@…` is read as a file path even after `--`, and a single argument is capped at 128 KB (`E2BIG`).
  - Stdin is byte-exact (tested with Unicode, zero-width characters, tabs, a 300 KB prompt, and leading `@`, `-` or `/`), **except that Pi trims leading and trailing whitespace**.
  - Rule for 0.4: the prompt goes through stdin, and `PiAgentLoop` asserts `prompt == prompt.strip()`. All Omanic prompts satisfy this.

## 5. Qwen3 reasoning (`reasoning_content`)

- Pi reads streamed `reasoning_content` (then `reasoning`, then `reasoning_text`) into a `thinking` content block with `thinkingSignature: "reasoning_content"`, even before `model.reasoning` is considered. It is "displayed" only as a separate block in the JSON event stream and the session; `-p` text output prints only the text blocks.
- **It is sent back on later turns** as `assistant.reasoning_content` (verified with a two-turn run), when the provider, api and model are unchanged.
  - In Phases 0–1 there is one call, so this does not matter.
  - For multi-call trajectories (Phase 6), note that the Qwen3 template drops `reasoning_content` from turns before the last user message anyway. The proxy renders with the template, so the rendering decides, not Pi.
- **Consequence for the proxy:** it returns `<think>…</think>` content as `reasoning_content` and the rest as `content`. Pi's view of the answer is then exactly the post-`</think>` text, which matches what `omanic/reward.py` parses.
- **Thinking level:** we run Pi with `--thinking off` so it doesn't send a misleading `reasoning_effort`. Qwen3 thinking is controlled by the proxy's `enable_thinking=True`, the same as training.

## 6. Invocation-mode overhead

Script: `spike/bench_overhead.py`; results in `spike/results_overhead.jsonl`.

- **Setup:** the stub streams 3,300 SSE deltas over ~18.5 s per request, which mimics one ~3.3k-token Omanic response token by token. Per-rollout overhead = client wall time − stub time from request arrival to stream end. It therefore covers everything Pi adds: process start, config load, request build, parsing 3,300 deltas, emitting events, and exit. The prompt is ~700 chars, sent through the lockdown flags.
- **Modes:**
  - **json:** one `pi -p --mode json` subprocess per rollout.
  - **rpc:** a pool of C long-lived `pi --mode rpc` processes, with `new_session` + `prompt` per rollout.
  - **sdk:** G Node gateway processes (`sdk_gateway.mjs`), one in-memory SDK session per rollout.

**Median direct rollout time (the 10% rule's denominator).**
- Script: `spike/measure_rollout_time.py`; result in `spike/results_rollout_time.json`.
- Setup: base Qwen3-8B on vLLM TP2, 640 Omanic train rollouts (128 prompts × 5) with 320 in flight, which is verl's per-replica load (1,280 rollouts over 4 TP2 replicas). Thinking on, T=1.0, top_p=1.0, max 8,192 tokens.
- Results:
  - **Median 101 s**, p10 26 s, p90 137 s, max 181 s.
  - Median 3,027 tokens; 1.9% truncated.
  - The batch took 253 s, matching the ≈ 270 s generation phase in the existing run's logs.
- **The 10% budget is therefore ≈ 10 s per rollout.**

**Pi overhead per rollout** (seconds; all rollouts succeeded in every row):

| Stream shape | In flight | Mode | Median | p90 | p99 | Max |
|---|---|---|---|---|---|---|
| 3,300 deltas over 18.5 s | 64 | json | 0.93 | 1.39 | 1.57 | 1.57 |
| | 64 | rpc (64 procs) | 0.40 | 0.70 | 0.76 | 0.76 |
| | 64 | sdk (1 gateway) | 0.10 | 0.20 | 0.21 | 0.21 |
| | 64 | sdk (8 gateways) | 0.08 | 0.13 | 0.14 | 0.14 |
| | 256 | json | 3.13 | 5.52 | 6.23 | 6.47 |
| | 256 | json, light parse* | 1.05 | 2.36 | 2.85 | 3.16 |
| | 256 | rpc (256 procs) | 0.62 | 1.93 | 1.98 | 2.01 |
| | 256 | sdk (1 gateway) | 0.25 | 0.60 | 0.62 | 0.62 |
| | 256 | sdk (8 gateways) | 0.09 | 0.23 | 0.24 | 0.25 |
| | **1,280** | json | 21.6 | 30.7 | 34.2 | 35.4 |
| | **1,280** | json, light parse* | 10.3 | 14.2 | 16.1 | 16.8 |
| | **1,280** | sdk (8 gateways) | 0.40 | 1.44 | 1.57 | 1.57 |
| **Proxy-like: 20 s wait, then 13 KB in 3 chunks** | **1,280** | json, light parse* | **8.3** | 11.3 | 12.4 | 12.6 |
| | **1,280** | sdk (8 gateways) | **0.36** | 1.26 | 1.36 | 1.37 |

\* "Light parse": the driver `json.loads` only the `message_end` lines, as `PiAgentLoop` will. In the first JSON rows the single Python driver parsed all ~4.2 M event lines, which inflated the numbers.

Reading the table:
- **The plan's 64/256 points understate the real load.** A training step has 256 prompts × 5 = **1,280 rollouts in flight at once** on one node with 208 cores. The 1,280 rows are the relevant ones.
- **Which stream shape applies.** The proxy calls verl's token-in/token-out `generate` and only streams the finished result, so Pi sees the proxy-like shape: a long wait, then a few chunks.
- **The JSON-mode cost is mostly startup.** 1,280 Node processes boot at once (about 0.4 s CPU each, plus config and extension load via jiti). Parsing the stream adds little.
- **SDK is about 20× cheaper.** It holds under 1.6 s p99 at every scale tested.


## Decision

**Under the plan's rule, JSON mode passes, but narrowly.**
- At full training concurrency with the proxy's stream shape, median Pi overhead is **8.3 s = 8.2 % of the 101 s median rollout** (rule: < 10 %).
- Its p99 of 12.4 s exceeds 10 %, and token-streamed shapes go over (10.3 s).
- Measured against step time (the Phase 0 gate: < 10 % of step time), JSON costs roughly the p99, ≈ 12 s of a ≈ 425 s step (≈ 3 %), because rollouts run in parallel and the tail sets the step's generation time.

**Recommendation:**
- Build 0.4 on the JSON-mode subprocess, as the plan's default, behind a small `PiRunner` interface: `run(prompt, rollout_home, rollout_id) -> final message`.
- Measure real step-time overhead in the 0.7.5 throughput check.
- The SDK gateway (`spike/sdk_gateway.mjs`, already working: 1,280/1,280 succeeded at 0.36 s median) is the drop-in second implementation if the 0.7.5 measurement shows > 10 %, or once multi-call trajectories in Phase 6 multiply startups.

The choice between the two is yours at this STOP; switching later costs one class.

**Caveats:**
- The JSON driver here is a single Python process managing 1,280 subprocesses. In verl the 8 agent-loop workers share that work, so production should be no worse.
- verl's own CPU load during generation is not reproduced here.

