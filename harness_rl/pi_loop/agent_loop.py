"""PiAgentLoop: a verl agent loop that runs Pi once per rollout, with tokens captured by the in-process model proxy.

Per sample:
  1. register a rollout id with this worker's ModelProxy (sampling params from verl, never from Pi)
  2. run Pi (JsonSubprocessRunner) on the task prompt with the proxy URL /r/{rollout_id}/v1
  3. collect the proxy's call records; Phase 0/1 contract: exactly one model call
  4. return the call's exact prompt_ids / response_ids / logprobs to verl
For a harness that sends no system message (vanilla), every rollout also asserts prompt ids == the direct rendering.

The reward is left to verl's reward manager, which decodes response_ids exactly as on the direct path
(SingleTurnAgentLoop). A rollout that breaks the contract (Pi error, zero or several calls) is marked invalid: its
response becomes a single masked-out <|im_end|>, which decodes to "" and so scores 0 through the same reward path,
and it contributes no gradient.

Config (top-level hydra keys, added with +):
  +harness_rl.harness=vanilla            harness package name or path
  +harness_rl.trace_dir=/path            metrics_<pid>.jsonl: one row per rollout (validity, calls, prompt match,
                                         overhead, global_steps); traces_<pid>.jsonl: full traces of a deterministic
                                         ~5% sample. scripts/pi_monitor.py summarizes both while a run trains.
  +harness_rl.trace_frac=0.05
  +harness_rl.checkpoint_id=..., +harness_rl.seed=...   copied into every trace record
Select it with actor_rollout_ref.rollout.agent.default_agent_loop=pi_agent and
actor_rollout_ref.rollout.agent.agent_loop_config_path=<harness_rl/pi_loop/agent_loop.yaml>.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register

from harness_rl.core.trace import RolloutTrace
from harness_rl.pi_loop.runner import HarnessPackage, JsonSubprocessRunner
from harness_rl.proxy.backends import VerlBackend
from harness_rl.proxy.server import ModelProxy

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_WORKER_STATE: dict[int, dict] = {}  # per process: one proxy and one runner, shared by all loop instances
_STATE_LOCK = asyncio.Lock()


def _hr_config(config) -> dict:
    hr = config.get("harness_rl", None) or {}
    return dict(hr)


async def _worker_state(server_manager, tokenizer, config) -> dict:
    pid = os.getpid()
    async with _STATE_LOCK:
        if pid not in _WORKER_STATE:
            hr = _hr_config(config)
            harness = HarnessPackage.load(hr.get("harness", "vanilla"))
            proxy = ModelProxy(backend=VerlBackend(server_manager), tokenizer=tokenizer)
            await proxy.start()
            trace_dir = Path(hr["trace_dir"]) if hr.get("trace_dir") else None
            if trace_dir:
                trace_dir.mkdir(parents=True, exist_ok=True)
            sp = harness.manifest.get("system_prompt", {})
            # A harness that sends no system message must produce exactly the direct path's prompt ids.
            expect_direct = sp.get("mode") == "replace" and (harness.root / sp["file"]).read_text() == ""
            _WORKER_STATE[pid] = {
                "harness": harness, "proxy": proxy, "runner": JsonSubprocessRunner(harness),
                "expect_direct_render": expect_direct,
                "trace_dir": trace_dir, "trace_frac": float(hr.get("trace_frac", 0.05)),
                "checkpoint_id": str(hr.get("checkpoint_id", "")), "seed": hr.get("seed", None),
            }
            logger.warning("PiAgentLoop worker %d: harness=%s version=%s proxy port=%d",
                           pid, harness.name, harness.version[:12], proxy.port)
        return _WORKER_STATE[pid]


def _sampled(rollout_id: str, frac: float) -> bool:
    h = int(hashlib.sha256(rollout_id.encode()).hexdigest()[:8], 16)
    return h < frac * 0xFFFFFFFF


@register("pi_agent")
class PiAgentLoop(AgentLoopBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.response_length = self.rollout_config.response_length
        self.im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        st = await _worker_state(self.server_manager, self.tokenizer, self.config)
        proxy: ModelProxy = st["proxy"]
        messages = list(kwargs["raw_prompt"])
        if len(messages) != 1 or messages[0]["role"] != "user" or not isinstance(messages[0]["content"], str):
            raise ValueError("PiAgentLoop (Phase 0/1) expects a single user text message per sample")
        prompt = messages[0]["content"]

        rollout_id = uuid.uuid4().hex
        proxy.register_rollout(rollout_id, sampling_params=dict(sampling_params), max_tokens=self.response_length,
                               chat_template_kwargs=dict(self.apply_chat_template_kwargs))
        t0 = time.time()
        try:
            res = await st["runner"].run(prompt, rollout_id, proxy.rollout_base_url(rollout_id))
        finally:
            calls = proxy.collect(rollout_id)
        wall = time.time() - t0

        problems = []
        if not res.ok:
            problems.append(f"pi: {res.error}")
        if len(calls) != 1:
            problems.append(f"model calls = {len(calls)}, expected 1")
        if res.assistant_messages != 1:
            problems.append(f"pi assistant messages = {res.assistant_messages}, expected 1")
        prompt_match = None
        if calls and st["expect_direct_render"]:
            prompt_match = list(calls[0].prompt_ids) == await self.ct_build_initial_tokens(messages)
            if not prompt_match:
                problems.append("prompt ids differ from the direct rendering")
        if calls and len(calls[0].prompt_ids) > self.rollout_config.prompt_length:
            # The direct path would left-truncate; we refuse instead (data.filter_overlong_prompts normally prevents it).
            problems.append(f"prompt {len(calls[0].prompt_ids)} tokens > rollout.prompt_length")
        valid = not problems

        if calls:
            call = calls[0]
            prompt_ids = list(call.prompt_ids)
            response_ids = list(call.response_ids)[: self.response_length]
            logprobs = list(call.response_logprobs)[: self.response_length] if call.response_logprobs else None
            gen_s = call.timings.get("t_gen_end", 0) - call.timings.get("t_gen_start", 0)
        else:
            prompt_ids = await self.ct_build_initial_tokens(messages)
            response_ids, logprobs, gen_s = [], None, 0.0
        if valid:
            response_mask = [1] * len(response_ids)
        else:
            logger.warning("invalid Pi rollout %s: %s", rollout_id, "; ".join(problems))
            response_ids, response_mask = [self.im_end_id], [0]
            logprobs = [0.0] if (calls and calls[0].response_logprobs) else None

        # verl's trainer reads these weight-version tags (set by LLMServerClient) from every rollout.
        engine = calls[0].engine_meta if calls else {}
        if engine.get("global_steps") is not None:
            st["last_global_steps"] = engine["global_steps"]
        fallback = st.get("last_global_steps", 0)
        engine_tags = {k: engine.get(k) if engine.get(k) is not None else fallback
                       for k in ("global_steps", "min_global_steps", "max_global_steps")}
        engine_tags["num_cached_tokens"] = engine.get("num_cached_tokens")

        extra = {
            **engine_tags,
            "pi_valid": int(valid), "pi_problems": "; ".join(problems), "pi_rollout_id": rollout_id,
            "pi_wall_s": wall, "pi_gen_s": gen_s, "pi_overhead_s": wall - gen_s, "pi_num_calls": len(calls),
            "harness_version": st["harness"].version, "turn_scores": [], "tool_rewards": [],
        }
        if st["trace_dir"] is not None:
            row = {"t": time.time(), "rollout_id": rollout_id, "valid": int(valid), "problems": extra["pi_problems"],
                   "calls": len(calls), "prompt_match": prompt_match, "wall_s": round(wall, 3),
                   "gen_s": round(gen_s, 3), "overhead_s": round(wall - gen_s, 3), "n_prompt": len(prompt_ids),
                   "n_response": len(response_ids), "finish": calls[0].finish_reason if calls else None,
                   "global_steps": engine_tags["global_steps"]}
            with open(st["trace_dir"] / f"metrics_{os.getpid()}.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            if _sampled(rollout_id, st["trace_frac"]):
                self._write_trace(st, rollout_id, calls, res, extra, kwargs)

        return AgentLoopOutput(
            prompt_ids=prompt_ids, response_ids=response_ids, response_mask=response_mask,
            response_logprobs=logprobs, num_turns=2, metrics={"generate_sequences": wall}, extra_fields=extra,
        )

    def _write_trace(self, st, rollout_id, calls, res, extra, kwargs):
        info = kwargs.get("extra_info") or {}
        trace = RolloutTrace(
            rollout_id=rollout_id, calls=calls, harness_version=st["harness"].version, adapter_version="",
            eval_set_hash="", checkpoint_id=st["checkpoint_id"], seed=st["seed"],
            task_id=str(info.get("id", info.get("index", ""))),
            extra={"t": time.time(), "data_source": kwargs.get("data_source"), "index": info.get("index"),
                   "pi_final": res.final, "pi_error": res.error, "pi_stderr_tail": res.stderr_tail,
                   **{k: v for k, v in extra.items() if k.startswith("pi_")}},
        )
        path = st["trace_dir"] / f"traces_{os.getpid()}.jsonl"
        with open(path, "a") as f:
            f.write(trace.to_json() + "\n")
