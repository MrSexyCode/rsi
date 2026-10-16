"""Token-in/token-out generation backends for the proxy.

- VerlBackend: verl's LLMServerClient (`server_manager` inside an agent loop); the rollout id is the sticky request id.
- VLLMHTTPBackend: a standalone `vllm serve` endpoint, via /v1/completions with a token-id prompt and
  `return_token_ids` (vLLM >= 0.29), for the eval runner outside verl.
- FakeBackend: deterministic scripted responses for tests.
All return GenOutput; `stop_reason` follows verl's TokenOutput ("completed" | "aborted" | None).
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


@dataclass
class GenOutput:
    token_ids: list[int]
    log_probs: list[float] | None = None
    stop_reason: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class Backend(Protocol):
    async def generate(self, request_id: str, prompt_ids: list[int], sampling_params: dict) -> GenOutput: ...


def _json_safe(d: dict) -> dict:
    out = {}
    for k, v in (d or {}).items():
        if hasattr(v, "item") and getattr(v, "ndim", None) == 0:  # numpy scalar, e.g. verl's global_steps
            v = v.item()
        try:
            json.dumps(v)
            out[k] = v
        except (TypeError, ValueError):
            pass
    return out


class VerlBackend:
    """Wraps verl's `LLMServerClient.generate`. Extra kwargs (e.g. priority) are forwarded on every call."""

    def __init__(self, server_manager, **generate_kwargs: Any):
        self.server_manager = server_manager
        self.generate_kwargs = generate_kwargs

    async def generate(self, request_id: str, prompt_ids: list[int], sampling_params: dict) -> GenOutput:
        out = await self.server_manager.generate(request_id=request_id, prompt_ids=list(prompt_ids),
                                                 sampling_params=dict(sampling_params), **self.generate_kwargs)
        meta = {"stop_reason": out.stop_reason, "num_preempted": out.num_preempted, **_json_safe(out.extra_fields)}
        return GenOutput(list(out.token_ids), list(out.log_probs) if out.log_probs else None, out.stop_reason, meta)


class VLLMHTTPBackend:
    """Token-in/token-out against `vllm serve` via POST {base_url}/v1/completions.

    The prompt is a list of token ids; `return_token_ids` returns the raw response ids (eos included, as in verl),
    and `logprobs: 0` returns the sampled token's logprob per position.
    """

    def __init__(self, base_url: str, model: str, timeout_s: float = 3600.0):
        self.url = base_url.rstrip("/") + "/v1/completions"
        self.model = model
        self.timeout_s = timeout_s
        self._session = None

    async def _sess(self):
        import aiohttp
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout_s),
                                                  connector=aiohttp.TCPConnector(limit=0))
        return self._session

    async def close(self):
        if self._session is not None:
            await self._session.close()

    async def generate(self, request_id: str, prompt_ids: list[int], sampling_params: dict) -> GenOutput:
        sp = dict(sampling_params)
        want_lp = bool(sp.pop("logprobs", False))
        body = {"model": self.model, "prompt": list(prompt_ids), "stream": False, "return_token_ids": True,
                "skip_special_tokens": False, **sp}
        if want_lp:
            body["logprobs"] = 0
        sess = await self._sess()
        async with sess.post(self.url, json=body) as r:
            data = await r.json()
            if r.status != 200:
                raise RuntimeError(f"vLLM HTTP {r.status}: {data}")
        ch = data["choices"][0]
        lps = ch["logprobs"]["token_logprobs"] if want_lp and ch.get("logprobs") else None
        stop = "aborted" if ch.get("finish_reason") == "abort" else "completed"
        meta = {"stop_reason": stop, "vllm_finish_reason": ch.get("finish_reason"), "vllm_stop_reason": ch.get("stop_reason"),
                "usage": data.get("usage")}
        return GenOutput(list(ch["token_ids"]), lps, stop, meta)


Script = list[int] | GenOutput | Callable[[str, int, list[int], dict], "list[int] | GenOutput"]


class FakeBackend:
    """Deterministic backend. `scripts` gives the response per call: a single script used for every call, a list
    indexed by the per-request-id call number (the last entry repeats), or a callable
    (request_id, call_no, prompt_ids, sampling_params) -> ids | GenOutput.

    Token ids are cut to sampling_params["max_tokens"], so a long script without `</think>`/eos is the truncated
    case. Logprobs are fake but deterministic, returned only when sampling_params["logprobs"] is truthy.
    """

    def __init__(self, scripts: Script | list[Script], delay_s: float = 0.0, stop_reason: str = "completed"):
        self.scripts = scripts
        self.delay_s = delay_s
        self.stop_reason = stop_reason
        self.calls: list[dict] = []  # every generate() call, for assertions
        self._n: dict[str, int] = {}

    def _pick(self, request_id: str, call_no: int, prompt_ids: list[int], sp: dict):
        s = self.scripts
        if isinstance(s, list) and s and not isinstance(s[0], int):
            s = s[min(call_no, len(s) - 1)]
        return s(request_id, call_no, prompt_ids, sp) if callable(s) else s

    async def generate(self, request_id: str, prompt_ids: list[int], sampling_params: dict) -> GenOutput:
        call_no = self._n.get(request_id, 0)
        self._n[request_id] = call_no + 1
        self.calls.append({"request_id": request_id, "prompt_ids": list(prompt_ids), "sampling_params": dict(sampling_params)})
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        s = self._pick(request_id, call_no, prompt_ids, sampling_params)
        out = s if isinstance(s, GenOutput) else GenOutput(list(s), stop_reason=self.stop_reason)
        if out.stop_reason == "aborted":
            return out
        ids = out.token_ids[: sampling_params.get("max_tokens", len(out.token_ids))]
        lps = [-((t * 7 + i) % 97 + 1) / 100.0 for i, t in enumerate(ids)] if sampling_params.get("logprobs") else None
        return GenOutput(ids, lps, out.stop_reason, {"stop_reason": out.stop_reason, "num_preempted": 0, **out.meta})


def qwen3_script(tokenizer, text: str, eos: bool = True) -> list[int]:
    """Token ids the model would emit for `text`, ending with <|im_end|> when eos."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids + [tokenizer.convert_tokens_to_ids("<|im_end|>")] if eos else ids
