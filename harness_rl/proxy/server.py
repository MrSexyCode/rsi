"""ModelProxy: the OpenAI-compatible endpoint Pi talks to, and the place where training tokens are captured.

Embeddable in a running asyncio loop (aiohttp AppRunner on 127.0.0.1, port 0 allowed); one proxy per verl
AgentLoopWorker serves all of that worker's rollouts. Per request: normalize messages, render + tokenize exactly as
verl's direct path, generate with the rollout's registered sampling (never the harness's), decode, split
reasoning/content, parse tool calls, answer as chat.completion JSON or SSE. Every call becomes a CallRecord on its
rollout; `collect()` pops them.

Streaming: the engine is not token-streamed, so SSE headers go out at once, `: keepalive` comments are written every
`keepalive_s` while generating (Pi's HTTP idle timeout), and the finished response is then sent as role ->
reasoning_content -> content -> tool_calls -> finish_reason -> usage -> [DONE]. A failure after the headers are
out is sent as a `data: {"error": ...}` event, which the OpenAI SDK in Pi raises as an APIError.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from aiohttp import web

from harness_rl.core.trace import CallRecord
from harness_rl.proxy.backends import Backend
from harness_rl.proxy.parse import api_content, api_reasoning, decode_response, finish_reason, parse_tool_calls, split_reasoning
from harness_rl.proxy.render import BadRequest, Renderer, normalize_messages

SSE_HEADERS = {"Content-Type": "text/event-stream", "Cache-Control": "no-cache", "Connection": "keep-alive"}


@dataclass
class _Rollout:
    sampling_params: dict
    max_tokens: int
    chat_template_kwargs: dict
    records: list[CallRecord] = field(default_factory=list)
    next_index: int = 0


@dataclass
class _Result:
    status: int = 200
    error: str | None = None
    reasoning: str = ""
    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict | None = None


class ModelProxy:
    def __init__(self, tokenizer, backend: Backend, *, host: str = "127.0.0.1", port: int = 0,
                 model_name: str = "qwen3-8b", keepalive_s: float = 15.0, stream_chunk_chars: int = 0,
                 render_workers: int = 1, hf_model_type: str | None = "qwen3", use_verl: bool = True,
                 eos_token_ids: set[int] | None = None, admin: bool = False):
        self.tokenizer = tokenizer
        self.backend = backend
        self.host, self.port = host, port
        self.model_name = model_name
        self.keepalive_s = keepalive_s
        self.stream_chunk_chars = stream_chunk_chars
        self.admin = admin
        self.renderer = Renderer(tokenizer, hf_model_type=hf_model_type, use_verl=use_verl)
        self.eos_ids = eos_token_ids or {tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")}
        # One thread by default: HF fast tokenizers are not safe under concurrent use, and rendering is cheap.
        self._pool = ThreadPoolExecutor(max_workers=render_workers, thread_name_prefix="proxy-render")
        self._rollouts: dict[str, _Rollout] = {}
        self._lock = threading.Lock()
        self._runner: web.AppRunner | None = None

    # ---- lifecycle ----

    async def start(self) -> int:
        app = web.Application(client_max_size=256 * 1024 * 1024)
        app.router.add_post("/r/{rollout_id}/v1/chat/completions", self._chat)
        app.router.add_get("/r/{rollout_id}/v1/models", self._models)
        if self.admin:
            app.router.add_post("/admin/register/{rollout_id}", self._admin_register)
            app.router.add_post("/admin/collect/{rollout_id}", self._admin_collect)
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        await web.TCPSite(self._runner, self.host, self.port, backlog=4096).start()
        self.port = self._runner.addresses[0][1]
        return self.port

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
        self._pool.shutdown(wait=False)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def rollout_base_url(self, rollout_id: str) -> str:
        """The literal OpenAI baseUrl for one rollout (goes into its models.json)."""
        return f"{self.base_url}/r/{rollout_id}/v1"

    # ---- rollout registry (thread-safe) ----

    def register_rollout(self, rollout_id: str, sampling_params: dict, max_tokens: int,
                         chat_template_kwargs: dict | None = None) -> None:
        with self._lock:
            if rollout_id in self._rollouts:
                raise ValueError(f"rollout {rollout_id!r} already registered")
            self._rollouts[rollout_id] = _Rollout(dict(sampling_params), int(max_tokens), dict(chat_template_kwargs or {}))

    def collect(self, rollout_id: str) -> list[CallRecord]:
        """Pop the rollout's records (in call order) and unregister it. Unknown id -> []."""
        with self._lock:
            ro = self._rollouts.pop(rollout_id, None)
        return list(ro.records) if ro else []

    def _new_record(self, rollout_id: str, t_recv: float) -> tuple[_Rollout | None, CallRecord | None]:
        with self._lock:
            ro = self._rollouts.get(rollout_id)
            if ro is None:
                return None, None
            rec = CallRecord(rollout_id=rollout_id, call_index=ro.next_index, timings={"t_recv": t_recv})
            ro.next_index += 1
            ro.records.append(rec)  # appended on arrival and filled in place, so failed calls are recorded too
        return ro, rec

    # ---- handlers ----

    async def _models(self, request: web.Request) -> web.Response:
        with self._lock:
            known = request.match_info["rollout_id"] in self._rollouts
        if not known:
            return _error_response(404, "unknown rollout id")
        return web.json_response({"object": "list", "data": [{"id": self.model_name, "object": "model", "owned_by": "harness_rl"}]})

    async def _chat(self, request: web.Request) -> web.StreamResponse:
        t_recv = time.time()
        rid = request.match_info["rollout_id"]
        hdr = request.headers.get("X-Rollout-Id")
        if hdr is not None and hdr != rid:
            return _error_response(400, f"X-Rollout-Id {hdr!r} does not match path rollout id {rid!r}")
        ro, rec = self._new_record(rid, t_recv)
        if ro is None:
            return _error_response(404, f"unknown rollout id {rid!r}")
        try:
            body = json.loads(await request.read())
            if not isinstance(body, dict):
                raise BadRequest("request body must be a JSON object")
            rec.requested = {k: v for k, v in body.items() if k not in ("messages", "tools")}
            rec.tools_in = body.get("tools") or None
            msgs, dropped = normalize_messages(body.get("messages"))
            rec.messages_in = msgs
            if dropped:
                rec.requested["_dropped_message_keys"] = dropped
        except (BadRequest, json.JSONDecodeError, UnicodeDecodeError) as e:
            _fail(rec, str(e))
            return _error_response(400, str(e))

        stream = bool(body.get("stream"))
        model = body.get("model") or self.model_name
        task = asyncio.ensure_future(self._run(ro, rec))
        if not stream:
            res = await task
            resp = (_error_response(res.status, res.error) if res.error
                    else web.json_response(self._completion(rec, res, model)))
            rec.timings["t_resp_done"] = time.time()
            return resp

        resp = web.StreamResponse(headers=SSE_HEADERS)
        await resp.prepare(request)
        alive = True
        while not task.done():
            done, _ = await asyncio.wait({task}, timeout=self.keepalive_s)
            if not done and alive:
                try:
                    await resp.write(b": keepalive\n\n")
                except Exception:
                    alive = False
        res = task.result()
        if res.error:
            payload = _sse({"error": {"message": res.error, "type": "server_error" if res.status >= 500 else "invalid_request_error",
                                      "code": res.status}})
        else:
            include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
            payload = b"".join(_sse(c) for c in self._chunks(rec, res, model, include_usage)) + b"data: [DONE]\n\n"
        if alive:
            try:
                await resp.write(payload)
                await resp.write_eof()
            except Exception:
                alive = False
        if not alive:
            rec.engine_meta["client_disconnected"] = True
        rec.timings["t_resp_done"] = time.time()
        return resp

    async def _run(self, ro: _Rollout, rec: CallRecord) -> _Result:
        loop = asyncio.get_running_loop()
        kw = ro.chat_template_kwargs
        try:
            rec.prompt_ids, rec.rendered_prompt = await loop.run_in_executor(
                self._pool, self.renderer.render, rec.messages_in, rec.tools_in, kw)
        except Exception as e:
            return _fail(rec, f"render failed: {e!r}", 400)
        rec.timings["t_render_done"] = time.time()
        sp = {**ro.sampling_params, "max_tokens": ro.max_tokens}
        rec.applied_sampling = dict(sp)
        rec.timings["t_gen_start"] = time.time()
        try:
            out = await self.backend.generate(rec.rollout_id, rec.prompt_ids, sp)
        except Exception as e:
            rec.timings["t_gen_end"] = time.time()
            return _fail(rec, f"backend error: {e!r}", 500)
        rec.timings["t_gen_end"] = time.time()
        rec.response_ids, rec.response_logprobs = list(out.token_ids), out.log_probs
        rec.engine_meta.update({**out.meta, "stop_reason": out.stop_reason})
        if out.stop_reason == "aborted":
            return _fail(rec, "generation aborted by the engine", 500)

        text, ended = await loop.run_in_executor(self._pool, decode_response, self.tokenizer, rec.response_ids, self.eos_ids)
        rec.reasoning_raw, rec.content_raw = split_reasoning(text, kw.get("enable_thinking", True) is not False)
        reasoning, content = api_reasoning(rec.reasoning_raw), api_content(rec.content_raw)
        if rec.tools_in:  # only parse tool calls when tools were offered; otherwise the text stays content
            content, rec.tool_calls = parse_tool_calls(content, f"call_{rec.call_index}_")
        rec.finish_reason = finish_reason(ended, bool(rec.tool_calls))
        if not ended and len(rec.response_ids) < ro.max_tokens:
            rec.engine_meta["length_below_max_tokens"] = True  # engine clamped to the context window
        usage = {"prompt_tokens": len(rec.prompt_ids), "completion_tokens": len(rec.response_ids),
                 "total_tokens": len(rec.prompt_ids) + len(rec.response_ids)}
        return _Result(reasoning=reasoning, content=content, tool_calls=rec.tool_calls,
                       finish_reason=rec.finish_reason, usage=usage)

    # ---- response bodies ----

    def _completion(self, rec: CallRecord, res: _Result, model: str) -> dict:
        msg: dict[str, Any] = {"role": "assistant", "content": res.content}
        if res.reasoning:
            msg["reasoning_content"] = res.reasoning
        if res.tool_calls:
            msg["tool_calls"] = res.tool_calls
        return {"id": f"chatcmpl-{rec.rollout_id}-{rec.call_index}", "object": "chat.completion",
                "created": int(rec.timings["t_recv"]), "model": model,
                "choices": [{"index": 0, "message": msg, "finish_reason": res.finish_reason}], "usage": res.usage}

    def _pieces(self, s: str) -> list[str]:
        n = self.stream_chunk_chars
        return [s[i:i + n] for i in range(0, len(s), n)] if n and n > 0 else ([s] if s else [])

    def _chunks(self, rec: CallRecord, res: _Result, model: str, include_usage: bool) -> list[dict]:
        base = {"id": f"chatcmpl-{rec.rollout_id}-{rec.call_index}", "object": "chat.completion.chunk",
                "created": int(rec.timings["t_recv"]), "model": model}

        def chunk(delta: dict, finish: str | None = None) -> dict:
            return {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

        out = [chunk({"role": "assistant", "content": ""})]
        out += [chunk({"reasoning_content": p}) for p in self._pieces(res.reasoning)]
        out += [chunk({"content": p}) for p in self._pieces(res.content)]
        out += [chunk({"tool_calls": [{"index": k, **tc}]}) for k, tc in enumerate(res.tool_calls)]
        out.append(chunk({}, res.finish_reason))
        if include_usage:
            out.append({**base, "choices": [], "usage": res.usage})
        return out

    # ---- admin (standalone use: register/collect over HTTP) ----

    async def _admin_register(self, request: web.Request) -> web.Response:
        b = await request.json()
        try:
            self.register_rollout(request.match_info["rollout_id"], b.get("sampling_params", {}), b["max_tokens"],
                                  b.get("chat_template_kwargs"))
        except (ValueError, KeyError) as e:
            return _error_response(400, repr(e))
        return web.json_response({"ok": True, "base_url": self.rollout_base_url(request.match_info["rollout_id"])})

    async def _admin_collect(self, request: web.Request) -> web.Response:
        return web.json_response({"records": [r.to_dict() for r in self.collect(request.match_info["rollout_id"])]})


def _fail(rec: CallRecord, msg: str, status: int = 400) -> _Result:
    rec.error, rec.finish_reason = msg, "error"
    return _Result(status=status, error=msg)


def _sse(obj: dict) -> bytes:
    return b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n"


def _error_response(status: int, message: str) -> web.Response:
    kind = "server_error" if status >= 500 else ("not_found_error" if status == 404 else "invalid_request_error")
    return web.json_response({"error": {"message": message, "type": kind, "code": status}}, status=status)
