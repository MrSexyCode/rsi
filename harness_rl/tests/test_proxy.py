"""0.3 model proxy: token-exact rendering, lossless reasoning split, SSE == JSON, sampling ownership, routing, load.

Rendering is checked against both `tokenizer.apply_chat_template(tokenize=True)` and verl's Continuous Token builder
(the direct path's tokenizer). Server tests run the real aiohttp proxy on port 0 with a FakeBackend.
"""

import asyncio
import json
import time

import aiohttp
import pytest
from transformers import AutoTokenizer

from harness_rl.core.env_paths import MODEL_ID, OMANIC_DATA
from harness_rl.core.trace import CallRecord, RolloutTrace, read_jsonl, write_jsonl
from harness_rl.proxy import FakeBackend, GenOutput, ModelProxy, VerlBackend, VLLMHTTPBackend, qwen3_script
from harness_rl.proxy.parse import api_content, api_reasoning, decode_response, parse_tool_calls, split_reasoning
from harness_rl.proxy.render import BadRequest, Renderer, normalize_ids, normalize_messages

SAMPLING = {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "repetition_penalty": 1.0, "logprobs": True}
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Weather for a city",
                                           "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                                          "required": ["city"]}}}]
NORMAL = "<think>\nLet me think. Option B fits.\n</think>\n\nThe answer follows.\nAnswer: B"
TOOL_TEXT = ('<think>\nI should check the weather.\n</think>\n\n<tool_call>\n{"name": "get_weather", '
             '"arguments": {"city": "Paris"}}\n</tool_call>')
LONG_THOUGHT = "<think>\n" + "Step: consider the options carefully. " * 40


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(MODEL_ID)


@pytest.fixture(scope="module")
def omanic_prompt():
    import pandas as pd
    return pd.read_parquet(OMANIC_DATA / "val.parquet").prompt.iloc[0][0]["content"]


def hf_ids(tok, messages, tools=None, **kw):
    return normalize_ids(tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, tools=tools, **kw))


def verl_ids(tok, messages, tools=None, **kw):
    from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder
    return normalize_ids(create_continuous_token_builder(tok, hf_model_type="qwen3", chat_template_kwargs=kw)
                         .build_initial_tokens(messages, tools=tools))


# ---------------- rendering ----------------

def _fixtures(omanic):
    """(name, messages as Pi sends them, equivalent plain messages, tools)."""
    asst = {"role": "assistant", "content": "Answer: B", "reasoning_content": "B because of X."}
    return [
        ("omanic", [{"role": "user", "content": [{"type": "text", "text": omanic}]}], [{"role": "user", "content": omanic}], None),
        ("system_user", [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Q?"}],
         [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Q?"}], None),
        ("developer", [{"role": "developer", "content": "Be brief."}, {"role": "user", "content": "Q?"}],
         [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Q?"}], None),
        ("parts", [{"role": "user", "content": [{"type": "text", "text": "Part one. "}, {"type": "text", "text": "Part two."}]}],
         [{"role": "user", "content": "Part one. Part two."}], None),
        ("multiturn_reasoning", [{"role": "user", "content": "Q1"}, asst, {"role": "user", "content": "Q2"}],
         [{"role": "user", "content": "Q1"}, asst, {"role": "user", "content": "Q2"}], None),
        ("tools", [{"role": "system", "content": "Use tools."}, {"role": "user", "content": "Weather in Paris?"}],
         [{"role": "system", "content": "Use tools."}, {"role": "user", "content": "Weather in Paris?"}], TOOLS),
        ("tool_roundtrip", [
            {"role": "user", "content": "Weather in Paris?"},
            {"role": "assistant", "content": "", "reasoning_content": "Check it.",
             "tool_calls": [{"id": "c0", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}]},
            {"role": "tool", "tool_call_id": "c0", "content": [{"type": "text", "text": "Sunny"}]}],
         [{"role": "user", "content": "Weather in Paris?"},
          {"role": "assistant", "content": "", "reasoning_content": "Check it.",
           "tool_calls": [{"id": "c0", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}]},
          {"role": "tool", "tool_call_id": "c0", "content": "Sunny"}], TOOLS),
    ]


@pytest.mark.parametrize("thinking", [True, False])
def test_render_matches_hf_and_verl(tok, omanic_prompt, thinking):
    kw = {"enable_thinking": thinking}
    verl_r, hf_r = Renderer(tok), Renderer(tok, use_verl=False)
    assert verl_r.uses_verl and not hf_r.uses_verl
    for name, pi_msgs, plain, tools in _fixtures(omanic_prompt):
        msgs, _ = normalize_messages(pi_msgs)
        ids, text = verl_r.render(msgs, tools, kw)
        expected = hf_ids(tok, plain, tools, **kw)
        assert ids == expected, name
        assert hf_r.render_ids(msgs, tools, kw) == expected, name
        assert verl_ids(tok, plain, tools, **kw) == expected, name
        assert tok.encode(text, add_special_tokens=False) == ids, name
        assert text.endswith("<|im_start|>assistant\n" + ("" if thinking else "<think>\n\n</think>\n\n")), name


def test_omanic_prompt_is_the_direct_path(tok, omanic_prompt):
    msgs, _ = normalize_messages([{"role": "user", "content": [{"type": "text", "text": omanic_prompt}]}])
    ids = Renderer(tok).render_ids(msgs, None, {"enable_thinking": True})
    assert ids == verl_ids(tok, [{"role": "user", "content": omanic_prompt}], enable_thinking=True)
    assert tok.decode(ids) == f"<|im_start|>user\n{omanic_prompt}<|im_end|>\n<|im_start|>assistant\n"


def test_normalize_rules():
    msgs, dropped = normalize_messages([
        {"role": "developer", "content": [{"type": "text", "text": "S"}], "cache_control": {"type": "ephemeral"}},
        {"role": "user", "content": "U", "reasoning_content": "x"},
        {"role": "assistant", "content": None, "reasoning_content": "R", "tool_calls": [], "refusal": None}])
    assert msgs == [{"role": "system", "content": "S"}, {"role": "user", "content": "U"},
                    {"role": "assistant", "content": None, "reasoning_content": "R", "tool_calls": []}]
    assert {(d["index"], d["key"]) for d in dropped} == {(0, "cache_control"), (1, "reasoning_content"), (2, "refusal")}
    for bad in ([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}],
                [{"role": "critic", "content": "x"}], [], "x"):
        with pytest.raises(BadRequest):
            normalize_messages(bad)


# ---------------- decode / split ----------------

EOS = {151645, 151643}


@pytest.mark.parametrize("case", ["normal", "truncated", "tool_call", "no_thinking", "double_close", "unicode"])
def test_split_is_lossless(tok, case):
    text, eos, thinking = {
        "normal": (NORMAL, True, True), "truncated": (LONG_THOUGHT, False, True), "tool_call": (TOOL_TEXT, True, True),
        "no_thinking": ("Answer: B", True, False), "double_close": ("<think>\na\n</think>\n\nb</think>c", True, True),
        "unicode": ("<think>\nÜnïcödé — 中文 ​\t\n</think>\n\n答案: B", True, True)}[case]
    ids = qwen3_script(tok, text, eos=eos)
    decoded, ended = decode_response(tok, ids, EOS)
    assert ended == eos and decoded == text
    r, c = split_reasoning(decoded, thinking)
    assert r + c == decoded == tok.decode(ids[:-1] if eos else ids, skip_special_tokens=False)
    if case == "normal":
        assert (api_reasoning(r), api_content(c)) == ("Let me think. Option B fits.", "The answer follows.\nAnswer: B")
    if case == "truncated":
        assert c == "" and api_reasoning(r) == LONG_THOUGHT[len("<think>\n"):].strip("\n")
    if case == "no_thinking":
        assert (r, c) == ("", "Answer: B")
    if case == "double_close":
        assert (r, c) == ("<think>\na\n</think>", "\n\nb</think>c")


def test_tool_call_parser():
    content, calls = parse_tool_calls(api_content(split_reasoning(TOOL_TEXT, True)[1]), "call_0_")
    assert content == ""
    assert calls == [{"id": "call_0_0", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}]
    two = 'Sure.\n<tool_call>\n{"name": "a", "arguments": {}}\n</tool_call>\n<tool_call>\n{"name": "b", "arguments": "{\\"x\\": 1}"}\n</tool_call>'
    content, calls = parse_tool_calls(two)
    assert content == "Sure." and [c["function"]["name"] for c in calls] == ["a", "b"] and calls[1]["function"]["arguments"] == '{"x": 1}'
    bad = "<tool_call>\nnot json\n</tool_call>"
    assert parse_tool_calls(bad) == (bad, [])


# ---------------- server ----------------

def run(coro):
    return asyncio.run(coro)


async def _with_proxy(tok, backend, fn, **kw):
    proxy = ModelProxy(tok, backend, **kw)
    await proxy.start()
    try:
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0)) as s:
            return await fn(proxy, s)
    finally:
        await proxy.stop()


def parse_sse(raw: str) -> tuple[list[dict], list[str]]:
    events, comments = [], []
    for block in raw.split("\n\n"):
        for line in block.splitlines():
            if line.startswith(":"):
                comments.append(line)
            elif line.startswith("data: "):
                events.append(line[6:])
    assert events and events[-1] == "[DONE]", events[-1:]
    return [json.loads(e) for e in events[:-1]], comments


def reassemble(chunks: list[dict]) -> dict:
    msg, finish, usage, tcs = {"role": None, "content": ""}, None, None, {}
    reasoning = ""
    for c in chunks:
        usage = c.get("usage") or usage
        for ch in c["choices"]:
            d = ch["delta"]
            msg["role"] = d.get("role", msg["role"])
            msg["content"] += d.get("content") or ""
            reasoning += d.get("reasoning_content") or ""
            for tc in d.get("tool_calls", []):
                cur = tcs.setdefault(tc["index"], {"id": tc.get("id"), "type": "function", "function": {"name": "", "arguments": ""}})
                cur["function"]["name"] += tc["function"].get("name") or ""
                cur["function"]["arguments"] += tc["function"].get("arguments") or ""
            finish = ch.get("finish_reason") or finish
    if reasoning:
        msg["reasoning_content"] = reasoning
    if tcs:
        msg["tool_calls"] = [tcs[k] for k in sorted(tcs)]
    return {"message": msg, "finish_reason": finish, "usage": usage}


def _body(stream, tools=None, **extra):
    b = {"model": "qwen3-8b", "messages": [{"role": "user", "content": [{"type": "text", "text": "Which option?"}]}],
         "stream": stream, **extra}
    if stream:
        b["stream_options"] = {"include_usage": True}
    if tools:
        b["tools"] = tools
    return b


@pytest.mark.parametrize("script,tools,max_tokens", [("normal", None, 64), ("truncated", None, 32), ("tool_call", TOOLS, 64)])
@pytest.mark.parametrize("chunk_chars", [0, 5])
def test_stream_reassembles_to_json(tok, script, tools, max_tokens, chunk_chars):
    ids = qwen3_script(tok, {"normal": NORMAL, "truncated": LONG_THOUGHT, "tool_call": TOOL_TEXT}[script],
                       eos=script != "truncated")

    async def go(proxy, s):
        out = {}
        for stream in (False, True):
            rid = f"r-{stream}"
            proxy.register_rollout(rid, SAMPLING, max_tokens, {"enable_thinking": True})
            async with s.post(f"{proxy.rollout_base_url(rid)}/chat/completions", json=_body(stream, tools)) as r:
                assert r.status == 200
                out[stream] = (await r.json()) if not stream else await r.text()
            out[(stream, "rec")] = proxy.collect(rid)
        return out

    out = run(_with_proxy(tok, FakeBackend(ids), go, stream_chunk_chars=chunk_chars))
    j = out[False]
    chunks, _ = parse_sse(out[True])
    assert reassemble(chunks) == {"message": j["choices"][0]["message"], "finish_reason": j["choices"][0]["finish_reason"],
                                  "usage": j["usage"]}
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    assert chunks[-1]["choices"] == [] and chunks[-2]["choices"][0]["finish_reason"] == j["choices"][0]["finish_reason"]
    (rec,) = out[(False, "rec")]
    exp_ids = ids[:max_tokens]
    assert rec.response_ids == exp_ids and len(rec.response_logprobs) == len(exp_ids)
    assert rec.reasoning_raw + rec.content_raw == tok.decode(exp_ids[:-1] if exp_ids[-1] in EOS else exp_ids)
    assert j["usage"] == {"prompt_tokens": len(rec.prompt_ids), "completion_tokens": len(exp_ids),
                          "total_tokens": len(rec.prompt_ids) + len(exp_ids)}
    msg = j["choices"][0]["message"]
    if script == "normal":
        assert j["choices"][0]["finish_reason"] == "stop"
        assert msg == {"role": "assistant", "content": "The answer follows.\nAnswer: B",
                       "reasoning_content": "Let me think. Option B fits."}
    if script == "truncated":
        assert j["choices"][0]["finish_reason"] == rec.finish_reason == "length" and len(exp_ids) == max_tokens
        assert msg["content"] == "" and msg["reasoning_content"] == api_reasoning(rec.reasoning_raw)
    if script == "tool_call":
        assert j["choices"][0]["finish_reason"] == "tool_calls" and msg["content"] == ""
        assert msg["tool_calls"][0]["function"] == {"name": "get_weather", "arguments": '{"city": "Paris"}'}
        assert rec.tool_calls == msg["tool_calls"]


def test_tool_call_text_stays_content_without_tools(tok):
    async def go(proxy, s):
        proxy.register_rollout("r", SAMPLING, 64)
        async with s.post(f"{proxy.rollout_base_url('r')}/chat/completions", json=_body(False)) as r:
            return await r.json()

    j = run(_with_proxy(tok, FakeBackend(qwen3_script(tok, TOOL_TEXT)), go))
    assert j["choices"][0]["finish_reason"] == "stop" and j["choices"][0]["message"]["content"].startswith("<tool_call>")
    assert "tool_calls" not in j["choices"][0]["message"]


def test_sampling_comes_from_registration(tok):
    backend = FakeBackend(qwen3_script(tok, NORMAL))
    pi_fields = {"temperature": 0.1, "top_p": 0.5, "top_k": 3, "max_tokens": 5, "max_completion_tokens": 7,
                 "reasoning_effort": "high", "seed": 9, "store": False}

    async def go(proxy, s):
        proxy.register_rollout("r", SAMPLING, 4096, {"enable_thinking": True})
        async with s.post(f"{proxy.rollout_base_url('r')}/chat/completions", json=_body(True, **pi_fields)) as r:
            await r.read()
        return proxy.collect("r")

    (rec,) = run(_with_proxy(tok, backend, go))
    assert backend.calls[0]["sampling_params"] == {**SAMPLING, "max_tokens": 4096} == rec.applied_sampling
    assert backend.calls[0]["request_id"] == "r"
    assert rec.requested == {"model": "qwen3-8b", "stream": True, "stream_options": {"include_usage": True}, **pi_fields}
    assert rec.response_ids == qwen3_script(tok, NORMAL)  # Pi's max_tokens=5 was not applied


def test_routing_errors_and_collect(tok):
    backend = FakeBackend(qwen3_script(tok, NORMAL))

    async def go(proxy, s):
        url = proxy.rollout_base_url
        res = {}
        async with s.post(f"{url('nope')}/chat/completions", json=_body(False)) as r:
            res["unknown"] = r.status
        async with s.get(f"{url('nope')}/models") as r:
            res["unknown_models"] = r.status
        proxy.register_rollout("a", SAMPLING, 64)
        with pytest.raises(ValueError):
            proxy.register_rollout("a", SAMPLING, 64)
        async with s.get(f"{url('a')}/models") as r:
            res["models"] = (r.status, (await r.json())["data"][0]["id"])
        async with s.post(f"{url('a')}/chat/completions", json=_body(False), headers={"X-Rollout-Id": "b"}) as r:
            res["mismatch"] = r.status
        async with s.post(f"{url('a')}/chat/completions", json=_body(False), headers={"X-Rollout-Id": "a"}) as r:
            res["match"] = r.status
        bad = {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]}
        async with s.post(f"{url('a')}/chat/completions", json=bad) as r:
            res["bad_part"] = (r.status, (await r.json())["error"]["code"])
        res["recs"] = proxy.collect("a")
        res["again"] = proxy.collect("a")
        async with s.post(f"{url('a')}/chat/completions", json=_body(False)) as r:
            res["after_collect"] = r.status
        return res

    res = run(_with_proxy(tok, backend, go))
    assert res["unknown"] == res["unknown_models"] == 404 and res["after_collect"] == 404
    assert res["mismatch"] == 400 and res["match"] == 200 and res["models"] == (200, "qwen3-8b")
    assert res["bad_part"] == (400, 400)
    assert [(r.call_index, r.finish_reason, r.error is None) for r in res["recs"]] == [(0, "stop", True), (1, "error", False)]
    assert res["again"] == []
    assert len(backend.calls) == 1  # unknown, mismatched and bad requests never reached the engine


@pytest.mark.parametrize("stream", [False, True])
def test_engine_abort_is_an_error(tok, stream):
    async def go(proxy, s):
        proxy.register_rollout("r", SAMPLING, 64)
        async with s.post(f"{proxy.rollout_base_url('r')}/chat/completions", json=_body(stream)) as r:
            return r.status, await r.text(), proxy.collect("r")

    status, text, (rec,) = run(_with_proxy(tok, FakeBackend(GenOutput([], stop_reason="aborted")), go))
    if stream:  # headers already went out: the error arrives as an SSE data event
        assert status == 200 and json.loads(text.strip()[len("data: "):])["error"]["code"] == 500
    else:
        assert status == 500 and json.loads(text)["error"]["type"] == "server_error"
    assert rec.finish_reason == "error" and "aborted" in rec.error and rec.engine_meta["stop_reason"] == "aborted"


def test_backend_exception_is_recorded(tok):
    def boom(*a):
        raise RuntimeError("engine down")

    async def go(proxy, s):
        proxy.register_rollout("r", SAMPLING, 64)
        async with s.post(f"{proxy.rollout_base_url('r')}/chat/completions", json=_body(False)) as r:
            return r.status, proxy.collect("r")

    status, (rec,) = run(_with_proxy(tok, FakeBackend(boom), go))
    assert status == 500 and "engine down" in rec.error and rec.prompt_ids


def test_keepalive_comments_while_generating(tok):
    async def go(proxy, s):
        proxy.register_rollout("r", SAMPLING, 64)
        t0 = time.time()
        async with s.post(f"{proxy.rollout_base_url('r')}/chat/completions", json=_body(True)) as r:
            first = await r.content.readline()
            t_first = time.time() - t0
            rest = await r.text()
        return first, t_first, rest

    first, t_first, rest = run(_with_proxy(tok, FakeBackend(qwen3_script(tok, NORMAL), delay_s=1.1), go, keepalive_s=0.25))
    assert t_first < 0.5 and first == b": keepalive\n"
    chunks, comments = parse_sse(first.decode() + rest)
    assert len(comments) >= 3 and reassemble(chunks)["finish_reason"] == "stop"


def test_200_concurrent_rollouts(tok):
    n = 200
    scripts = {f"rid{i}": qwen3_script(tok, f"<think>\nrollout rid{i}\n</think>\n\nAnswer: rid{i}") for i in range(n)}
    backend = FakeBackend(lambda rid, k, p, sp: scripts[rid], delay_s=0.5)

    async def one(proxy, s, i):
        rid = f"rid{i}"
        proxy.register_rollout(rid, SAMPLING, 64)
        body = {"messages": [{"role": "user", "content": f"question for {rid}"}], "stream": i % 2 == 0}
        async with s.post(f"{proxy.rollout_base_url(rid)}/chat/completions", json=body) as r:
            assert r.status == 200
            await r.read()
        return rid, proxy.collect(rid)

    async def go(proxy, s):
        t0 = time.time()
        res = await asyncio.gather(*(one(proxy, s, i) for i in range(n)))
        return res, time.time() - t0

    res, dt = run(_with_proxy(tok, backend, go))
    assert dt < 20, dt  # 100 s if serialized
    for rid, recs in res:
        assert len(recs) == 1
        (rec,) = recs
        assert rec.rollout_id == rid and rec.response_ids == scripts[rid]
        assert f"question for {rid}<|im_end|>" in rec.rendered_prompt and rec.content_raw == f"\n\nAnswer: {rid}"
        assert rec.timings["t_recv"] <= rec.timings["t_render_done"] <= rec.timings["t_gen_start"] <= rec.timings["t_gen_end"] <= rec.timings["t_resp_done"]


# ---------------- vLLM HTTP backend (against a stub with vLLM 0.29's /v1/completions response shape) ----------------

def test_vllm_http_backend(tok):
    from aiohttp import web
    seen = {}

    async def completions(request):
        b = await request.json()
        seen.update(b)
        ids = [11, 22, 151645]
        return web.json_response({"id": "cmpl-1", "object": "text_completion", "choices": [{
            "index": 0, "text": "xx", "finish_reason": "stop", "stop_reason": None, "token_ids": ids,
            "logprobs": {"tokens": [f"token_id:{t}" for t in ids], "token_logprobs": [-0.1, -0.2, -0.3],
                         "top_logprobs": None, "text_offset": [0, 1, 2]}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6}})

    async def go():
        app = web.Application()
        app.router.add_post("/v1/completions", completions)
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        be = VLLMHTTPBackend(f"http://127.0.0.1:{runner.addresses[0][1]}", "Qwen/Qwen3-8B")
        try:
            return await be.generate("r", [1, 2, 3], {**SAMPLING, "max_tokens": 16})
        finally:
            await be.close()
            await runner.cleanup()

    out = run(go())
    assert out.token_ids == [11, 22, 151645] and out.log_probs == [-0.1, -0.2, -0.3] and out.stop_reason == "completed"
    assert seen["prompt"] == [1, 2, 3] and seen["return_token_ids"] is True and seen["logprobs"] == 0
    assert seen["max_tokens"] == 16 and seen["temperature"] == 1.0 and seen["top_k"] == -1 and seen["stream"] is False


# ---------------- trace schema ----------------

def test_trace_round_trip(tmp_path):
    rec = CallRecord(rollout_id="r", call_index=0, prompt_ids=[1, 2], response_ids=[3, 151645], response_logprobs=[-0.5, -0.1],
                     rendered_prompt="<|im_start|>user\nQ<|im_end|>\n", messages_in=[{"role": "user", "content": "Q"}],
                     requested={"stream": True}, applied_sampling={"temperature": 1.0, "max_tokens": 8}, finish_reason="stop",
                     reasoning_raw="<think>\nR\n</think>", content_raw="\n\nA", timings={"t_recv": 1.5}, engine_meta={"num_preempted": 0})
    tr = RolloutTrace("r", [rec, CallRecord("r", 1)], harness_version="h", adapter_version="a", eval_set_hash="e",
                      checkpoint_id="c", seed=3, task_id="t", extra={"k": [1]})
    assert RolloutTrace.from_json(tr.to_json()) == tr
    p = tmp_path / "t.jsonl"
    assert write_jsonl(p, [tr, tr]) == 2 and list(read_jsonl(p)) == [tr, tr]
    assert rec.decoded_text == "<think>\nR\n</think>\n\nA"


def test_verl_backend_calls_server_manager_like_single_turn_loop(tok):
    from verl.workers.rollout.replica import TokenOutput

    class FakeServerManager:  # stands in for verl's LLMServerClient
        def __init__(self):
            self.calls = []

        async def generate(self, request_id, *, prompt_ids, sampling_params, **kw):
            self.calls.append((request_id, prompt_ids, sampling_params, kw))
            sampling_params.pop("max_tokens")  # vLLM server pops it; the proxy's copy must not care
            return TokenOutput(token_ids=[16141, 151645], log_probs=[-0.1, -0.2], stop_reason="completed", num_preempted=0,
                               extra_fields={"global_steps": 3, "response_topk_ids": object()})

    sm = FakeServerManager()

    async def go(proxy, s):
        proxy.register_rollout("rid-7", SAMPLING, 512, {"enable_thinking": False})
        async with s.post(f"{proxy.rollout_base_url('rid-7')}/chat/completions", json=_body(False)) as r:
            return await r.json(), proxy.collect("rid-7")

    j, (rec,) = run(_with_proxy(tok, VerlBackend(sm, priority=2), go))
    request_id, prompt_ids, sp, kw = sm.calls[0]
    assert request_id == "rid-7" and prompt_ids == rec.prompt_ids and kw == {"priority": 2}
    assert rec.applied_sampling == {**SAMPLING, "max_tokens": 512}
    assert rec.response_ids == [16141, 151645] and rec.response_logprobs == [-0.1, -0.2]
    assert rec.engine_meta == {"stop_reason": "completed", "num_preempted": 0, "global_steps": 3}
    assert j["choices"][0]["message"]["content"] == "Answer" and j["choices"][0]["finish_reason"] == "stop"
