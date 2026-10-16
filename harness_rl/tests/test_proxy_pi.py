"""0.3 end to end: the real pinned Pi, isolated as in test_pi_spike.py, talking to the ModelProxy with a FakeBackend.

Checks the exact tokens captured for an Omanic prompt, how Pi sees reasoning/content/finish_reason, that SSE
keepalive comments keep Pi's HTTP idle timeout from firing during a long generation, and that an engine failure
after the SSE headers are out reaches Pi as an error.
"""

import asyncio
import json
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from harness_rl.core.env_paths import MODEL_ID, OMANIC_DATA, PI_BIN
from harness_rl.proxy import FakeBackend, GenOutput, ModelProxy, qwen3_script
from harness_rl.proxy.render import normalize_ids

pytestmark = pytest.mark.skipif(not PI_BIN.exists(), reason="pinned Pi not installed")

LOCKDOWN = ["--no-session", "-ne", "-ns", "-np", "--no-themes", "-nc", "-na", "--no-mcp", "-nt", "--offline"]
FORCE_EXT = 'export default function (pi: any) { pi.on("before_agent_start", () => ({ systemPrompt: process.env.FORCE_SYSTEM_PROMPT ?? "" })); }\n'
SAMPLING = {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "repetition_penalty": 1.0, "logprobs": True}
NORMAL = "<think>\nLet me think. Option B fits.\n</think>\n\nThe answer follows.\nAnswer: B"


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(MODEL_ID)


@pytest.fixture(scope="module")
def omanic_prompt():
    import pandas as pd
    return pd.read_parquet(OMANIC_DATA / "val.parquet").prompt.iloc[0][0]["content"]


def make_home(root: Path, base_url: str, settings: dict | None = None) -> Path:
    agent = root / "home/.pi/agent"
    agent.mkdir(parents=True)
    (agent / "models.json").write_text(json.dumps({"providers": {"rlproxy": {
        "baseUrl": base_url, "api": "openai-completions", "apiKey": "${ROLLOUT_ID}",
        "headers": {"X-Rollout-Id": "${ROLLOUT_ID}"},
        "compat": {"supportsDeveloperRole": False, "supportsStore": False},
        "models": [{"id": "qwen3-8b", "reasoning": True, "contextWindow": 40960, "maxTokens": 8192}]}}}))
    if settings is not None:
        (agent / "settings.json").write_text(json.dumps(settings))
    return root / "home"


async def run_pi(root: Path, proxy: ModelProxy, rid: str, prompt: str, settings: dict | None = None):
    home = make_home(root, proxy.rollout_base_url(rid), settings)
    ext = root / "force_prompt.ts"
    ext.write_text(FORCE_EXT)
    (root / "cwd").mkdir()
    env = {"PATH": "/usr/bin:/bin", "HOME": str(home), "PI_OFFLINE": "1", "PI_TELEMETRY": "0", "ROLLOUT_ID": rid}
    p = await asyncio.create_subprocess_exec(
        str(PI_BIN), "-p", "--mode", "json", *LOCKDOWN, "-e", str(ext), "--model", "rlproxy/qwen3-8b", "--thinking", "off",
        cwd=root / "cwd", env=env, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    out, err = await asyncio.wait_for(p.communicate(prompt.encode()), timeout=90)
    events = [json.loads(l) for l in out.decode().splitlines() if l.strip()]
    finals = [e["message"] for e in events if e.get("type") == "message_end" and e["message"]["role"] == "assistant"]
    return p.returncode, finals[-1] if finals else None, err.decode()


def rollout(tmp_path, tok, backend, prompt, max_tokens=8192, settings=None, **proxy_kw):
    async def go():
        proxy = ModelProxy(tok, backend, **proxy_kw)
        await proxy.start()
        try:
            proxy.register_rollout("rid-1", SAMPLING, max_tokens, {"enable_thinking": True})
            rc, final, err = await run_pi(tmp_path, proxy, "rid-1", prompt, settings)
            return rc, final, err, proxy.collect("rid-1")
        finally:
            await proxy.stop()
    return asyncio.run(go())


def test_pi_omanic_single_call_is_token_exact(tmp_path, tok, omanic_prompt):
    script = qwen3_script(tok, NORMAL)
    rc, final, err, recs = rollout(tmp_path, tok, FakeBackend(script), omanic_prompt)
    assert rc == 0, err
    assert len(recs) == 1
    (rec,) = recs
    assert rec.error is None and rec.finish_reason == "stop"
    assert rec.prompt_ids == normalize_ids(tok.apply_chat_template(
        [{"role": "user", "content": omanic_prompt}], tokenize=True, add_generation_prompt=True, enable_thinking=True))
    assert rec.response_ids == script and len(rec.response_logprobs) == len(script)
    assert rec.messages_in == [{"role": "user", "content": omanic_prompt}]
    assert rec.applied_sampling == {**SAMPLING, "max_tokens": 8192}
    assert rec.requested["stream"] is True and "reasoning_effort" not in rec.requested and "temperature" not in rec.requested
    assert final["stopReason"] == "stop"
    assert final["content"] == [{"type": "thinking", "thinking": "Let me think. Option B fits.", "thinkingSignature": "reasoning_content"},
                                {"type": "text", "text": "The answer follows.\nAnswer: B"}]
    assert final["usage"]["input"] == len(rec.prompt_ids) and final["usage"]["output"] == len(script)


def test_pi_truncated_is_length_with_reasoning_only(tmp_path, tok):
    script = qwen3_script(tok, "<think>\n" + "Consider every option again. " * 30, eos=False)
    rc, final, err, (rec,) = rollout(tmp_path, tok, FakeBackend(script), "Q?", max_tokens=40)
    assert rc == 0, err
    assert rec.finish_reason == "length" and rec.response_ids == script[:40] and rec.content_raw == ""
    assert final["stopReason"] == "length"
    assert [b["type"] for b in final["content"]] == ["thinking"]


def test_pi_survives_long_generation_with_keepalive(tmp_path, tok):
    """Pi's httpIdleTimeoutMs=3000 vs a 5 s generation: keepalive comments every 1 s keep the stream alive."""
    rc, final, err, (rec,) = rollout(tmp_path, tok, FakeBackend(qwen3_script(tok, NORMAL), delay_s=5.0), "Q?",
                                     settings={"httpIdleTimeoutMs": 3000, "retry": {"enabled": False}}, keepalive_s=1.0)
    assert rc == 0, err
    assert final["stopReason"] == "stop" and final["content"][-1]["text"] == "The answer follows.\nAnswer: B"
    assert rec.error is None and "client_disconnected" not in rec.engine_meta


def test_pi_idle_timeout_fires_without_keepalive(tmp_path, tok):
    """Control for the test above: same setup, keepalive longer than the generation, Pi gives up."""
    rc, final, err, (rec,) = rollout(tmp_path, tok, FakeBackend(qwen3_script(tok, NORMAL), delay_s=5.0), "Q?",
                                     settings={"httpIdleTimeoutMs": 3000, "retry": {"enabled": False}}, keepalive_s=60.0)
    assert final["stopReason"] == "error"
    assert rec.engine_meta.get("client_disconnected") is True


def test_pi_sees_engine_abort_as_error(tmp_path, tok):
    """Failure after the SSE headers are out: a `data: {"error": ...}` event. With retries off, exactly one call."""
    rc, final, err, recs = rollout(tmp_path, tok, FakeBackend(GenOutput([], stop_reason="aborted")), "Q?",
                                   settings={"retry": {"enabled": False}})
    assert final["stopReason"] == "error" and "aborted" in final["errorMessage"]
    assert len(recs) == 1 and recs[0].finish_reason == "error" and "aborted" in recs[0].error
