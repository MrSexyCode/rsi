"""0.2 spike findings as tests: what the pinned Pi sends, and that nothing outside the harness leaks in.

Each test runs the real pinned Pi against the logging stub in harness_rl/spike/stub_server.py.
"""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from harness_rl.core.env_paths import PI_BIN

SPIKE = Path(__file__).resolve().parents[1] / "spike"
LOCKDOWN = ["--no-session", "-ne", "-ns", "-np", "--no-themes", "-nc", "-na", "--no-mcp", "-nt", "--offline"]
FORCE_EXT = 'export default function (pi: any) { pi.on("before_agent_start", () => ({ systemPrompt: process.env.FORCE_SYSTEM_PROMPT ?? "" })); }\n'

pytestmark = pytest.mark.skipif(not PI_BIN.exists(), reason="pinned Pi not installed")


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def stub(tmp_path):
    port, log = _free_port(), tmp_path / "req.jsonl"
    proc = subprocess.Popen([sys.executable, str(SPIKE / "stub_server.py"), "--port", str(port), "--log", str(log),
                             "--reasoning", "Let me think. B.", "--content", "Answer: B"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.1).close()
            break
        except OSError:
            time.sleep(0.05)
    yield port, log
    proc.terminate()
    proc.wait()


def _agent_dir(home: Path, base_url: str):
    agent = home / ".pi/agent"
    agent.mkdir(parents=True, exist_ok=True)
    (agent / "models.json").write_text(json.dumps({"providers": {"rlproxy": {
        "baseUrl": base_url, "api": "openai-completions", "apiKey": "${ROLLOUT_ID}",
        "headers": {"X-Rollout-Id": "${ROLLOUT_ID}"},
        "compat": {"supportsDeveloperRole": False, "supportsStore": False},
        "models": [{"id": "qwen3-8b", "reasoning": True, "contextWindow": 40960, "maxTokens": 8192}]}}}))
    return agent


def run_pi(tmp_path, port, prompt, extra=(), env_extra=None, base_url=None, cwd=None, home=None):
    home = home or tmp_path / "home"
    _agent_dir(home, base_url or f"http://127.0.0.1:{port}/v1")
    ext = tmp_path / "force_prompt.ts"
    ext.write_text(FORCE_EXT)
    cwd = cwd or tmp_path / "cwd"
    cwd.mkdir(parents=True, exist_ok=True)
    env = {"PATH": "/usr/bin:/bin", "HOME": str(home), "PI_OFFLINE": "1", "PI_TELEMETRY": "0", "ROLLOUT_ID": "rid-1",
           **(env_extra or {})}
    cmd = [str(PI_BIN), "-p", "--mode", "json", *LOCKDOWN, "-e", str(ext), "--model", "rlproxy/qwen3-8b", *extra]
    r = subprocess.run(cmd, input=prompt, cwd=cwd, env=env, capture_output=True, text=True, timeout=60)
    events = [json.loads(l) for l in r.stdout.splitlines() if l.strip()]
    return r, events


def requests(log):
    return [json.loads(l) for l in open(log)] if log.exists() else []


def final(events):
    return [e["message"] for e in events if e.get("type") == "message_end" and e["message"]["role"] == "assistant"][-1]


def test_single_call_wire_format(tmp_path, stub):
    port, log = stub
    r, events = run_pi(tmp_path, port, "What is 2+2?")
    assert r.returncode == 0
    reqs = requests(log)
    assert len(reqs) == 1
    body = reqs[0]["body"]
    assert reqs[0]["path"] == "/v1/chat/completions"
    assert reqs[0]["headers"]["x-rollout-id"] == "rid-1"
    assert body["stream"] is True and body["max_completion_tokens"] == 8192
    assert "tools" not in body and "temperature" not in body and "store" not in body
    assert body["reasoning_effort"] == "medium"  # default thinking level; the proxy ignores it
    assert body["messages"] == [{"role": "user", "content": [{"type": "text", "text": "What is 2+2?"}]}]
    msg = final(events)
    assert msg["stopReason"] == "stop"
    assert msg["content"] == [{"type": "thinking", "thinking": "Let me think. B.", "thinkingSignature": "reasoning_content"},
                              {"type": "text", "text": "Answer: B"}]


def test_forced_system_prompt_is_exact_and_uses_system_role(tmp_path, stub):
    port, log = stub
    run_pi(tmp_path, port, "Q", env_extra={"FORCE_SYSTEM_PROMPT": "You are a careful reader.\n\nBe brief."})
    msgs = requests(log)[0]["body"]["messages"]
    assert msgs[0] == {"role": "system", "content": "You are a careful reader.\n\nBe brief."}


def test_thinking_off_drops_reasoning_effort(tmp_path, stub):
    port, log = stub
    run_pi(tmp_path, port, "Q", extra=["--thinking", "off"])
    assert "reasoning_effort" not in requests(log)[0]["body"]


@pytest.mark.parametrize("prompt", [
    "Answer the following multiple-choice question.\n\nOptions:\nA. x\nB. y\n\nAnswer: <the letter>",
    "@notafile.txt starts with at", "- starts with dash", "/model looks like a command",
    "Ünïcödé — 中文 ​\t tab", "x" * 300_000,
], ids=["omanic", "at", "dash", "slash", "unicode", "300k"])
def test_stdin_prompt_is_byte_exact(tmp_path, stub, prompt):
    port, log = stub
    r, _ = run_pi(tmp_path, port, prompt)
    assert r.returncode == 0, r.stderr
    assert requests(log)[0]["body"]["messages"][-1]["content"] == [{"type": "text", "text": prompt}]


def test_stdin_prompt_is_stripped(tmp_path, stub):
    """Pi trims surrounding whitespace of a stdin prompt; PiAgentLoop must only send stripped prompts."""
    port, log = stub
    run_pi(tmp_path, port, "  padded  \n\n")
    assert requests(log)[0]["body"]["messages"][-1]["content"][0]["text"] == "padded"


def test_base_url_is_not_env_interpolated(tmp_path, stub):
    port, log = stub
    r, events = run_pi(tmp_path, port, "Q", base_url="${PROXY_BASE_URL}", env_extra={"PROXY_BASE_URL": f"http://127.0.0.1:{port}/v1"})
    assert requests(log) == []
    assert r.returncode == 0  # JSON mode exits 0 on provider errors: read stopReason instead
    assert final(events)["stopReason"] == "error" and final(events)["errorMessage"] == "Invalid URL"


def test_missing_rollout_id_fails_loudly(tmp_path, stub):
    port, log = stub
    r, events = run_pi(tmp_path, port, "Q", env_extra={"ROLLOUT_ID": ""})
    assert requests(log) == [] and r.returncode != 0 and "No API key" in r.stderr


def test_nothing_leaks_from_cwd_parents_or_agent_dir(tmp_path, stub):
    port, log = stub
    parent = tmp_path / "parent"
    cwd = parent / "cwd"
    home = tmp_path / "home"
    agent = _agent_dir(home, f"http://127.0.0.1:{port}/v1")
    for d in (parent, cwd):
        (d / ".pi/extensions").mkdir(parents=True)
        (d / ".pi/skills/s").mkdir(parents=True)
        for name in ["AGENTS.md", "CLAUDE.md", ".pi/SYSTEM.md", ".pi/APPEND_SYSTEM.md"]:
            (d / name).write_text(f"LEAK_{d.name}_{name}")
        (d / ".pi/extensions/leak.ts").write_text(
            'export default function (pi: any) { pi.on("before_agent_start", (e: any) => { e.systemPromptOptions.sections.x = "LEAK_EXT"; }); }')
        (d / ".pi/skills/s/SKILL.md").write_text("---\nname: s\ndescription: LEAK_SKILL\n---\nbody")
        (d / ".pi/settings.json").write_text('{"defaultThinkingLevel": "high"}')
    for name in ["SYSTEM.md", "APPEND_SYSTEM.md", "AGENTS.md"]:
        (agent / name).write_text(f"LEAK_AGENTDIR_{name}")
    (agent / "extensions").mkdir()
    (agent / "extensions/leak.ts").write_text(
        'export default function (pi: any) { pi.on("before_agent_start", (e: any) => { e.systemPromptOptions.sections.y = "LEAK_AGENTEXT"; }); }')
    run_pi(tmp_path, port, "Q", cwd=cwd, home=home)
    raw = log.read_text()
    assert "LEAK" not in raw
    body = requests(log)[0]["body"]
    assert body["messages"] == [{"role": "user", "content": [{"type": "text", "text": "Q"}]}]
    assert body["reasoning_effort"] == "medium"  # project settings.json ("high") ignored


def test_reasoning_is_replayed_on_later_turns(tmp_path, stub):
    """Second positional message = second turn. Pi sends the earlier reasoning back as reasoning_content."""
    port, log = stub
    home = tmp_path / "home"
    _agent_dir(home, f"http://127.0.0.1:{port}/v1")
    (tmp_path / "cwd").mkdir()
    (tmp_path / "f.ts").write_text(FORCE_EXT)
    env = {"PATH": "/usr/bin:/bin", "HOME": str(home), "PI_OFFLINE": "1", "PI_TELEMETRY": "0", "ROLLOUT_ID": "r"}
    subprocess.run([str(PI_BIN), "-p", "--mode", "json", *LOCKDOWN, "-e", str(tmp_path / "f.ts"),
                    "--model", "rlproxy/qwen3-8b", "Q1", "Q2"], cwd=tmp_path / "cwd", env=env,
                   stdin=subprocess.DEVNULL, capture_output=True, timeout=60)
    reqs = requests(log)
    assert len(reqs) == 2
    assert reqs[1]["body"]["messages"][1] == {"role": "assistant", "content": "Answer: B", "reasoning_content": "Let me think. B."}
