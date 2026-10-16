"""0.5 harness packages, content hashing, and the JSON-subprocess Pi runner against the logging stub."""

import asyncio
import json
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from harness_rl.core.env_paths import PI_BIN
from harness_rl.core.versioning import content_hash
from harness_rl.pi_loop.runner import HarnessPackage, JsonSubprocessRunner, materialize

SPIKE = Path(__file__).resolve().parents[1] / "spike"
OMANIC_PROMPT = ("Answer the following multiple-choice question. Answering it requires several reasoning steps.\n\n"
                 "Question: Who?\n\nOptions:\nA. x\nB. y\nC. z\nD. w\n\n"
                 "Reason step by step. Then end your response with exactly this line:\n"
                 "Answer: <the letter of the correct option>")


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


def requests(log):
    return [json.loads(l) for l in open(log)] if log.exists() else []


# ---- content hash ----

def test_content_hash_is_deterministic_and_location_independent(tmp_path):
    a = tmp_path / "a"
    shutil.copytree(HarnessPackage.load("vanilla").root, a)
    b = tmp_path / "elsewhere" / "b"
    shutil.copytree(a, b)
    assert content_hash(a) == content_hash(b) == HarnessPackage.load("vanilla").version


@pytest.mark.parametrize("change", ["edit", "add", "rename", "empty_file_added"])
def test_content_hash_changes_with_content(tmp_path, change):
    pkg = tmp_path / "p"
    shutil.copytree(HarnessPackage.load("vanilla").root, pkg)
    before = content_hash(pkg)
    if change == "edit":
        (pkg / "system_prompt.md").write_text("x")
    elif change == "add":
        (pkg / "notes.txt").write_text("x")
    elif change == "rename":
        (pkg / "settings.json").rename(pkg / "settings2.json")
    else:
        (pkg / "empty").write_text("")
    assert content_hash(pkg) != before


def test_content_hash_ignores_caches(tmp_path):
    pkg = tmp_path / "p"
    shutil.copytree(HarnessPackage.load("vanilla").root, pkg)
    before = content_hash(pkg)
    (pkg / "__pycache__").mkdir()
    (pkg / "__pycache__" / "x.pyc").write_bytes(b"junk")
    assert content_hash(pkg) == before


def test_packages_differ_and_load():
    v, d = HarnessPackage.load("vanilla"), HarnessPackage.load("pi-default")
    assert v.name == "vanilla" and d.name == "pi-default" and v.version != d.version
    assert len(v.version) == 64


def test_materialize_writes_literal_base_url(tmp_path):
    home, cwd = materialize(HarnessPackage.load("vanilla"), tmp_path / "r", "http://127.0.0.1:1/r/abc/v1")
    models = json.loads((home / ".pi/agent/models.json").read_text())
    assert models["providers"]["rlproxy"]["baseUrl"] == "http://127.0.0.1:1/r/abc/v1"
    assert json.loads((home / ".pi/agent/settings.json").read_text())["retry"]["enabled"] is False
    assert list(cwd.iterdir()) == []


# ---- runner against the stub ----

pi_required = pytest.mark.skipif(not PI_BIN.exists(), reason="pinned Pi not installed")


@pi_required
def test_vanilla_sends_only_the_user_message(tmp_path, stub):
    port, log = stub
    runner = JsonSubprocessRunner(HarnessPackage.load("vanilla"), work_root=tmp_path)
    res = asyncio.run(runner.run(OMANIC_PROMPT, "rid-v", f"http://127.0.0.1:{port}/v1"))
    assert res.ok, res.error
    assert res.text == "Answer: B" and res.thinking == "Let me think. B." and res.assistant_messages == 1
    reqs = requests(log)
    assert len(reqs) == 1
    body = reqs[0]["body"]
    assert body["messages"] == [{"role": "user", "content": [{"type": "text", "text": OMANIC_PROMPT}]}]
    assert "tools" not in body and "reasoning_effort" not in body
    assert reqs[0]["headers"]["x-rollout-id"] == "rid-v"
    assert list((tmp_path / "harness_rl_rollouts").iterdir()) == []  # rollout dir cleaned up


@pi_required
def test_pi_default_prompt_is_identical_across_rollouts(tmp_path, stub):
    port, log = stub
    runner = JsonSubprocessRunner(HarnessPackage.load("pi-default"), work_root=tmp_path)

    async def two():
        return await asyncio.gather(*(runner.run("Q", f"rid-{i}", f"http://127.0.0.1:{port}/v1") for i in range(2)))

    assert all(r.ok for r in asyncio.run(two()))
    systems = [r["body"]["messages"][0] for r in requests(log)]
    assert len(systems) == 2 and systems[0] == systems[1] and systems[0]["role"] == "system"
    assert systems[0]["content"].startswith("You are an expert coding assistant operating inside pi")
    assert systems[0]["content"].endswith("<cwd>\n/workspace\n</cwd>")
    assert "<tools>\n(none)" in systems[0]["content"]


@pi_required
def test_provider_failure_is_not_ok_even_though_pi_exits_zero(tmp_path):
    runner = JsonSubprocessRunner(HarnessPackage.load("vanilla"), work_root=tmp_path)
    res = asyncio.run(runner.run("Q", "rid-x", f"http://127.0.0.1:{_free_port()}/v1"))  # nothing listening
    assert not res.ok and res.returncode == 0 and res.final["stopReason"] == "error"


@pi_required
def test_unstripped_prompt_is_rejected(tmp_path):
    runner = JsonSubprocessRunner(HarnessPackage.load("vanilla"), work_root=tmp_path)
    with pytest.raises(ValueError):
        asyncio.run(runner.run(" Q\n", "rid", "http://127.0.0.1:1/v1"))
