"""Pi (vanilla) and the runner's direct path must send the engine identical prompt ids for every eval set's tasks."""

import asyncio

import aiohttp
import pytest
from transformers import AutoTokenizer

from harness_rl.core.env_paths import MODEL_ID, PI_BIN
from harness_rl.evals.loader import list_sets, load_set
from harness_rl.evals.run import run_direct
from harness_rl.pi_loop.runner import HarnessPackage, JsonSubprocessRunner
from harness_rl.proxy.backends import FakeBackend, qwen3_script
from harness_rl.proxy.server import ModelProxy

pytestmark = pytest.mark.skipif(not PI_BIN.exists(), reason="pinned Pi not installed")
PER_SET = 2


def test_pi_and_direct_prompt_ids_identical_on_every_set():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tasks = [(n, t) for n in list_sets() for t in load_set(n)[:PER_SET]]

    async def go():
        backend = FakeBackend(qwen3_script(tok, "<think>\nok\n</think>\n\nAnswer: x"))
        proxy = ModelProxy(tok, backend)
        await proxy.start()
        runner = JsonSubprocessRunner(HarnessPackage.load("vanilla"))
        sem = asyncio.Semaphore(32)
        mismatches = []
        async with aiohttp.ClientSession() as http:
            async def one(i, name, task):
                async with sem:
                    ids = {}
                    for mode in ("direct", "pi"):
                        rid = f"t{i}{mode}"
                        proxy.register_rollout(rid, {"temperature": 0.6}, 16, {"enable_thinking": True})
                        if mode == "direct":
                            await run_direct(proxy, http, rid, task["prompt"])
                        else:
                            await runner.run(task["prompt"], rid, proxy.rollout_base_url(rid))
                        calls = proxy.collect(rid)
                        assert len(calls) == 1, (name, mode, len(calls))
                        ids[mode] = calls[0].prompt_ids
                    expected = tok.apply_chat_template([{"role": "user", "content": task["prompt"]}], tokenize=True,
                                                       add_generation_prompt=True, enable_thinking=True)
                    if hasattr(expected, "input_ids"):
                        expected = expected["input_ids"]
                    if not (ids["direct"] == ids["pi"] == list(expected)):
                        mismatches.append(name)

            await asyncio.gather(*(one(i, n, t) for i, (n, t) in enumerate(tasks)))
        await proxy.stop()
        return mismatches

    assert asyncio.run(go()) == []
