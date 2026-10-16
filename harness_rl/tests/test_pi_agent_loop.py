"""0.4: PiAgentLoop vs verl's SingleTurnAgentLoop on the same deterministic fake engine.

Both loops get the same `server_manager` stand-in, whose output is a deterministic function of the prompt ids. With the
vanilla harness the Pi path must hand verl exactly what the direct path does: same prompt_ids, response_ids,
response_mask, logprobs, and therefore the same reward. Also covers the invalid-rollout contract.
"""

import asyncio
import importlib.util
import json

import numpy as np
import pytest

pytest.importorskip("verl")
from omegaconf import OmegaConf  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from verl.experimental.agent_loop.agent_loop import DictConfigWrap  # noqa: E402
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop  # noqa: E402
from verl.workers.rollout.replica import TokenOutput  # noqa: E402

from harness_rl.core.env_paths import MODEL_ID, OMANIC_REWARD, PI_BIN  # noqa: E402
from harness_rl.pi_loop import agent_loop as pal  # noqa: E402

pytestmark = pytest.mark.skipif(not PI_BIN.exists(), reason="pinned Pi not installed")

PROMPT = ("Answer the following multiple-choice question. Answering it requires several reasoning steps.\n\n"
          "Question: Which city hosted it?\n\nOptions:\nA. Oslo\nB. Bergen\nC. Turku\nD. Malmo\n\n"
          "Reason step by step. Then end your response with exactly this line:\n"
          "Answer: <the letter of the correct option>")
RESPONSE_LENGTH = 96


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(MODEL_ID)


@pytest.fixture(scope="module")
def omanic_reward():
    spec = importlib.util.spec_from_file_location("omanic_reward", OMANIC_REWARD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeServerManager:
    """Stands in for verl's LLMServerClient: output ids depend only on the prompt ids (and the script mode)."""

    def __init__(self, tok, mode="normal"):
        self.tok, self.mode, self.calls = tok, mode, []

    async def generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):
        self.calls.append({"request_id": request_id, "prompt_ids": list(prompt_ids), "sampling_params": dict(sampling_params)})
        n = len(prompt_ids)
        if self.mode == "normal":
            text = f"<think>\nThe prompt has {n} tokens; Bergen fits.\n</think>\n\nReasoning done.\nAnswer: B"
            ids = self.tok.encode(text, add_special_tokens=False) + [self.tok.convert_tokens_to_ids("<|im_end|>")]
        else:  # "truncated": unclosed think with an Answer line inside, longer than the budget
            text = "<think>\nAnswer: B\n" + "and so on " * 200
            ids = self.tok.encode(text, add_special_tokens=False)
        max_tokens = sampling_params.get("max_tokens", RESPONSE_LENGTH)
        ids = ids[:max_tokens]
        logprobs = [-((i * 7919 + n) % 97) / 100.0 for i in range(len(ids))]
        tags = {"global_steps": np.int64(7), "min_global_steps": np.int64(7), "max_global_steps": np.int64(7),
                "num_cached_tokens": 0}  # what LLMServerClient adds; the trainer's metrics require them
        return TokenOutput(token_ids=ids, log_probs=logprobs, stop_reason="completed", num_preempted=0, extra_fields=tags)


def make_config(harness="vanilla"):
    return OmegaConf.create({
        "actor_rollout_ref": {"rollout": {"prompt_length": 1024, "response_length": RESPONSE_LENGTH}},
        "data": {"apply_chat_template_kwargs": {"enable_thinking": True}},
        "harness_rl": {"harness": harness},
    })


def make_loop(cls, tok, sm, config):
    return cls(trainer_config=DictConfigWrap(config), server_manager=sm, tokenizer=tok, processor=None,
               dataset_cls=None, data_config=DictConfigWrap(config.data), hf_model_type="qwen3")


SAMPLING = {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "repetition_penalty": 1.0, "logprobs": True}


async def _run_both(tok, mode, harness="vanilla"):
    pal._WORKER_STATE.clear()
    config = make_config(harness)
    sm_direct, sm_pi = FakeServerManager(tok, mode), FakeServerManager(tok, mode)
    kwargs = {"raw_prompt": [{"role": "user", "content": PROMPT}], "extra_info": {"index": 3}, "data_source": "omanic"}
    direct = await make_loop(SingleTurnAgentLoop, tok, sm_direct, config).run(dict(SAMPLING), **kwargs)
    pi = await make_loop(pal.PiAgentLoop, tok, sm_pi, config).run(dict(SAMPLING), **kwargs)
    for st in pal._WORKER_STATE.values():
        await st["proxy"].stop()
    pal._WORKER_STATE.clear()
    return direct, pi, sm_direct, sm_pi


@pytest.mark.parametrize("mode", ["normal", "truncated"])
def test_pi_vanilla_matches_direct_path_exactly(tok, omanic_reward, mode):
    direct, pi, sm_direct, sm_pi = asyncio.run(_run_both(tok, mode))
    assert pi.extra_fields["pi_valid"] == 1, pi.extra_fields["pi_problems"]
    assert pi.prompt_ids == direct.prompt_ids
    assert pi.response_ids == direct.response_ids
    assert pi.response_mask == direct.response_mask == [1] * len(direct.response_ids)
    assert pi.response_logprobs == direct.response_logprobs
    assert pi.num_turns == direct.num_turns == 2
    for k in ("global_steps", "min_global_steps", "max_global_steps"):
        assert pi.extra_fields[k] == direct.extra_fields[k] == 7
    # The engine saw the same prompt and the sampling params from verl (Pi's request fields are ignored).
    assert sm_pi.calls[0]["prompt_ids"] == sm_direct.calls[0]["prompt_ids"]
    assert {k: sm_pi.calls[0]["sampling_params"][k] for k in SAMPLING} == SAMPLING
    assert sm_pi.calls[0]["sampling_params"]["max_tokens"] == RESPONSE_LENGTH
    # Same reward through the same decode as verl's reward manager.
    gt = json.dumps({"label": "B", "answer": "Bergen"})
    score = lambda o: omanic_reward.compute_score("omanic", tok.decode(o.response_ids, skip_special_tokens=True), gt)
    assert score(pi) == score(direct)
    if mode == "truncated":
        assert len(pi.response_ids) == RESPONSE_LENGTH


def test_pi_default_harness_changes_only_the_prompt(tok):
    direct, pi, _, _ = asyncio.run(_run_both(tok, "normal", harness="pi-default"))
    assert pi.extra_fields["pi_valid"] == 1
    assert len(pi.prompt_ids) > len(direct.prompt_ids)
    text, direct_text = tok.decode(pi.prompt_ids), tok.decode(direct.prompt_ids)
    system_block, rest = text.split("<|im_end|>\n", 1)
    assert system_block.startswith("<|im_start|>system\nYou are an expert coding assistant operating inside pi")
    assert system_block.endswith("<cwd>\n/workspace\n</cwd>")
    assert rest == direct_text


class FailingServerManager(FakeServerManager):
    async def generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):
        return TokenOutput(token_ids=[], log_probs=[], stop_reason="aborted", num_preempted=0)


def test_engine_failure_gives_invalid_masked_rollout(tok):
    async def go():
        pal._WORKER_STATE.clear()
        loop = make_loop(pal.PiAgentLoop, tok, FailingServerManager(tok), make_config())
        out = await loop.run(dict(SAMPLING), raw_prompt=[{"role": "user", "content": PROMPT}], extra_info={})
        for st in pal._WORKER_STATE.values():
            await st["proxy"].stop()
        pal._WORKER_STATE.clear()
        return out

    out = asyncio.run(go())
    assert out.extra_fields["pi_valid"] == 0 and out.extra_fields["pi_problems"]
    assert out.response_ids == [tok.convert_tokens_to_ids("<|im_end|>")] and out.response_mask == [0]
    assert tok.decode(out.response_ids, skip_special_tokens=True) == ""
    assert isinstance(out.extra_fields["min_global_steps"], int)  # never None, or the trainer's metrics crash
