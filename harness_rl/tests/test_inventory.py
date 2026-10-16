"""Pins the facts recorded in docs/inventory.md, so drift in the wrapped environment fails loudly."""

import importlib.util
import json

import pytest

from harness_rl.core.env_paths import MODEL_ID, OMANIC_DATA, OMANIC_REWARD, PI_BIN


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained(MODEL_ID)


@pytest.fixture(scope="module")
def omanic_reward():
    spec = importlib.util.spec_from_file_location("omanic_reward", OMANIC_REWARD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_qwen3_special_tokens(tokenizer):
    ids = {s: tokenizer.convert_tokens_to_ids(s) for s in ["<think>", "</think>", "<|im_end|>"]}
    assert ids == {"<think>": 151667, "</think>": 151668, "<|im_end|>": 151645}
    assert 151667 not in tokenizer.all_special_ids and 151668 not in tokenizer.all_special_ids


def test_thinking_generation_prompt(tokenizer):
    msgs = [{"role": "user", "content": "Q?"}]
    on = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    assert on == "<|im_start|>user\nQ?<|im_end|>\n<|im_start|>assistant\n"


def test_think_tags_survive_reward_decoding(tokenizer, omanic_reward):
    text = "<think>\nAnswer: A\n</think>\n\nAnswer: B<|im_end|>"
    ids = tokenizer.encode(text, add_special_tokens=False)
    decoded = tokenizer.decode(ids, skip_special_tokens=True)
    assert decoded == "<think>\nAnswer: A\n</think>\n\nAnswer: B"
    assert omanic_reward.parse(decoded) == "B"


def test_omanic_reward_contract(omanic_reward):
    gt = json.dumps({"label": "B", "answer": "x"})
    assert omanic_reward.compute_score("omanic", "Answer: B", gt)["score"] == 1.0
    assert omanic_reward.compute_score("omanic", "Answer: B or C", gt)["format_ok"] == 0.0


def test_omanic_parquet_present():
    for name in ["train", "test", "test_shuffled", "val"]:
        assert (OMANIC_DATA / f"{name}.parquet").exists()


def test_pinned_pi_installed():
    lock = json.loads((PI_BIN.parents[2] / "package-lock.json").read_text())
    assert lock["packages"]["node_modules/@earendil-works/pi-coding-agent"]["version"] == "1.2.0"
