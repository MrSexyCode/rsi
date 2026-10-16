"""verl reward for the multihop pool: rebuild the Task from extra_info["spec"] and score the extracted answer with
MultihopAdapter.verify (normalized EM; on UNDETERMINED tasks only the abstention string scores).

Hook up with: reward.custom_reward_function.path=<this file> reward.custom_reward_function.name=compute_score
(the repo root must be on PYTHONPATH). Returns numbers only, like playground/omanic/reward.py.
"""

from __future__ import annotations

import json
from functools import lru_cache

from harness_rl.core.adapter import Task, canonical_json
from harness_rl.evals.scoring import extract_answer


@lru_cache(maxsize=8192)
def _task(spec_json: str) -> Task:
    from harness_rl.adapters.multihop import ADAPTER

    return ADAPTER.instantiate(json.loads(spec_json))


def task_for(spec: dict | str) -> Task:
    return _task(spec if isinstance(spec, str) else canonical_json(spec))


def score_response(spec: dict | str, response: str) -> dict:
    """{"score", "em", "abstained", "gold_undetermined", "in_context", "format_ok", "pred"} for one response."""
    from harness_rl.adapters.multihop import ADAPTER

    task = task_for(spec)
    pred = extract_answer(response)
    r = ADAPTER.verify(task, pred)
    return {"score": r.score, **r.components, "format_ok": float(pred is not None), "pred": pred}


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> dict:
    info = extra_info or {}
    out = score_response(info["spec"], solution_str)
    out.pop("pred")
    out["gt_match"] = float(ground_truth == task_for(info["spec"]).meta["answer_text"])
    return out
