"""verl reward for progout tasks: exact match of the "Answer:" line against the program's printed output.

The task is rebuilt from extra_info["spec"] (a dict or its JSON), so the reward follows the adapter, not a copy of
the answer; the program runs once per spec per process (sandbox cache). A ground_truth that disagrees with the
rebuilt answer means the data was built with another adapter version, and raises.
Hook up with: reward.custom_reward_function.path=<this file> reward.custom_reward_function.name=compute_score
"""

import json

from harness_rl.adapters.progout import ADAPTER
from harness_rl.core.adapter import rng_for_spec
from harness_rl.evals.scoring import extract_answer


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> dict:
    spec = extra_info["spec"]
    spec = json.loads(spec) if isinstance(spec, str) else dict(spec)
    task = ADAPTER.instantiate(spec, rng_for_spec(spec))
    if ground_truth and ground_truth != task.answer:
        raise ValueError(f"ground_truth {ground_truth!r} != rebuilt answer {task.answer!r} for {task.task_id}")
    pred = extract_answer(solution_str)
    if pred is None:
        return {"score": 0.0, "em": 0.0, "em_nospace": 0.0, "format_ok": 0.0}
    r = ADAPTER.verify(task, pred)
    return {"score": r.score, **r.components, "format_ok": 1.0}
