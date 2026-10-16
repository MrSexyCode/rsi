"""Scorers for the frozen eval sets. Every scorer takes (task, response_text) and returns a dict with a primary
`score` in [0, 1] plus components; `format_ok` is 1 when an answer line was found.

| scorer          | sets                      | score                                   | components                    |
|-----------------|---------------------------|-----------------------------------------|-------------------------------|
| omanic_letter   | omanic_heldout            | playground/omanic/reward.py, unchanged  | acc, format_ok, has_math_hop  |
| qa_em_f1        | musique, 2wiki, hotpotqa  | max token F1 over gold + aliases        | em, f1, format_ok             |
| babilong_match  | babilong_*                | official compare_answers on the answer  | format_ok                     |
| ruler_vt_recall | ruler_vt_*                | RULER string_match_all (recall)         | all_match, exact_set, ...     |

Answers are taken from the response with extract_answer (last "Answer:" line after the last </think>), except
Omanic, which uses its own reward.parse (the same rule plus the letter checks).
"""

from __future__ import annotations

import importlib.util
import json
import re
import string
from collections import Counter
from functools import lru_cache

from harness_rl.core.env_paths import OMANIC_REWARD

# Same pattern as playground/musique/reward.py::_ANSWER_RE.
_ANSWER_RE = re.compile(r"^\s*\**\s*answer\s*\**\s*:\s*(.*?)\s*$", re.IGNORECASE)


def extract_answer(response: str) -> str | None:
    """The answer on the last "Answer:" line after the last </think>, or None.

    Mirrors playground/musique/reward.py::parse (answer part only): only text after the last </think> is read,
    lines are scanned bottom-up, the first match wins, surrounding '*' are stripped, and an empty answer is None.
    As in the training reward, a response with no </think> at all (e.g. truncated while thinking) is read whole.
    """
    final = response.rsplit("</think>", 1)[-1]
    for line in reversed(final.strip().splitlines()):
        if m := _ANSWER_RE.match(line):
            return m.group(1).strip().strip("*").strip() or None
    return None


# ---------------------------------------------------------------- SQuAD EM / F1

def normalize_answer(s: str) -> str:
    """SQuAD v1.1 normalize_answer: lower, drop punctuation, drop articles a/an/the, collapse whitespace."""
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(ch for ch in s if ch not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


_SPECIAL = {"yes", "no", "noanswer"}


def f1(pred: str, gold: str) -> float:
    """SQuAD token F1 on normalized answers, with the HotpotQA / 2Wiki official rule that a yes/no/noanswer on
    either side scores 0 unless both normalize identically."""
    p, g = normalize_answer(pred), normalize_answer(gold)
    if (p in _SPECIAL or g in _SPECIAL) and p != g:
        return 0.0
    pt, gt = p.split(), g.split()
    if not pt or not gt:
        return float(pt == gt)
    common = sum((Counter(pt) & Counter(gt)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(pt), common / len(gt)
    return 2 * precision * recall / (precision + recall)


def golds(task: dict) -> list[str]:
    return [task["answer"], *task.get("aliases", [])]


def score_qa(task: dict, response: str) -> dict:
    pred = extract_answer(response)
    if pred is None:
        return {"score": 0.0, "em": 0.0, "f1": 0.0, "format_ok": 0.0, "pred": None}
    em = max(exact_match(pred, g) for g in golds(task))
    f = max(f1(pred, g) for g in golds(task))
    return {"score": f, "em": em, "f1": f, "format_ok": 1.0, "pred": pred}


# ---------------------------------------------------------------- Omanic

@lru_cache(maxsize=1)
def omanic_reward():
    """playground/omanic/reward.py imported by path, unchanged."""
    spec = importlib.util.spec_from_file_location("omanic_reward", OMANIC_REWARD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def score_omanic(task: dict, response: str) -> dict:
    mod = omanic_reward()
    gt = {"label": task["answer"], "answer": task["meta"]["answer_text"]}
    out = mod.compute_score("omanic", response, json.dumps(gt, ensure_ascii=False), task["meta"])
    return {**out, "pred": mod.parse(response)}


# ---------------------------------------------------------------- BABILong

# booydar/babilong@7a6efee2 babilong/metrics.py (qa1-qa5 entries).
BABILONG_TASK_LABELS = {
    "qa1": ["bathroom", "bedroom", "garden", "hallway", "kitchen", "office"],
    "qa2": ["bathroom", "bedroom", "garden", "hallway", "kitchen", "office"],
    "qa3": ["bathroom", "bedroom", "garden", "hallway", "kitchen", "office"],
    "qa4": ["bathroom", "bedroom", "garden", "hallway", "kitchen", "office"],
    "qa5": ["Bill", "Fred", "Jeff", "Mary", "apple", "football", "milk"],
}


def babilong_preprocess_output(output: str) -> str:
    """Official preprocess_output: lower-case, keep the first sentence, cut at <context>/<example>/'Question'.
    (The 'Question' cut never fires after lower-casing; kept as in the official code.)"""
    output = output.lower()
    output = output.split(".")[0]
    output = output.split("<context>")[0]
    output = output.split("<example>")[0]
    output = output.split("Question")[0]
    return output


def babilong_compare_answers(target: str, output: str, question: str, task_labels: list[str]) -> bool:
    """Official compare_answers: the target must be the only task label mentioned in the (first sentence of the)
    output, ignoring labels that also occur in the question; comma-separated targets (qa8) need all parts."""
    output = babilong_preprocess_output(output)
    target = target.lower()
    labels = {label.lower() for label in task_labels}
    in_question = {label for label in labels if label in question.lower()}
    in_output = {label for label in labels if label in output} - in_question
    if "," in target and len(target) > 3:
        subtargets = target.split(",")
        return all(t in in_output for t in subtargets) and len(in_output) == len(subtargets)
    return target in in_output and len(in_output) == 1


def score_babilong(task: dict, response: str) -> dict:
    pred = extract_answer(response)
    if pred is None:
        return {"score": 0.0, "format_ok": 0.0, "pred": None}
    meta = task["meta"]
    ok = babilong_compare_answers(task["answer"], pred, meta["question"], BABILONG_TASK_LABELS[meta["task"]])
    return {"score": float(ok), "format_ok": 1.0, "pred": pred}


# ---------------------------------------------------------------- RULER variable tracking

def ruler_postprocess_pred(pred: str) -> str:
    """RULER scripts/eval/evaluate.py postprocess_pred: strip, control characters -> newline, strip."""
    return re.sub(r"[\x00-\x1f]", "\n", pred.strip()).strip()


def ruler_string_match_all(pred: str, refs: list[str]) -> float:
    """RULER string_match_all for one sample (as a fraction, not x100): share of references contained in the
    prediction, case-insensitive."""
    return sum(1.0 if r.lower() in pred.lower() else 0.0 for r in refs) / len(refs)


def score_ruler_vt(task: dict, response: str) -> dict:
    refs = task["meta"]["vars"]
    pred = extract_answer(response)
    if pred is None:
        return {"score": 0.0, "all_match": 0.0, "exact_set": 0.0, "n_pred": 0, "format_ok": 0.0, "pred": None}
    pred = ruler_postprocess_pred(pred)
    recall = ruler_string_match_all(pred, refs)
    names = set(re.findall(r"\b[A-Z]{5}\b", pred))
    return {"score": recall, "all_match": float(recall == 1.0), "exact_set": float(names == set(refs)),
            "n_pred": len(names), "format_ok": 1.0, "pred": pred}


SCORERS = {
    "omanic_letter": score_omanic,
    "qa_em_f1": score_qa,
    "babilong_match": score_babilong,
    "ruler_vt_recall": score_ruler_vt,
}


def score(task: dict, response: str) -> dict:
    return SCORERS[task["scorer"]](task, response)
