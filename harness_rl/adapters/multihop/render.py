"""Facts and questions to text, and text back to facts (plan 1.2).

Context: a numbered fact list. Ordering knob, by the facts' slots (hop index from the question entity; decoys and
distractors take the slot of the hop they mimic): path = ascending slot, reversed = descending slot, shuffled =
random; ties are broken at random, so all facts of one slot are exchangeable. Questions nest the relations'
noun-phrase templates ("the birthplace of the founder of X"). The prompt states the answer format and the
abstention string, and ends with the eval sets' answer instruction so evals.scoring.extract_answer applies.
"""

from __future__ import annotations

import random
import re

from harness_rl.adapters.multihop.dsl import Chain, Comparison, Intersection, Program
from harness_rl.adapters.multihop.schema import (ATTRIBUTES, RELATIONS, TYPE_NOUN, fact_regexes, fact_templates,
                                                 fmt_value, parse_value)

ABSTAIN = "cannot be determined"
INTRO = ("Answer the question using only the numbered facts below. All names are fictional. Answering it requires "
         "several reasoning steps.")
ABSTAIN_RULE = f'If the facts do not determine the answer, the answer is "{ABSTAIN}".'
ANSWER_INSTRUCTION = "Reason step by step. Then end your response with exactly this line:"  # = evals.prompts
ANSWER_LINE = f'Answer: <the name exactly as written in the facts, or "{ABSTAIN}">'
CHAIN_QUESTIONS = ("Which {t} is {np}?", "What is the name of {np}?")
INTERSECTION_QUESTIONS = ("Which is the {t} {vp_a} and {vp_b}?", "Name the {t} {vp_a} and {vp_b}.")


def order_facts(facts, ordering: str, rng: random.Random) -> list:
    keys = [rng.random() for _ in facts]
    sign = {"path": 1, "reversed": -1, "shuffled": 0}[ordering]
    return [facts[i] for _, _, i in sorted((sign * f.slot, k, i) for i, (f, k) in enumerate(zip(facts, keys)))]


def pick(rng: random.Random, options, mix: str):
    return options[0] if mix == "first" else options[rng.randrange(len(options))]


def fact_sentence(subj: str, rel: str, obj, tidx: int) -> str:
    t = fact_templates(rel)[tidx]
    return t.format(s=subj, o=obj) if rel in RELATIONS else t.format(s=subj, v=fmt_value(rel, obj))


def _np(c: Chain, rng: random.Random, mix: str) -> str:
    x = c.start
    for r in c.rels:
        x = pick(rng, RELATIONS[r].nps, mix).format(x=x)
    return x


def question(program: Program, answer_type: str, rng: random.Random, mix: str) -> str:
    t = TYPE_NOUN[answer_type]
    if isinstance(program, Chain):
        np, tmpl = _np(program, rng, mix), pick(rng, CHAIN_QUESTIONS, mix)
        if np.startswith(f"the {t} "):  # "Which city is the city that ..." -> "What is the city that ...?"
            tmpl = CHAIN_QUESTIONS[1]
        return tmpl.format(t=t, np=np)
    if isinstance(program, Comparison):
        a, b = _np(program.a, rng, mix), _np(program.b, rng, mix)
        return pick(rng, ATTRIBUTES[program.attr].questions[program.cmp], mix).format(a=a, b=b)
    assert isinstance(program, Intersection)
    vp_a = pick(rng, RELATIONS[program.rel_a].vps, mix).format(y=_np(program.a, rng, mix))
    vp_b = pick(rng, RELATIONS[program.rel_b].vps, mix).format(y=_np(program.b, rng, mix))
    return pick(rng, INTERSECTION_QUESTIONS, mix).format(t=t, vp_a=vp_a, vp_b=vp_b)


def prompt(sentences: list[str], q: str) -> str:
    facts = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences, 1))
    return f"{INTRO}\n\nFacts:\n{facts}\n\nQuestion: {q}\n\n{ABSTAIN_RULE}\n\n{ANSWER_INSTRUCTION}\n{ANSWER_LINE}"


# ---------------------------------------------------------------- parsing back

_LINE = re.compile(r"^(\d+)\. (.*)$")


def parse_sentence(sentence: str) -> list[tuple[str, int, tuple]]:
    """Every (relation, template index, triple) whose template regex matches the sentence."""
    out = []
    for rel, i, rx in fact_regexes():
        if m := rx.match(sentence):
            obj = parse_value(m["v"]) if rel in ATTRIBUTES else m["o"]
            out.append((rel, i, (m["s"], rel, obj)))
    return out


def context_sentences(text: str) -> list[str]:
    body = text.split("\nFacts:\n", 1)[1].split("\n\nQuestion: ", 1)[0]
    out = []
    for k, line in enumerate(body.splitlines(), 1):
        m = _LINE.match(line)
        assert m and int(m[1]) == k, line
        out.append(m[2])
    return out


def parse_context(text: str) -> list[tuple]:
    """The fact triples of a rendered prompt, in order. Raises if a line does not parse to exactly one fact."""
    out = []
    for s in context_sentences(text):
        hits = parse_sentence(s)
        if len(hits) != 1:
            raise ValueError(f"fact line parses {len(hits)} ways: {s!r}")
        out.append(hits[0][2])
    return out


def question_of(text: str) -> str:
    return text.split("\n\nQuestion: ", 1)[1].split("\n\n", 1)[0]
