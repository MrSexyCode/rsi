"""Shortcut probes (plan 1.3). Each returns a final-answer string; accuracy near chance means the shortcut is closed.

The candidate set is what a solver who skips the reasoning would guess among: every context entity of the answer
type (the question says "which city ..."), or, for comparisons, the two compared entities (chance 1/2, as the plan
defines it). chance(task) = 1 / len(candidates). Every positional or frequency probe is restricted to the
candidates; when its heuristic lands outside them (e.g. the most frequent entity is a bridge) it guesses uniformly
among them, so a heuristic that carries no information scores exactly chance rather than below it.

  first_fact           first candidate mentioned in the context (reading order)
  last_fact            last candidate mentioned
  most_frequent        the most mentioned entity of any type (guess if it is not a candidate)
  most_frequent_typed  the most mentioned candidate
  neighbor             a candidate sharing a fact with a question entity (the one-hop shortcut; guess if none)
  random_typed         a uniformly random candidate
Ties are broken at random.
"""

from __future__ import annotations

import random
from collections import Counter

from harness_rl.core.adapter import Probe, Task


def _cands(task: Task) -> list[str]:
    return task.meta["candidates"]


def chance(task: Task) -> float:
    return 1.0 / len(_cands(task)) if _cands(task) else 0.0


def _mentions(task: Task) -> list[str]:
    return [n for line in task.meta["mentions"] for n in line]


def _guess(task: Task, rng: random.Random) -> str:
    return rng.choice(_cands(task)) if _cands(task) else ""


def _top(counts: Counter, rng: random.Random) -> str | None:
    if not counts:
        return None
    best = max(counts.values())
    return rng.choice(sorted(n for n, c in counts.items() if c == best))


def first_fact(task, rng):
    c = set(_cands(task))
    return next((n for n in _mentions(task) if n in c), None) or _guess(task, rng)


def last_fact(task, rng):
    c = set(_cands(task))
    return next((n for n in reversed(_mentions(task)) if n in c), None) or _guess(task, rng)


def most_frequent(task, rng):
    top = _top(Counter(_mentions(task)), rng)
    return top if top in set(_cands(task)) else _guess(task, rng)


def most_frequent_typed(task, rng):
    c = set(_cands(task))
    return _top(Counter(n for n in _mentions(task) if n in c), rng) or _guess(task, rng)


def neighbor(task, rng):
    q, c = set(task.meta["question_entities"]), set(_cands(task))
    near = sorted({n for line in task.meta["mentions"] if q & set(line) for n in line} & c - q)
    return rng.choice(near) if near else _guess(task, rng)


def random_typed(task, rng):
    return _guess(task, rng)


PROBES = [Probe(f.__name__, f, chance)
          for f in (first_fact, last_fact, most_frequent, most_frequent_typed, neighbor, random_typed)]
