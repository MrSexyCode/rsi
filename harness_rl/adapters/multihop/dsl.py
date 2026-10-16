"""The multihop Spec, its knob values, the reserved seed ranges, and the v1 query programs (plan 1.2).

Spec (a JSON dict; `instantiate` rebuilds the task from it alone):
  seed              int; training draws come from TRAIN_SEEDS, the held-out eval grid from HELDOUT_SEEDS
  op                "chain" | "comparison" | "intersection"
  hops              2, 3, 4, 5, 6, 8, 10, 12
  distractors       "none" | "low" | "high" | "very_high"
  distractor_types  the near-miss types used (none: [], low: 2 of the 4, high and very_high: all 4)
  ordering          "path" | "reversed" | "shuffled"
  template_mix      "uniform" (each fact/question template drawn uniformly) | "first" (template 0 only)
  program           relation names of the query (below); sampled with the spec so the spec documents the structure
  edit              optional, twins only: {"kind": "counterfactual", "fact": id, "obj": entity id} or
                    {"kind": "ablation", "fact": id}, applied to the base task's rendered fact list

What "hops" means (the length of the longest dependency path of facts the answer needs, so difficulty grows with
hops for every operator):
  chain(start, rels)                         len(rels) = hops. Answer: the entity reached.
  comparison(chain_a, chain_b, attr, op)     len(rels_a) = len(rels_b) = hops - 1, plus one attribute fact per side.
                                             Answer: the end entity with the larger (op "max") / smaller ("min")
                                             attribute; ties are AMBIGUOUS.
  intersection(chain_a, rel_a, chain_b, rel_b)
                                             len(rels_a) = len(rels_b) = hops - 1, plus the two join facts. Answer:
                                             the unique E with rel_a(E) = end(chain_a) and rel_b(E) = end(chain_b).
"""

from __future__ import annotations

from dataclasses import dataclass

OPS = ("chain", "comparison", "intersection")
HOPS = (2, 3, 4, 5, 6, 8, 10, 12)
LEVELS = ("none", "low", "high", "very_high")
ORDERINGS = ("path", "reversed", "shuffled")
DISTRACTOR_TYPES = ("same_rel_other_subj", "same_subj_other_rel", "dead_end", "branch")
TEMPLATE_MIXES = ("uniform", "first")
CMP_OPS = ("max", "min")

# Reserved seed ranges: the only place they are defined. sample_spec draws from TRAIN_SEEDS; only the held-out
# grid builder uses HELDOUT_SEEDS.
TRAIN_SEEDS = range(0, 1 << 30)
HELDOUT_SEEDS = range(1 << 30, (1 << 30) + (1 << 24))

ADAPTER_RELEASE = "v2"  # human-readable release; the adapter version proper is the package content hash

# Structure sizes (part of the adapter version, not of the spec). very_high also adds decoys.
CHAIN_DECOYS = {"none": 5, "low": 5, "high": 5, "very_high": 9}      # full-length decoy chains (gold relations)
COMPARISON_DECOYS = {"none": 2, "low": 2, "high": 2, "very_high": 4}  # per side; their ends carry the attribute
INTERSECTION_GRID = {"none": 3, "low": 3, "high": 3, "very_high": 4}  # ends per side; 2n candidates


def distractors_per_type(level: str, hops: int) -> int:
    """Near-miss insertions per distractor type: grows with hops so long chains get proportionally many."""
    base = 1 + hops // 3
    return {"none": 0, "low": 1, "high": base, "very_high": 3 * base}[level]


def branch_len(op: str, hops: int) -> int:
    return hops if op == "chain" else hops - 1


# ---------------------------------------------------------------- programs (over entity names)

@dataclass(frozen=True)
class Chain:
    start: str
    rels: tuple[str, ...]


@dataclass(frozen=True)
class Comparison:
    a: Chain
    b: Chain
    attr: str
    cmp: str  # "max" | "min"


@dataclass(frozen=True)
class Intersection:
    a: Chain
    rel_a: str
    b: Chain
    rel_b: str


Program = Chain | Comparison | Intersection


def program_relations(spec: dict) -> tuple:
    """The relation-type sequence of a spec's program, as a hashable tuple."""
    p, op = spec["program"], spec["op"]
    if op == "chain":
        return tuple(p["rels"])
    if op == "comparison":
        return tuple(p["a"]), tuple(p["b"]), p["attr"], p["cmp"]
    return tuple(p["a"]), p["join_a"], tuple(p["b"]), p["join_b"]


def check_spec(spec: dict) -> None:
    assert spec["op"] in OPS and spec["hops"] >= 2, spec
    assert spec["distractors"] in LEVELS and spec["ordering"] in ORDERINGS, spec
    assert spec["template_mix"] in TEMPLATE_MIXES, spec
    assert set(spec["distractor_types"]) <= set(DISTRACTOR_TYPES), spec
    L = branch_len(spec["op"], spec["hops"])
    p = spec["program"]
    if spec["op"] == "chain":
        assert len(p["rels"]) == L, spec
    else:
        assert len(p["a"]) == len(p["b"]) == L, spec
