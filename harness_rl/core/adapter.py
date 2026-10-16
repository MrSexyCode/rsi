"""Domain-agnostic adapter interface (plan 1.1). An adapter turns a knob grid into verifiable tasks.

Contract, checked for every registered adapter by tests/test_adapter_contract.py:
  - determinism: sample_spec(cell, rng_for(seed, ...)) and instantiate(spec, rng_for_spec(spec)) depend only on their
    inputs; the same seed gives byte-identical specs and tasks. instantiate's rng is a convenience: an adapter may
    ignore it and derive its own RNGs from spec fields (needed when twins must share parts of the base task).
  - leakage() and the "answer not in question" part of validity are adapter-defined: domains without a separate
    context (e.g. program output) redefine them and document how.
  - EXECUTABLE adapters may execute code in sample_spec/instantiate (sandboxed, with a timeout) and must cache the
    result in task.meta so verify/validity/probes never execute.
  - spec is JSON-serializable and fully reproduces the task (round trip through JSON gives the same task)
  - sampled tasks pass validity()
  - every twin's expected answer is what verify() accepts for the twin task
  - every probe scores near its chance level on a sample

core/ never imports an adapter. Adapters import this module and call core.registry.register(); the package
harness_rl.adapters imports every built-in adapter, so `import harness_rl.adapters` populates the registry.
"""

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from harness_rl.core.versioning import content_hash

UNDETERMINED = "<UNDETERMINED>"  # the context does not determine an answer; the model should abstain
AMBIGUOUS = "<AMBIGUOUS>"  # the context admits several answers; such a task is invalid


class TrustLevel(Enum):
    COMPUTED = "computed"  # answer computed exactly from the task's own structure
    EXECUTABLE = "executable"  # answer obtained by executing something (a program) in a sandbox
    JUDGED = "judged"  # answer needs a judge model


@dataclass(frozen=True)
class Cell:
    """One point of the adapter's knob grid. `knobs` must be JSON-serializable."""

    knobs: dict

    @property
    def key(self) -> str:
        return ",".join(f"{k}={self.knobs[k]}" for k in sorted(self.knobs))

    def to_dict(self) -> dict:
        return dict(self.knobs)


@dataclass
class Task:
    task_id: str
    spec: dict  # JSON-serializable, fully reproduces the task
    prompt: str
    files: dict[str, str] = field(default_factory=dict)  # empty until documents (Phase 6)
    tools: list[str] = field(default_factory=list)  # empty in Phases 0-1
    answer: Any = None  # gold answer, or UNDETERMINED
    meta: dict = field(default_factory=dict)  # gold path, bridge entities, cell, etc.
    # Contract: meta["answer_text"] is a final-answer string verify(task, ...) scores 1.0, meta["cell"] the cell knobs.

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        return cls(**d)


@dataclass
class Reward:
    score: float
    components: dict = field(default_factory=dict)


@dataclass
class ValidityReport:
    ok: bool
    reasons: list[str] = field(default_factory=list)


@dataclass
class Twin:
    kind: str  # e.g. "counterfactual", "ablation", "rename", "perturb"
    task: Task
    expected: str  # a final-answer string verify(task, expected) scores 1.0 (the abstention string for UNDETERMINED)
    expect_valid: bool = True  # False for twins that are deliberately not valid tasks, e.g. ablations (UNDETERMINED)


@dataclass
class Probe:
    """A shortcut heuristic. Accuracy near chance(task) means the shortcut is closed."""

    name: str
    solve: Callable[[Task, random.Random], str]  # returns a final-answer string, scored with adapter.verify
    chance: Callable[[Task], float]


@runtime_checkable
class Adapter(Protocol):
    name: str
    version: str  # content hash of the adapter package
    trust: TrustLevel

    def cells(self) -> list[Cell]: ...
    def sample_spec(self, cell: Cell, rng: random.Random) -> dict: ...
    def instantiate(self, spec: dict, rng: random.Random) -> Task: ...
    def verify(self, task: Task, final_answer: str) -> Reward: ...
    def validity(self, task: Task) -> ValidityReport: ...
    def twins(self, task: Task, rng: random.Random) -> list[Twin]: ...
    def probes(self) -> list[Probe]: ...
    def signature(self, spec: dict) -> tuple: ...
    def leakage(self, scaffold: str, task: Task) -> bool: ...


# ---- helpers shared by adapters and tests ----

def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def rng_for(*parts) -> random.Random:
    """A Random seeded from a stable hash of `parts` (ints/strs/JSON-able). Never use the global RNG."""
    digest = hashlib.sha256(canonical_json([str(p) if not isinstance(p, (int, str)) else p for p in parts]).encode())
    return random.Random(int.from_bytes(digest.digest()[:8], "big"))


def rng_for_spec(spec: dict) -> random.Random:
    """The RNG instantiate() receives for a spec: a function of the spec alone, so the spec reproduces the task."""
    return rng_for("instantiate", canonical_json(spec))


def spec_id(spec: dict, prefix: str = "") -> str:
    return prefix + hashlib.sha256(canonical_json(spec).encode()).hexdigest()[:16]


def package_version(module_file: str | Path) -> str:
    """Adapter version = content hash of the directory holding the adapter's module."""
    return content_hash(Path(module_file).resolve().parent)


def make_task(adapter: Adapter, cell: Cell, seed: int) -> Task:
    """Canonical way to draw one task: sample a spec for (cell, seed), then instantiate it from the spec alone."""
    spec = adapter.sample_spec(cell, rng_for(adapter.name, "sample", cell.key, seed))
    return adapter.instantiate(spec, rng_for_spec(spec))
