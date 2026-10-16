"""1.1 adapter contract, parametrized over every registered adapter (plus a toy adapter that exercises the checks).

Checks: determinism by seed, spec JSON round trip, validity of sampled tasks, twin expectations, probes near chance,
version = package content hash, and that core/ never imports an adapter.
"""

import ast
import json
import math
import random
from pathlib import Path

import pytest

import harness_rl.adapters  # noqa: F401  (registers the built-in adapters)
from harness_rl.core import registry
from harness_rl.core.adapter import (Adapter, Cell, Probe, Reward, Task, TrustLevel, Twin, ValidityReport, make_task,
                                     package_version, rng_for, rng_for_spec, spec_id)

TASKS_PER_CELL = 3
MAX_CELLS = 40
PROBE_TASKS = 150


class ToyAdd:
    """Smallest possible adapter: add n integers. Lives in the test so the contract checks always run."""

    name = "toy_add"
    trust = TrustLevel.COMPUTED
    version = package_version(__file__)

    def cells(self):
        return [Cell({"n": n}) for n in (2, 3)]

    def sample_spec(self, cell, rng):
        return {"n": cell.knobs["n"], "seed": rng.randrange(2**31)}

    def instantiate(self, spec, rng):
        nums = [rng.randrange(100, 1000) for _ in range(spec["n"])]
        ans = str(sum(nums))
        return Task(task_id=spec_id(spec, "toy-"), spec=spec, prompt=f"What is {' + '.join(map(str, nums))}?",
                    answer=ans, meta={"answer_text": ans, "cell": {"n": spec["n"]}, "nums": nums})

    def verify(self, task, final_answer):
        return Reward(float(final_answer.strip() == task.answer))

    def validity(self, task):
        return ValidityReport(task.answer not in task.prompt, [] if task.answer not in task.prompt else ["answer in prompt"])

    def twins(self, task, rng):
        spec = {**task.spec, "seed": task.spec["seed"] + 1}
        t = self.instantiate(spec, rng_for_spec(spec))
        return [Twin("perturb", t, t.answer)]

    def probes(self):
        return [Probe("last_number", lambda t, r: str(t.meta["nums"][-1]), lambda t: 0.0)]

    def signature(self, spec):
        return ("add", spec["n"])

    def leakage(self, scaffold, task):
        return any(str(x) in scaffold for x in task.meta["nums"])


TOY = ToyAdd()
ADAPTERS = [TOY] + registry.all_adapters()
IDS = [a.name for a in ADAPTERS]


def sample_tasks(adapter, per_cell=TASKS_PER_CELL, seed0=0):
    cells = adapter.cells()[:MAX_CELLS]
    return [make_task(adapter, c, seed0 + s) for c in cells for s in range(per_cell)]


@pytest.fixture(params=ADAPTERS, ids=IDS, scope="module")
def adapter(request):
    return request.param


@pytest.fixture(scope="module")
def tasks(adapter):
    return sample_tasks(adapter)


def test_protocol_and_version(adapter):
    assert isinstance(adapter, Adapter)
    assert isinstance(adapter.trust, TrustLevel)
    assert len(adapter.version) == 64
    assert adapter.cells() and len({c.key for c in adapter.cells()}) == len(adapter.cells())


def test_determinism_by_seed(adapter):
    a = [t.to_json() for t in sample_tasks(adapter, per_cell=2, seed0=11)]
    b = [t.to_json() for t in sample_tasks(adapter, per_cell=2, seed0=11)]
    c = [t.to_json() for t in sample_tasks(adapter, per_cell=2, seed0=12)]
    assert a == b
    assert a != c


def test_spec_json_round_trip(adapter, tasks):
    for t in tasks:
        spec2 = json.loads(json.dumps(t.spec))
        assert spec2 == t.spec
        t2 = adapter.instantiate(spec2, rng_for_spec(spec2))
        assert t2.to_json() == t.to_json()
        assert Task.from_dict(json.loads(t.to_json())).to_json() == t.to_json()
        assert hash(adapter.signature(t.spec)) is not None


def test_sampled_tasks_are_valid(adapter, tasks):
    for t in tasks:
        rep = adapter.validity(t)
        assert rep.ok, (t.task_id, rep.reasons)
        assert t.prompt == t.prompt.strip()  # Pi strips stdin prompts
        assert t.files == {} and t.tools == []  # Phases 0-1
        assert adapter.verify(t, t.meta["answer_text"]).score == 1.0
        assert not adapter.leakage("", t)


def test_twin_expectations_hold(adapter, tasks):
    for i, t in enumerate(tasks):
        twins = adapter.twins(t, rng_for("twins", t.task_id))
        for tw in twins:
            if tw.expect_valid:
                assert adapter.validity(tw.task).ok, (tw.kind, adapter.validity(tw.task).reasons)
            assert adapter.verify(tw.task, tw.expected).score == 1.0, (t.task_id, tw.kind)
            assert tw.task.to_json() != t.to_json()


def test_probes_near_chance(adapter):
    tasks = sample_tasks(adapter, per_cell=max(1, PROBE_TASKS // max(1, min(MAX_CELLS, len(adapter.cells())))), seed0=1000)
    for probe in adapter.probes():
        rng = random.Random(0)
        acc = sum(adapter.verify(t, probe.solve(t, rng)).score for t in tasks) / len(tasks)
        chance = sum(probe.chance(t) for t in tasks) / len(tasks)
        tol = 0.05 + 3 * math.sqrt(max(chance * (1 - chance), 1e-4) / len(tasks))
        assert abs(acc - chance) <= tol, (probe.name, acc, chance)


def test_core_never_imports_adapters():
    core = Path(__file__).resolve().parents[1] / "core"
    for py in core.glob("*.py"):
        for node in ast.walk(ast.parse(py.read_text())):
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods = [node.module or ""]
            else:
                continue
            assert not any("adapters" in m for m in mods), (py.name, mods)
