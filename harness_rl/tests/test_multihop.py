"""Milestones 1.2-1.4: the multihop world compiler (render round trip, executor, distractors, twins, validity,
names, leakage, seed ranges, eval grid, pool, verl reward). The adapter contract itself is test_adapter_contract.py.

Heavier sweeps (probes on the full grid, full eval-grid rebuild) run with HARNESS_RL_SLOW=1.
"""

import json
import os
import random
import re
from collections import Counter
from pathlib import Path

import pytest

import harness_rl.adapters  # noqa: F401
from harness_rl.adapters.multihop import ADAPTER as A
from harness_rl.adapters.multihop import build_sets, names, reward, schema
from harness_rl.adapters.multihop.adapter import heldout_spec, render_spec
from harness_rl.adapters.multihop.dsl import (DISTRACTOR_TYPES, HELDOUT_SEEDS, TRAIN_SEEDS, Chain, Comparison,
                                              Intersection)
from harness_rl.adapters.multihop.executor import execute
from harness_rl.adapters.multihop.render import (ABSTAIN, ANSWER_LINE, parse_context, parse_sentence)
from harness_rl.core import registry
from harness_rl.core.adapter import AMBIGUOUS, UNDETERMINED, make_task, rng_for
from harness_rl.core.versioning import file_hash
from harness_rl.evals import load_manifest, load_set, scoring
from harness_rl.evals.common import SETS_DIR, sha256_bytes
from harness_rl.evals.prompts import ANSWER_INSTRUCTION

SLOW = os.environ.get("HARNESS_RL_SLOW") == "1"
CELLS = A.cells()


def sample(per_cell=1, seed0=0, every=1, **knobs):
    cells = [c for c in CELLS if all(c.knobs[k] == v for k, v in knobs.items())][::every]
    return [make_task(A, c, seed0 + s) for c in cells for s in range(per_cell)]


@pytest.fixture(scope="module")
def tasks():
    return sample(per_cell=2, seed0=500, every=1)


def test_registered_and_grid():
    assert registry.get("multihop") is A
    assert len(CELLS) == 8 * 3 * 4 * 3 == 288
    assert {c.knobs["hops"] for c in CELLS} == {2, 3, 4, 5, 6, 8, 10, 12}
    assert {c.knobs["distractors"] for c in CELLS} == {"none", "low", "high", "very_high"}
    assert {tuple(sorted(c.knobs)) for c in CELLS} == {("distractors", "hops", "op", "ordering")}


# ------------------------------------------------------------------------------------------------ schema & render

def test_template_counts():
    assert len(schema.RELATIONS) >= 20
    for r in schema.RELATIONS.values():
        assert len(r.facts) >= 5 and len(r.nps) >= 2 and len(r.vps) >= 2, r.name
        assert all(t.endswith("{x}") for t in r.nps) and all(t.endswith("{y}") for t in r.vps), r.name
    for a in schema.ATTRIBUTES.values():
        assert len(a.facts) >= 5 and all(len(a.questions[op]) >= 2 for op in ("max", "min")), a.name
    assert set(schema.ATTR_OF_TYPE) == set(names.TYPES)


def _has_key(text: str, key: str) -> bool:
    return re.search(rf"\b{re.escape(key)}\b", text.lower()) is not None


def test_templates_unambiguous():
    """Every template of a relation names it with one of its own key words and uses no other relation's keys."""
    rels = list(schema.RELATIONS.values()) + list(schema.ATTRIBUTES.values())
    for r in rels:
        texts = list(r.facts) + (list(r.nps) + list(r.vps) if r.name in schema.RELATIONS
                                 else [q for qs in r.questions.values() for q in qs])
        for t in texts:
            assert any(_has_key(t, k) for k in r.keys), (r.name, t)
            for o in rels:
                if o is not r:
                    assert not any(_has_key(t, k) for k in o.keys), (r.name, t, o.name)
    keys = [k for r in rels for k in r.keys]
    assert len(keys) == len(set(keys))
    from harness_rl.adapters.multihop import render
    frame = " ".join(render.CHAIN_QUESTIONS + render.INTERSECTION_QUESTIONS) + render.INTRO + render.ABSTAIN_RULE
    assert not [k for k in keys if _has_key(frame, k)]


def test_question_wording():
    for t in sample(per_cell=2, seed0=3, every=2):
        q = t.meta["question"]
        t_noun = schema.TYPE_NOUN[t.meta["answer_type"]]
        assert not re.search(rf"\bWhich {t_noun} is the {t_noun}\b", q), q
        assert not re.search(r"\b(\w+) that is the \1\b", q), q
        assert q[0].isupper() and q[-1] in "?.", q
        assert q.count("Which") + q.count("Name ") + q.count("What") <= 1, q


def test_every_template_parses_back_uniquely():
    rng = random.Random(0)
    gen = names.NameGen(rng)
    for rel in list(schema.RELATIONS.values()) + list(schema.ATTRIBUTES.values()):
        s = gen.new(rel.subj)
        o = gen.new(rel.obj) if rel.name in schema.RELATIONS else 1_234_567 if rel.kind == "count" else 1987
        for i, t in enumerate(rel.facts):
            is_rel = rel.name in schema.RELATIONS
            sent = t.format(s=s, o=o) if is_rel else t.format(s=s, v=schema.fmt_value(rel.name, o))
            assert parse_sentence(sent) == [(rel.name, i, (s, rel.name, o))], sent


def test_render_round_trip(tasks):
    """Rendered facts parse back, with the template regexes, to exactly the source triples, in display order."""
    for t in tasks[::3] + [tw.task for tw in A.twins(tasks[0], random.Random(1))]:
        r = render_spec(t.spec)
        src = [r.triples[f.id] for f, _ in r.keep]
        assert parse_context(t.prompt) == src
        assert [list(x) for x in src] == t.meta["context"]
        for (f, ti), sent in zip(r.keep, r.sentences):
            assert [(h[0], h[1]) for h in parse_sentence(sent)] == [(f.rel, ti)]


def test_prompt_format(tasks):
    for t in tasks:
        p = t.prompt
        assert p == p.strip()
        assert p.endswith(ANSWER_INSTRUCTION + "\n" + ANSWER_LINE)
        assert f'the answer is "{ABSTAIN}"' in p and "\n\nQuestion: " in p
        assert scoring.extract_answer(f"<think>x</think>\nAnswer: {t.meta['answer_text']}") == t.meta["answer_text"]


def test_ordering_knob():
    for ordering, ok in [("path", lambda s: s == sorted(s)), ("reversed", lambda s: s == sorted(s, reverse=True))]:
        for t in sample(ordering=ordering, every=5):
            assert ok(t.meta["slots"]), (ordering, t.meta["slots"])
    shuffled = sample(ordering="shuffled", hops=4)
    assert any(t.meta["slots"] != sorted(t.meta["slots"]) for t in shuffled)


def test_difficulty_grows_with_hops_and_level():
    hops = sorted({c.knobs["hops"] for c in CELLS})
    for op in ("chain", "comparison", "intersection"):
        gold = [sum(len(t.meta["gold_lines"]) for t in sample(per_cell=3, hops=h, op=op, distractors="none")) for h
                in hops]
        assert gold == sorted(gold) and len(set(gold)) == len(hops), (op, gold)
        for h in (2, 6, 12):
            size = {d: sum(len(t.meta["context"]) for t in sample(per_cell=3, hops=h, op=op, distractors=d,
                                                                   ordering="shuffled"))
                    for d in ("none", "low", "high", "very_high")}
            assert size["none"] < size["low"] < size["high"] < size["very_high"], (op, h, size)
            cands = {d: sum(len(t.meta["candidates"])
                            for t in sample(per_cell=3, hops=h, op=op, distractors=d, ordering="shuffled"))
                     for d in ("high", "very_high")}
            assert op == "comparison" or cands["very_high"] > cands["high"], (op, h, cands)


# ------------------------------------------------------------------------------------------------ executor

def test_executor_hand_built():
    ctx = [("Ann", "works_for", "Acme"), ("Acme", "headquartered_in", "Rome"), ("Bob", "works_for", "Bolt"),
           ("Bolt", "headquartered_in", "Oslo"), ("Rome", "population", 300), ("Oslo", "population", 200),
           ("Eve", "born_in", "Rome"), ("Eve", "works_for", "Bolt"), ("Dan", "born_in", "Rome"),
           ("Dan", "works_for", "Acme")]
    ann, bob = Chain("Ann", ("works_for", "headquartered_in")), Chain("Bob", ("works_for", "headquartered_in"))
    r = execute(ann, ctx)
    assert r.answer == "Rome" and r.bridges == ["Acme"] and len(r.used) == 2
    assert execute(Comparison(ann, bob, "population", "max"), ctx).answer == "Rome"
    assert execute(Comparison(ann, bob, "population", "min"), ctx).answer == "Oslo"
    eve = execute(Intersection(Chain("Ann", ("works_for", "headquartered_in")), "born_in", Chain("Bob", ("works_for",)),
                               "works_for"), ctx)
    assert eve.answer == "Eve" and set(eve.bridges) >= {"Rome", "Bolt"}
    # a missing link -> UNDETERMINED
    assert execute(ann, ctx[:1]).answer == UNDETERMINED
    assert execute(Comparison(ann, bob, "population", "max"), ctx[:5]).answer == UNDETERMINED
    bob1 = Chain("Bob", ("works_for",))
    assert execute(Intersection(ann, "born_in", bob1, "works_for"), ctx[:7]).answer == UNDETERMINED
    # not unique -> AMBIGUOUS
    assert execute(ann, ctx + [("Acme", "headquartered_in", "Oslo")]).answer == AMBIGUOUS
    assert execute(Comparison(ann, ann, "population", "max"), ctx).answer == AMBIGUOUS
    tie = [x if x != ("Oslo", "population", 200) else ("Oslo", "population", 300) for x in ctx]
    assert execute(Comparison(ann, bob, "population", "max"), tie).answer == AMBIGUOUS
    assert execute(Intersection(ann, "born_in", Chain("Ann", ("works_for",)), "works_for"),
                   ctx + [("Fay", "born_in", "Rome"), ("Fay", "works_for", "Acme")]).answer == AMBIGUOUS


# ------------------------------------------------------------------------------------------------ distractors

def test_distractors_never_change_the_answer(tasks):
    for t in tasks:
        core = [tuple(x) for x, role in zip(t.meta["context"], t.meta["roles"])
                if role in ("gold", "decoy", "join", "attr")]
        assert execute(A.program(t.meta["program"]), core).answer == t.answer, t.task_id
        if t.meta["cell"]["distractors"] == "none":
            assert set(t.meta["roles"]) <= {"gold", "decoy", "join", "attr"}


def test_every_distractor_type_at_high_level():
    high = sample(distractors="high", ordering="shuffled") + sample(distractors="very_high", ordering="shuffled")
    for t in high:
        assert sorted(t.spec["distractor_types"]) == sorted(DISTRACTOR_TYPES)
        assert all(n >= 1 for n in t.meta["distractors_placed"].values()), t.meta["distractors_placed"]
        assert set(DISTRACTOR_TYPES) <= set(t.meta["roles"])
    low = sample(distractors="low", ordering="shuffled")
    assert all(len(t.spec["distractor_types"]) == 2 for t in low)


def test_answer_type_only_at_answer_slot(tasks):
    """The symmetry the probes rely on: every answer-type candidate sits at the answer slot, mentioned equally often."""
    for t in tasks:
        if t.meta["cell"]["op"] == "comparison":
            continue
        cands = set(t.meta["candidates"])
        h = t.meta["cell"]["hops"]
        mentions = Counter(n for line in t.meta["mentions"] for n in line)
        assert len({mentions[c] for c in cands}) == 1, t.task_id
        for line, slot in zip(t.meta["mentions"], t.meta["slots"]):
            if cands & set(line):
                assert slot == h, (t.task_id, line, slot)


# ------------------------------------------------------------------------------------------------ twins & validity

def test_twins(tasks):
    for t in tasks[::2]:
        cf, ab = A.twins(t, rng_for("twins", t.task_id))
        assert cf.kind == "counterfactual" and cf.expect_valid and A.validity(cf.task).ok
        assert cf.task.answer not in (t.answer, UNDETERMINED, AMBIGUOUS) and cf.expected == cf.task.answer
        assert A.verify(cf.task, cf.expected).score == 1.0 and A.verify(cf.task, t.meta["answer_text"]).score == 0.0
        diff = set(cf.task.prompt.splitlines()) ^ set(t.prompt.splitlines())
        assert len(diff) == 2  # exactly one fact line changed
        assert ab.kind == "ablation" and not ab.expect_valid and ab.task.answer == UNDETERMINED
        assert ab.expected == ABSTAIN and A.verify(ab.task, ABSTAIN).score == 1.0
        assert A.verify(ab.task, t.meta["answer_text"]).score == 0.0
        assert len(ab.task.meta["context"]) == len(t.meta["context"]) - 1
        assert A.instantiate(json.loads(json.dumps(cf.task.spec))).to_json() == cf.task.to_json()


def _with_question(t, q):
    old = t.meta["question"]
    return type(t)(**{**t.__dict__, "prompt": t.prompt.replace(f"Question: {old}", f"Question: {q}")})


def test_validity_rejects_leaky_questions(tasks):
    t = next(t for t in tasks if t.meta["bridges"])
    assert A.validity(t).ok
    bad = A.validity(_with_question(t, t.meta["question"] + f" (Hint: {t.meta['answer_text']}.)"))
    assert not bad.ok and "answer in question" in bad.reasons
    bridge = t.meta["bridges"][0]
    bad = A.validity(_with_question(t, t.meta["question"] + f" (Hint: {bridge}.)"))
    assert not bad.ok and f"bridge in question: {bridge}" in bad.reasons


def test_signature():
    t = sample(hops=3, op="intersection", distractors="high", ordering="path")[0]
    op, hops, rels, dtypes = A.signature(t.spec)
    assert (op, hops, dtypes) == ("intersection", 3, tuple(sorted(DISTRACTOR_TYPES)))
    assert rels == (tuple(t.spec["program"]["a"]), t.spec["program"]["join_a"], tuple(t.spec["program"]["b"]),
                    t.spec["program"]["join_b"])


# ------------------------------------------------------------------------------------------------ names & leakage

def test_names_fictional_unique_deterministic(tasks):
    block = names.blocklist()
    assert len(block) > 30_000
    for t in tasks:
        ents = list(t.meta["entity_types"])
        assert len(ents) == len(set(ents))
        cores = [w for n in ents for w in names.core_words(n)]
        assert len(cores) == len(set(cores))
        for i, a in enumerate(cores):
            assert a not in block and len(a) >= names.MIN_LEN and names.GRAMMAR.match(a)
            for b in cores[i + 1:]:
                assert not names.close(a, b), (a, b)
    a, b = names.NameGen(random.Random(7)), names.NameGen(random.Random(7))
    seq_a = [a.new(ty) for ty in names.TYPES * 20]
    assert seq_a == [b.new(ty) for ty in names.TYPES * 20]
    assert seq_a != [names.NameGen(random.Random(8)).new(ty) for ty in names.TYPES * 20]
    assert all(re.fullmatch(schema.NAME_RE, n) for n in seq_a)


def test_close():
    assert names.close("tavin", "tavin") and names.close("tavin", "tavim") and names.close("tavin", "tavins")
    assert names.close("tavin", "avin") and names.close("tavinor", "tavin")  # substring / prefix
    assert not names.close("tavin", "tolim") and not names.close("tavin", "davim")


RAW_LISTS = Path("/home/ubuntu/data/harness_rl_raw/english_words/words_alpha.txt")


@pytest.mark.skipif(not RAW_LISTS.exists(), reason="raw blocklist sources not downloaded")
def test_blocklist_matches_raw_sources():
    from harness_rl.adapters.multihop import build_blocklist

    lists = build_blocklist.raw_lists()
    assert len(lists["last"]) == 10_000 and len(lists["first"]) >= 8_000
    assert build_blocklist.blocklist_text(lists) == names.BLOCKLIST_FILE.read_text()
    assert {"london", "paris", "smith", "mary", "garden"} <= set().union(*lists.values())


def test_leakage(tasks):
    t = tasks[0]
    q_ent = t.meta["question_entities"][0]
    other = next(n for n in t.meta["entity_types"] if n not in t.meta["question_entities"])
    assert not A.leakage("", t)
    assert not A.leakage(f"First find the facts about {q_ent}, then follow each relation in turn.", t)
    assert A.leakage(f"Start from {q_ent}; the next entity is {other}.", t)
    assert A.leakage(f"maybe {other.split()[0].lower()}?", t)  # one core word is enough


# ----------------------------------------------------------------------------------------- seeds, eval grid, pool

def test_seed_ranges_disjoint():
    assert TRAIN_SEEDS.stop <= HELDOUT_SEEDS.start or HELDOUT_SEEDS.stop <= TRAIN_SEEDS.start
    for c in CELLS[::9]:
        for s in range(5):
            assert A.sample_spec(c, rng_for("x", c.key, s))["seed"] in TRAIN_SEEDS
    assert heldout_spec(CELLS[0], 0)["seed"] == HELDOUT_SEEDS.start


@pytest.fixture(scope="module")
def grid():
    return load_set(build_sets.SET_NAME)


def test_eval_grid_manifest(grid):
    m = load_manifest(build_sets.SET_NAME)
    data = (SETS_DIR / build_sets.SET_NAME / "data.jsonl").read_bytes()
    assert m["sha256"] == sha256_bytes(data) and m["size"] == len(grid) == 1800
    assert m["splits"]["dev"] == {"n": 1800, "sha256": sha256_bytes(data)} and m["splits"]["final"]["n"] == 0
    assert m["scorer"] == "multihop" and m["source"]["adapter_version"] == A.version, "eval grid stale: rebuild it"
    for path, h in m["generator"]["files"].items():
        assert file_hash(Path(__file__).resolve().parents[2] / path) == h, path
    cells = Counter((r["meta"]["cell"]["hops"], r["meta"]["cell"]["op"], r["meta"]["cell"]["distractors"],
                     r["meta"]["cell"]["ordering"]) for r in grid)
    assert len(cells) == 60 and set(cells.values()) == {30}
    assert {k[0] for k in cells} == {2, 4, 6, 8, 12} and {k[3] for k in cells} == {"shuffled"}
    assert {k[2] for k in cells} == {"none", "low", "high", "very_high"}
    assert m["prompt_tokens"]["max"] <= 6500 and m["source"]["adapter_release"] == "v2"


def test_eval_grid_items(grid):
    seeds = set()
    for i, r in enumerate(grid):
        assert r["split"] == "dev" and r["scorer"] == "multihop" and r["aliases"] == []
        assert r["meta"]["spec"]["seed"] in HELDOUT_SEEDS
        seeds.add(r["meta"]["spec"]["seed"])
        if i % 17 == 0:  # the spec rebuilds the item
            t = A.instantiate(r["meta"]["spec"])
            assert (t.prompt, t.meta["answer_text"], t.task_id) == (r["prompt"], r["answer"], r["id"])
    assert len(seeds) == len(grid)


def test_eval_grid_gold_scores_one(grid):
    for r in grid:
        out = scoring.score(r, f"<think>\nAnswer: wrong\n</think>\n\nReasoning.\nAnswer: {r['answer']}")
        assert out["score"] == 1.0 and out["format_ok"] == 1.0, (r["id"], out)
        assert scoring.score(r, f"<think>\nAnswer: {r['answer']}\n</think>\n\nnothing")["score"] == 0.0
        assert scoring.score(r, f"Answer: {ABSTAIN}")["score"] == 0.0


POOL = build_sets.POOL_DIR


@pytest.mark.skipif(not (POOL / "train.parquet").exists(), reason="training pool not built")
def test_pool_disjoint_from_heldout_and_uniform(grid):
    import pyarrow.parquet as pq

    held = {json.dumps(r["meta"]["spec"], sort_keys=True) for r in grid}
    held_ids = {r["id"] for r in grid}
    m = json.loads((POOL / "manifest.json").read_text())
    assert m["adapter_version"] == A.version, "pool stale: rebuild it"
    ids = []
    for split in ("train", "val"):
        tab = pq.read_table(POOL / f"{split}.parquet").to_pylist()
        assert len(tab) == m["splits"][split]["n"]
        assert file_hash(POOL / f"{split}.parquet") == m["splits"][split]["file_sha256"]
        for row in tab:
            info = row["extra_info"]
            assert row["data_source"] == "multihop" and info["spec"] not in held
            assert json.loads(info["spec"])["seed"] in TRAIN_SEEDS
            ids.append(info["task_id"])
        if split == "train":
            per_cell = Counter(json.dumps(r["extra_info"]["cell"], sort_keys=True) for r in tab)
            assert len(per_cell) == 288 and max(per_cell.values()) - min(per_cell.values()) <= 1
    assert len(ids) == len(set(ids)) and not set(ids) & held_ids


def test_pool_rows_deterministic(tmp_path):
    a = build_sets.pool_rows("val", 30, seed=3, workers=1)
    assert a == build_sets.pool_rows("val", 30, seed=3, workers=1)
    assert a != build_sets.pool_rows("val", 30, seed=4, workers=1)
    build_sets.write_parquet(a, tmp_path / "a.parquet")
    build_sets.write_parquet(a, tmp_path / "b.parquet")
    assert (tmp_path / "a.parquet").read_bytes() == (tmp_path / "b.parquet").read_bytes()
    row = a[0]
    assert row["prompt"][0]["role"] == "user" and row["reward_model"]["ground_truth"]
    t = reward.task_for(row["extra_info"]["spec"])
    assert t.prompt == row["prompt"][0]["content"] and t.meta["cell"] == row["extra_info"]["cell"]


# ------------------------------------------------------------------------------------------------ verl reward

def test_compute_score_agrees_with_verify(tasks):
    pool = tasks[::4]
    pool += [tw.task for tw in A.twins(pool[0], random.Random(2))]
    for t in pool:
        info = {"spec": json.dumps(t.spec), "index": 0}
        gt = t.meta["answer_text"]
        for ans in [gt, ABSTAIN, t.meta["candidates"][0] if t.meta["candidates"] else "x", "nonsense"]:
            resp = f"<think>\nmaybe {gt}\n</think>\n\nSo:\nAnswer: **{ans}**"
            out = reward.compute_score("multihop", resp, gt, info)
            v = A.verify(t, ans)
            assert out["score"] == v.score and all(out[k] == v.components[k] for k in v.components)
            assert out["format_ok"] == 1.0 and out["gt_match"] == 1.0
            assert all(isinstance(x, float) for x in out.values())
        assert reward.compute_score("multihop", f"<think>Answer: {gt}</think>\nno line", gt, info)["score"] == 0.0


def test_verify_components():
    t = sample(hops=2, op="chain", distractors="none", ordering="path")[0]
    assert A.verify(t, t.meta["answer_text"].upper() + ".").components["em"] == 1.0
    assert A.verify(t, "The " + t.meta["answer_text"]).score == 1.0  # SQuAD normalization drops articles
    r = A.verify(t, ABSTAIN)
    assert r.score == 0.0 and r.components["abstained"] == 1.0
    other = next(c for c in t.meta["candidates"] if c != t.answer)
    assert A.verify(t, other).components == {"em": 0.0, "abstained": 0.0, "gold_undetermined": 0.0,
                                             "in_context": 1.0}


# ------------------------------------------------------------------------------------------------ slow

@pytest.mark.skipif(not SLOW, reason="probe sweep over the full grid; set HARNESS_RL_SLOW=1")
def test_probes_near_chance_every_cell():
    from harness_rl.adapters.multihop import probe_table

    rows = probe_table.table(n=600, workers=min(64, os.cpu_count() or 1))
    for r in rows:
        assert abs(r["gap"]) <= 0.05 + 2 * r["se"], r


@pytest.mark.skipif(not SLOW, reason="full eval-grid rebuild; set HARNESS_RL_SLOW=1")
def test_eval_grid_rebuild_byte_identical():
    data, manifest = build_sets.build_eval(workers=min(64, os.cpu_count() or 1))
    assert data == (SETS_DIR / build_sets.SET_NAME / "data.jsonl").read_bytes()
    assert manifest == (SETS_DIR / build_sets.SET_NAME / "manifest.json").read_bytes()


@pytest.mark.skipif(not SLOW, reason="twins over the full grid; set HARNESS_RL_SLOW=1")
def test_twins_full_grid():
    for t in sample(per_cell=3, seed0=77):
        cf, ab = A.twins(t, rng_for("twins", t.task_id))
        assert A.validity(cf.task).ok and cf.task.answer != t.answer and ab.task.answer == UNDETERMINED
