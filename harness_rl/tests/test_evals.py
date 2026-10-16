"""Milestone 0.6: frozen eval sets, prompt adapters, scorers, dev/final split and loader.

Full byte-identical rebuilds of every set take ~40 s and need the raw sources; they run with HARNESS_RL_SLOW=1.
Reduced-size rebuilds (determinism by seed) always run when the sources are present.
"""

import ast
import json
import os
from collections import Counter
from fractions import Fraction
from pathlib import Path

import pytest

from harness_rl.core.env_paths import OMANIC_REWARD
from harness_rl.core.versioning import file_hash
from harness_rl.evals import FinalSplitLocked, list_sets, load_manifest, load_set
from harness_rl.evals import common, prompts, scoring
from harness_rl.evals.common import SETS_DIR, sha256_bytes

SLOW = os.environ.get("HARNESS_RL_SLOW") == "1"
EXPECTED_SETS = (["omanic_heldout", "musique_ans_dev", "2wiki_dev", "hotpotqa_distractor_dev"]
                 + [f"babilong_qa{t}_{l}" for t in range(1, 6) for l in ("0k", "4k", "8k")]
                 + ["ruler_vt_4k", "ruler_vt_8k"])
EXPECTED_SIZES = {"omanic_heldout": 967, "musique_ans_dev": 500, "2wiki_dev": 500, "hotpotqa_distractor_dev": 500,
                  **{f"babilong_qa{t}_{l}": 100 for t in range(1, 6) for l in ("0k", "4k", "8k")},
                  "ruler_vt_4k": 200, "ruler_vt_8k": 200}
NO_SPLIT = {"omanic_heldout"}
NATURAL = [s for s in EXPECTED_SETS if s not in NO_SPLIT]
TASK_KEYS = {"id", "set", "split", "prompt", "answer", "aliases", "scorer", "meta"}
RAW = common.RAW_DIR
needs_raw = pytest.mark.skipif(not (RAW / "2wiki/extracted/dev.json").exists(), reason="raw sources not downloaded")


def all_rows(name):
    return load_set(name, ("dev", "final"), allow_final=True)


@pytest.fixture(scope="module")
def rows():
    return {name: all_rows(name) for name in EXPECTED_SETS}


# ------------------------------------------------------------------------------------------------ sets & manifests

ADAPTER_SETS = ["multihop_grid"]  # built by adapter builders (tested in test_multihop.py), not by evals/build.py


def test_all_sets_present():
    assert list_sets() == sorted(EXPECTED_SETS + ADAPTER_SETS)


@pytest.mark.parametrize("name", EXPECTED_SETS)
def test_manifest_hashes_match_files(name):
    m = load_manifest(name)
    data = (SETS_DIR / name / "data.jsonl").read_bytes()
    assert m["sha256"] == sha256_bytes(data) == file_hash(SETS_DIR / name / "data.jsonl")
    lines = data.splitlines(keepends=True)
    assert m["size"] == len(lines) == EXPECTED_SIZES[name]
    by_split = {"dev": [], "final": []}
    for line in lines:
        by_split[json.loads(line)["split"]].append(line)
    for s, v in by_split.items():
        assert m["splits"][s] == {"n": len(v), "sha256": sha256_bytes(b"".join(v))}
    # the generator recorded in the manifest is the code in the tree (a stale manifest fails here)
    for path, h in m["generator"]["files"].items():
        assert file_hash(common.REPO_ROOT / path) == h, f"{name}: {path} changed since the set was built"
    assert m["prompt_adapter"] == prompts.get_adapter(m["prompt_adapter"]["name"]).manifest()
    assert m["seed"] == 0 and m["source"] and m["source_split"]


@pytest.mark.parametrize("name", EXPECTED_SETS)
def test_canonical_lines_and_schema(name, rows):
    data = (SETS_DIR / name / "data.jsonl").read_bytes()
    for line, r in zip(data.splitlines(), rows[name]):
        assert set(r) == TASK_KEYS
        assert line.decode() == common.dumps_line(r)
        assert r["set"] == name and r["scorer"] in scoring.SCORERS
        assert isinstance(r["answer"], str) and r["answer"]
        assert isinstance(r["aliases"], list) and r["answer"] not in r["aliases"]
    assert len({r["id"] for r in rows[name]}) == len(rows[name])


@pytest.mark.parametrize("name", EXPECTED_SETS)
def test_split_proportions(name, rows):
    n = Counter(r["split"] for r in rows[name])
    total = len(rows[name])
    if name in NO_SPLIT:
        assert n == {"dev": total}
        return
    assert n["final"] == common.round_half_up(Fraction(total, 5)) and n["dev"] + n["final"] == total
    dev = {r["id"] for r in rows[name] if r["split"] == "dev"}
    final = {r["id"] for r in rows[name] if r["split"] == "final"}
    assert not dev & final
    # stratified: each stratum's final count is within 1 of 20% of the stratum
    for k, c in load_manifest(name)["split_rule"]["per_stratum"].items():
        assert abs(c["final"] - Fraction(c["dev"] + c["final"], 5)) < 1, (k, c)


def test_split_sizes_table(rows):
    sizes = {name: Counter(r["split"] for r in rows[name]) for name in NATURAL}
    assert sizes["musique_ans_dev"] == {"dev": 400, "final": 100}
    assert sizes["ruler_vt_8k"] == {"dev": 160, "final": 40}
    assert sizes["babilong_qa3_4k"] == {"dev": 80, "final": 20}


# ------------------------------------------------------------------------------------------------ loader

def test_load_set_refuses_final():
    for splits in [("final",), ("dev", "final"), "final"]:
        with pytest.raises(FinalSplitLocked):
            load_set("musique_ans_dev", splits)
    assert issubclass(FinalSplitLocked, PermissionError)
    assert {r["split"] for r in load_set("musique_ans_dev", ("final",), allow_final=True)} == {"final"}
    with pytest.raises(ValueError):
        load_set("musique_ans_dev", ("test",))


@pytest.mark.parametrize("name", NATURAL)
def test_no_final_leak_into_dev(name, rows):
    dev = load_set(name)  # default: dev only
    assert dev and all(r["split"] == "dev" for r in dev)
    final = [r for r in rows[name] if r["split"] == "final"]
    assert not {r["id"] for r in dev} & {r["id"] for r in final}
    assert not {r["prompt"] for r in dev} & {r["prompt"] for r in final}
    if "question" in dev[0]["meta"] and not name.startswith("babilong"):
        assert not {r["meta"]["question"] for r in dev} & {r["meta"]["question"] for r in final}


def test_load_set_detects_tampering(tmp_path):
    d = tmp_path / "hotpotqa_distractor_dev"
    d.mkdir()
    (d / "manifest.json").write_bytes((SETS_DIR / d.name / "manifest.json").read_bytes())
    (d / "data.jsonl").write_bytes((SETS_DIR / d.name / "data.jsonl").read_bytes().replace(b'"dev"', b'"dev" ', 1))
    with pytest.raises(ValueError, match="sha256"):
        load_set(d.name, root=tmp_path)


# ------------------------------------------------------------------------------------------------ prompts

@pytest.mark.parametrize("name", EXPECTED_SETS)
def test_prompts_stripped_with_answer_line(name, rows):
    adapter = prompts.get_adapter(load_manifest(name)["prompt_adapter"]["name"])
    for r in rows[name]:
        p = r["prompt"]
        assert p == p.strip()
        assert p.endswith(prompts.ANSWER_INSTRUCTION + "\n" + adapter.answer_line)
        assert "\nAnswer: <" in p


def test_qa_prompt_format(rows):
    r = rows["musique_ans_dev"][0]
    p = r["prompt"]
    assert p.startswith("Answer the following question using the paragraphs below. Answering it requires several "
                        "reasoning steps.\n\n[P1] ")
    assert f"\n\nQuestion: {r['meta']['question']}\n\nReason step by step." in p
    assert p.count("\n[P") == 20 and "[P20] " in p  # MuSiQue: 20 paragraphs
    assert prompts.format_paragraphs([("T1", " a "), ("T2", "b")]) == "[P1] T1\na\n\n[P2] T2\nb"


def test_omanic_prompts_verbatim(rows):
    pd = pytest.importorskip("pandas")
    df = pd.read_parquet(common.Path("/home/ubuntu/data/omanic/test_shuffled.parquet"))
    src = [p[0]["content"] for p in df.prompt]
    assert [r["prompt"] for r in rows["omanic_heldout"]] == src
    labels = [json.loads(g["ground_truth"])["label"] for g in df.reward_model]
    assert [r["answer"] for r in rows["omanic_heldout"]] == labels


def test_render_is_single_pass():
    a = prompts.get_adapter("hotpotqa_ctx")
    p = a.render(paragraphs="[P1] T\ncontains {question}", question="Q?")
    assert "contains {question}" in p
    with pytest.raises(KeyError):
        a.render(paragraphs="x")
    with pytest.raises(ValueError):
        prompts.check_prompt(" x\n" + prompts.answer_suffix(a.answer_line), a.answer_line)


def test_chat_rendering_matches_inventory():
    assert common.chat_text("Q?") == "<|im_start|>user\nQ?<|im_end|>\n<|im_start|>assistant\n"


@pytest.mark.parametrize("name", [n for n in EXPECTED_SETS if n.startswith(("babilong", "ruler"))])
def test_token_buckets(name, rows):
    lo, hi = load_manifest(name)["token_bucket"]
    toks = [r["meta"]["prompt_tokens"] for r in rows[name]]
    assert all(lo < t <= hi for t in toks), (name, min(toks), max(toks))
    for r in rows[name][:3]:  # stored counts are real Qwen3 counts
        assert common.prompt_tokens(r["prompt"]) == r["meta"]["prompt_tokens"]


def test_ruler_items_consistent(rows):
    for name in ["ruler_vt_4k", "ruler_vt_8k"]:
        for r in rows[name]:
            m, p = r["meta"], r["prompt"]
            vars_ = m["vars"]
            assert len(vars_) == 5 == len(set(vars_)) and r["answer"] == " ".join(vars_)
            assert f"VAR {vars_[0]} = {m['value']}" in p
            for a, b in zip(vars_, vars_[1:]):
                assert f"VAR {b} = VAR {a} " in p
            assert f"assigned the value {m['value']} in the text above.\n\nReason step by step." in p
            # the ICL example has a different value and 3-letter names
            assert p.count(m["value"]) == 2


@needs_raw
def test_babilong_prompts_and_labels_match_official():
    repo = RAW / "babilong_code/repo/babilong"
    consts = {}
    for node in ast.parse((repo / "prompts.py").read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                consts[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    assert consts["TASK_TEMPLATE"] == prompts.BABILONG_TASK_TEMPLATE
    assert consts["USER_TEMPLATE"] == prompts.BABILONG_USER_TEMPLATE
    for t in prompts.BABILONG_PROMPTS:
        assert consts["DEFAULT_PROMPTS"][t] == prompts.BABILONG_PROMPTS[t]
    labels = ast.literal_eval(ast.parse((repo / "metrics.py").read_text()).body[0].value)
    for t, v in scoring.BABILONG_TASK_LABELS.items():
        assert labels[t] == v


@needs_raw
def test_ruler_template_matches_official():
    path = RAW / "ruler/repo/scripts/data/synthetic/constants.py"
    tasks = ast.literal_eval(ast.parse(path.read_text()).body[-1].value)
    vt = tasks["variable_tracking"]
    assert vt["template"] == prompts.RULER_VT_TASK_TEMPLATE
    assert vt["answer_prefix"] == prompts.RULER_VT_ANSWER_PREFIX


# ------------------------------------------------------------------------------------------------ determinism

REDUCED = {"omanic_heldout": 20, "musique_ans_dev": 40, "2wiki_dev": 40, "hotpotqa_distractor_dev": 40,
           "babilong_qa2_8k": 10, "babilong_qa5_0k": 10, "ruler_vt_4k": 3, "ruler_vt_8k": 2}


@needs_raw
@pytest.mark.parametrize("name", list(REDUCED))
def test_reduced_rebuild_deterministic(name):
    from harness_rl.evals import build

    a = build.build(name, seed=0, size=REDUCED[name])
    b = build.build(name, seed=0, size=REDUCED[name])
    assert a == b
    if name in ("musique_ans_dev", "2wiki_dev", "hotpotqa_distractor_dev", "ruler_vt_4k"):
        assert build.build(name, seed=1, size=REDUCED[name])[0] != a[0]


@needs_raw
def test_reduced_sample_is_stratified():
    from harness_rl.evals import build

    data, manifest = build.build("musique_ans_dev", size=40)
    m = json.loads(manifest)
    assert sum(m["sampling"]["per_stratum"].values()) == 40
    assert m["splits"]["final"]["n"] == 8


@pytest.mark.skipif(not SLOW, reason="full rebuild (~40 s); set HARNESS_RL_SLOW=1")
@needs_raw
@pytest.mark.parametrize("name", EXPECTED_SETS)
def test_full_rebuild_byte_identical(name):
    from harness_rl.evals import build

    data, manifest = build.build(name)
    assert data == (SETS_DIR / name / "data.jsonl").read_bytes()
    assert manifest == (SETS_DIR / name / "manifest.json").read_bytes()


def test_allocation_exact():
    alloc = common.allocate({"a": 5, "b": 3, "c": 2}, 5)
    assert sum(alloc.values()) == 5 and alloc == {"a": 3, "b": 1, "c": 1}
    items = [{"id": f"x{i}", "s": "abc"[i % 3]} for i in range(101)]
    split, per = common.assign_splits(items, stratum=lambda r: r["s"], item_id=lambda r: r["id"], seed=0, set_name="t")
    assert Counter(split.values()) == {"dev": 81, "final": 20}
    assert split == common.assign_splits(items, stratum=lambda r: r["s"], item_id=lambda r: r["id"], seed=0,
                                         set_name="t")[0]


# ------------------------------------------------------------------------------------------------ scorers

@pytest.mark.parametrize("raw,norm", [
    ("The Eiffel Tower!", "eiffel tower"),
    ("  an  Apple a day ", "apple day"),
    ("U.S.A.", "usa"),
    ("Theater", "theater"),
    ("1,000", "1000"),
    ("Paris, France", "paris france"),
])
def test_squad_normalization(raw, norm):
    assert scoring.normalize_answer(raw) == norm


def test_em_f1():
    assert scoring.exact_match("the Beatles", "Beatles") == 1.0
    assert scoring.f1("Barack Obama", "Obama") == pytest.approx(2 / 3)
    assert scoring.f1("Paris", "London") == 0.0
    assert scoring.f1("yes it is", "yes") == 0.0  # HotpotQA/2Wiki yes/no rule
    assert scoring.f1("yes", "yes") == 1.0
    assert scoring.f1("", "x") == 0.0


def test_qa_aliases_and_max_over_golds():
    task = {"answer": "New York City", "aliases": ["NYC", "New York"], "scorer": "qa_em_f1", "meta": {}}
    assert scoring.score(task, "Answer: NYC") | {"pred": None} == {
        "score": 1.0, "em": 1.0, "f1": 1.0, "format_ok": 1.0, "pred": None}
    s = scoring.score(task, "Answer: New York State")
    assert s["em"] == 0.0 and s["f1"] == pytest.approx(0.8)  # vs "New York": p=2/3, r=1
    assert scoring.score(task, "no answer line")["format_ok"] == 0.0
    assert scoring.score(task, "**Answer:** **NYC**")["em"] == 1.0


@pytest.mark.parametrize("resp,ans", [
    ("<think>\nAnswer: wrong\n</think>\n\nIt is Paris.\nAnswer: Paris", "Paris"),
    ("<think>Answer: X</think>\nI am not sure.", None),
    ("<think>a</think><think>Answer: Y</think>\nAnswer: Z", "Z"),
    ("**Answer:** Paris", "Paris"),
    ("**Answer:** **Paris**", "**Paris"),  # same as musique/reward.py::parse; normalization drops the '*'
    ("answer: Paris\nAnswer: Rome\n", "Rome"),
    ("Answer:   ", None),
    ("Answer: The most recent location of Mary is kitchen.", "The most recent location of Mary is kitchen."),
    ("no think block\nAnswer: 42", "42"),
])
def test_extract_answer(resp, ans):
    assert scoring.extract_answer(resp) == ans


def test_omanic_scorer_uses_existing_reward():
    assert Path(scoring.omanic_reward().__file__) == OMANIC_REWARD
    task = {"answer": "B", "aliases": [], "scorer": "omanic_letter", "meta": {"answer_text": "x", "has_math_hop": 1}}
    assert scoring.score(task, "<think>Answer: A</think>\nAnswer: B")["score"] == 1.0
    assert scoring.score(task, "Answer: B or C")["format_ok"] == 0.0
    assert scoring.score(task, "Answer: A")["score"] == 0.0


def test_babilong_rule():
    def s(task, target, answer, question):
        t = {"answer": target, "aliases": [], "scorer": "babilong_match", "meta": {"task": task, "question": question}}
        return scoring.score(t, f"<think>bedroom?</think>\nAnswer: {answer}")["score"]

    assert s("qa1", "kitchen", "The most recent location of Mary is kitchen.", "Where is Mary?") == 1.0
    assert s("qa1", "kitchen", "kitchen or bedroom", "Where is Mary?") == 0.0  # two labels
    assert s("qa1", "kitchen", "Kitchen. Earlier she was in the bedroom.", "Where is Mary?") == 1.0  # 1st sentence
    assert s("qa3", "bathroom", "Before the kitchen the apple was in the bathroom.",
             "Where was the apple before the kitchen?") == 1.0  # labels in the question are ignored
    assert s("qa5", "Fred", "Fred", "Who did Mary give the apple to?") == 1.0
    assert s("qa5", "apple", "the football", "What did Fred give to Bill?") == 0.0
    assert scoring.babilong_compare_answers("apple,milk", "apple and milk", "What is Daniel carrying?",
                                            ["apple", "football", "milk", "nothing"])


def test_ruler_vt_rule():
    refs = ["ABCDE", "FGHIJ", "KLMNO", "PQRST", "UVWXY"]
    task = {"answer": " ".join(refs), "aliases": [], "scorer": "ruler_vt_recall", "meta": {"vars": refs}}
    full = scoring.score(task, "<think>ZZZZZ</think>\nAnswer: " + " ".join(refs))
    assert full["score"] == full["all_match"] == full["exact_set"] == 1.0
    part = scoring.score(task, "Answer: ABCDE, fghij, KLMNO")
    assert part["score"] == pytest.approx(0.6) and part["all_match"] == 0.0
    extra = scoring.score(task, "Answer: they are: " + " ".join(refs) + " ZZZZZ")
    assert extra["score"] == 1.0 and extra["exact_set"] == 0.0  # RULER recall ignores extra names
    assert scoring.score(task, "<think>Answer: " + " ".join(refs) + "</think>\nnone")["score"] == 0.0


@pytest.mark.parametrize("name", EXPECTED_SETS)
def test_gold_answer_scores_one(name, rows):
    """Every item is solvable under its own scorer when the model answers with the gold string."""
    for r in rows[name]:
        out = scoring.score(r, f"<think>\nAnswer: wrong\n</think>\n\nReasoning.\nAnswer: {r['answer']}")
        assert out["score"] == 1.0 and out["format_ok"] == 1.0, (r["id"], out)
        assert scoring.score(r, "<think>\nAnswer: " + r["answer"] + "\n</think>\n\nnothing")["score"] == 0.0
