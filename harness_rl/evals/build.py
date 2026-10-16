"""Build the frozen eval sets under harness_rl/evals/sets/<name>/{data.jsonl,manifest.json}.

    python -m harness_rl.evals.build                    # (re)build every set in place
    python -m harness_rl.evals.build --sets 2wiki_dev   # some sets
    python -m harness_rl.evals.build --check            # rebuild in memory, compare bytes with the files on disk
    python -m harness_rl.evals.build --out /tmp/x       # write elsewhere

Same seed => byte-identical files. Sources are read from fixed local paths (see SOURCES / README.md); nothing is
downloaded here and no code from a download is executed.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Callable

from harness_rl.core.versioning import file_hash
from harness_rl.evals import common, ruler_vt
from harness_rl.evals.common import RAW_DIR, SETS_DIR, dumps_line, dumps_manifest, rel, sha256_bytes
from harness_rl.evals.prompts import format_paragraphs, get_adapter, omanic_preprocess

SEED = 0
GENERATOR_FILES = [Path(__file__), Path(common.__file__), Path(ruler_vt.__file__),
                   Path(common.EVALS_DIR / "prompts.py")]

OMANIC_PARQUET = Path("/home/ubuntu/data/omanic/test_shuffled.parquet")
OMANIC_RAW = Path("/home/ubuntu/playground/data/omanic/OmanicBench.jsonl")
MUSIQUE_DEV = Path("/home/ubuntu/playground/data/musique/musique_ans_v1.0_dev.jsonl")
TWOWIKI_DIR = RAW_DIR / "2wiki"
HOTPOT_PARQUET = RAW_DIR / "hotpotqa/distractor/validation-00000-of-00001.parquet"
BABILONG_DIR = RAW_DIR / "babilong"

QA_SIZE = 500
BABILONG_TASKS = ["qa1", "qa2", "qa3", "qa4", "qa5"]
BABILONG_LENGTHS = ["0k", "4k", "8k"]
# Nominal BABILong length -> allowed prompt-token range (lo, hi], Qwen3 tokens with the chat template. BABILong's
# "4k"/"8k" are context budgets in its own tokenizer; with Qwen3 the contexts measure ~2.2k-4.5k and ~5.1k-8.3k,
# plus ~300-450 tokens of instructions and examples.
BABILONG_BUCKETS = {"0k": (0, 2048), "4k": (2048, 5120), "8k": (5120, 9216)}
RULER_LENGTHS = {"4k": 4096, "8k": 8192}
RULER_SIZE = 200
RULER_BUCKET_SLACK = 64  # every RULER prompt lands in (L - 64, L]


@dataclass
class Built:
    rows: list[dict]                      # id, prompt, answer, aliases, meta (no split yet), in output order
    adapter: str
    scorer: str
    source: dict
    source_split: str
    sampling: dict
    stratum: Callable[[dict], str] | None  # None: no dev/final split (all dev)
    stratified_by: str = ""
    extra: dict = field(default_factory=dict)


def _hf(repo: str, revision: str, path: str, local: Path) -> dict:
    return {"kind": "huggingface", "repo": repo, "revision": revision, "file": path, "local_path": str(local),
            "sha256": file_hash(local)}


# ------------------------------------------------------------------------------------------------ builders

def build_omanic(name: str, seed: int, size: int | None = None) -> Built:
    import pandas as pd

    df = pd.read_parquet(OMANIC_PARQUET)
    adapter = get_adapter("omanic_mc")
    rows = []
    for i, r in enumerate(df.itertuples(index=False)):
        gt = json.loads(r.reward_model["ground_truth"])
        info = r.extra_info
        rows.append({
            "id": info["id"],
            "prompt": adapter.render(prompt=r.prompt[0]["content"]),
            "answer": gt["label"],
            "aliases": [],
            "meta": {"source_index": i, "answer_text": gt["answer"], "motif": info["motif"],
                     "has_math_hop": int(info["has_math_hop"]), "original_label": info["original_label"]},
        })
    rows = rows[:size] if size else rows
    pre = omanic_preprocess()
    return Built(
        rows, "omanic_mc", "omanic_letter",
        source={"kind": "local_parquet", "path": str(OMANIC_PARQUET), "sha256": file_hash(OMANIC_PARQUET),
                "produced_by": {"script": "/home/ubuntu/playground/omanic/preprocess.py",
                                "sha256": file_hash(Path(pre.__file__)),
                                "call": "multi_hop_rows(OmanicBench, 'test_shuffled', shuffle=True, seed=1)"},
                "upstream": {"kind": "huggingface", "repo": "li-lab/Omanic",
                             "revision": "edbe032a2ea714a28a0e80ad93cfd0e6e7456e2f", "file": "OmanicBench.jsonl",
                             "local_path": str(OMANIC_RAW), "sha256": file_hash(OMANIC_RAW)}},
        source_split="OmanicBench (all 967 questions), options shuffled with seed 1 = test_shuffled.parquet",
        sampling={"method": "all rows, parquet order", "n": len(rows), "population": len(df)},
        stratum=None,
        extra={"split_note": "Training-distribution (closed-book, synthetic-style) held-out set: the dev/final rule "
                             "applies to natural sets only, so every item is dev."},
    )


def _qa_rows(items, *, adapter_name, id_of, question_of, paragraphs_of, answer_of, aliases_of, meta_of):
    adapter = get_adapter(adapter_name)
    rows = []
    for idx, x in items:
        answer = answer_of(x).strip()
        aliases = []
        for a in aliases_of(x):
            a = a.strip()
            if a and a != answer and a not in aliases:
                aliases.append(a)
        rows.append({
            "id": id_of(x),
            "prompt": adapter.render(paragraphs=format_paragraphs(paragraphs_of(x)), question=question_of(x).strip()),
            "answer": answer,
            "aliases": aliases,
            "meta": {"source_index": idx, "question": question_of(x).strip(), **meta_of(x)},
        })
    return rows


def build_musique(name: str, seed: int, size: int | None = None) -> Built:
    with open(MUSIQUE_DEV, encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    hop_type = lambda x: x["id"].split("__")[0]  # 2hop, 3hop1, 3hop2, 4hop1, 4hop2, 4hop3
    indexed = list(enumerate(data))
    picked, per_stratum = common.stratified_sample(indexed, size or QA_SIZE, stratum=lambda t: hop_type(t[1]),
                                                   item_id=lambda t: t[1]["id"], seed=seed, set_name=name)
    rows = _qa_rows(
        picked, adapter_name="musique_ctx", id_of=lambda x: x["id"], question_of=lambda x: x["question"],
        paragraphs_of=lambda x: [(p["title"], p["paragraph_text"]) for p in x["paragraphs"]],
        answer_of=lambda x: x["answer"], aliases_of=lambda x: x["answer_aliases"],
        meta_of=lambda x: {"hop_type": hop_type(x), "hops": int(x["id"][0]),
                           "supporting": [i + 1 for i, p in enumerate(x["paragraphs"]) if p["is_supporting"]]})
    return Built(
        rows, "musique_ctx", "qa_em_f1",
        source={**_hf("dgslibisey/MuSiQue", "c8f4f8c9465fb69d31a8eae894c3fd509c4ca321", "musique_ans_v1.0_dev.jsonl",
                      MUSIQUE_DEV), "note": "local copy downloaded by the playground (revision from its HF cache)"},
        source_split="MuSiQue-Ans v1.0 dev (2,417; answerable only)",
        sampling={"method": "stratified, largest-remainder allocation, lowest sha256(seed|sample|set|id) first",
                  "stratified_by": "hop type (id prefix)", "n": len(rows), "population": len(data),
                  "per_stratum": per_stratum},
        stratum=lambda r: r["meta"]["hop_type"], stratified_by="hop type (id prefix)",
    )


def build_2wiki(name: str, seed: int, size: int | None = None) -> Built:
    dev_path, alias_path = TWOWIKI_DIR / "extracted/dev.json", TWOWIKI_DIR / "extracted/id_aliases.json"
    data = json.loads(dev_path.read_text(encoding="utf-8"))
    indexed = list(enumerate(data))
    picked, per_stratum = common.stratified_sample(indexed, size or QA_SIZE, stratum=lambda t: t[1]["type"],
                                                   item_id=lambda t: t[1]["_id"], seed=seed, set_name=name)
    need = {x["answer_id"] for _, x in picked}
    aliases = {}
    with open(alias_path, encoding="utf-8") as f:
        for line in f:
            a = json.loads(line)
            if a["Q_id"] in need:  # official eval: aliases + demonyms of the answer's Wikidata id
                aliases[a["Q_id"]] = sorted(set(a["aliases"] + a["demonyms"]))
    rows = _qa_rows(
        picked, adapter_name="2wiki_ctx", id_of=lambda x: x["_id"], question_of=lambda x: x["question"],
        paragraphs_of=lambda x: [(t, " ".join(s.strip() for s in sents)) for t, sents in x["context"]],
        answer_of=lambda x: x["answer"], aliases_of=lambda x: aliases.get(x["answer_id"], []),
        meta_of=lambda x: {"type": x["type"], "answer_id": x["answer_id"],
                           "supporting_titles": sorted({t for t, _ in x["supporting_facts"]})})
    zip_path = TWOWIKI_DIR / "data_ids_april7.zip"
    return Built(
        rows, "2wiki_ctx", "qa_em_f1",
        source={"kind": "url", "url": "https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?dl=1",
                "listed_in": "https://github.com/Alab-NII/2wikimultihop README @ 13800e5be57df1b4040b9b1588c6c811779e69e9",
                "local_path": str(zip_path), "sha256": file_hash(zip_path),
                "members": {"dev.json": file_hash(dev_path), "id_aliases.json": file_hash(alias_path)}},
        source_split="2WikiMultihopQA dev (12,576), data_ids_april7 release",
        sampling={"method": "stratified, largest-remainder allocation, lowest sha256(seed|sample|set|id) first",
                  "stratified_by": "question type", "n": len(rows), "population": len(data),
                  "per_stratum": per_stratum},
        stratum=lambda r: r["meta"]["type"], stratified_by="question type",
    )


def build_hotpotqa(name: str, seed: int, size: int | None = None) -> Built:
    import pandas as pd

    df = pd.read_parquet(HOTPOT_PARQUET)
    data = [{"id": r.id, "question": r.question, "answer": r.answer, "type": r.type, "level": r.level,
             "titles": [str(t) for t in r.context["title"]],
             "sentences": [[str(s) for s in sents] for sents in r.context["sentences"]],
             "sp_titles": sorted({str(t) for t in r.supporting_facts["title"]})}
            for r in df.itertuples(index=False)]
    indexed = list(enumerate(data))
    picked, per_stratum = common.stratified_sample(indexed, size or QA_SIZE, stratum=lambda t: t[1]["type"],
                                                   item_id=lambda t: t[1]["id"], seed=seed, set_name=name)
    rows = _qa_rows(
        picked, adapter_name="hotpotqa_ctx", id_of=lambda x: x["id"], question_of=lambda x: x["question"],
        paragraphs_of=lambda x: [(t, " ".join(s.strip() for s in sents)) for t, sents in zip(x["titles"], x["sentences"])],
        answer_of=lambda x: x["answer"], aliases_of=lambda x: [],
        meta_of=lambda x: {"type": x["type"], "level": x["level"], "supporting_titles": x["sp_titles"]})
    return Built(
        rows, "hotpotqa_ctx", "qa_em_f1",
        source=_hf("hotpotqa/hotpot_qa", "1908d6afbbead072334abe2965f91bd2709910ab",
                   "distractor/validation-00000-of-00001.parquet", HOTPOT_PARQUET),
        source_split="HotpotQA distractor validation (7,405; = hotpot_dev_distractor_v1.json)",
        sampling={"method": "stratified, largest-remainder allocation, lowest sha256(seed|sample|set|id) first",
                  "stratified_by": "question type (bridge/comparison)", "n": len(rows), "population": len(data),
                  "per_stratum": per_stratum},
        stratum=lambda r: r["meta"]["type"], stratified_by="question type (bridge/comparison)",
    )


def build_babilong(name: str, seed: int, size: int | None = None) -> Built:
    _, task, length = name.split("_")
    path = BABILONG_DIR / f"data/{task}/{length}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    adapter = get_adapter(f"babilong_{task}")
    rows = []
    for i, x in enumerate(data[:size] if size else data):
        rows.append({
            "id": f"{task}-{length}-{i:03d}",
            "prompt": adapter.render(context=x["input"].strip(), question=x["question"].strip()),
            "answer": x["target"],
            "aliases": [],
            "meta": {"source_index": i, "task": task, "length": length, "question": x["question"].strip()},
        })
    return Built(
        rows, f"babilong_{task}", "babilong_match",
        source=_hf("RMT-team/babilong", "ee0d588794c7ac098062ee0d247c733d62e94fe2", f"data/{task}/{length}.json", path)
        | {"prompts_and_metric": "https://github.com/booydar/babilong @ 7a6efee29f5cac03c3c410e6799c80fd2ffe3610 "
                                 "(babilong/prompts.py DEFAULT_PROMPTS/DEFAULT_TEMPLATE, babilong/metrics.py)"},
        source_split=f"config {length}, split {task} (100 samples: the whole split)",
        sampling={"method": "all rows, file order", "n": len(rows), "population": len(data)},
        stratum=lambda r: r["answer"], stratified_by="target answer",
        extra={"token_bucket": list(BABILONG_BUCKETS[length])},
    )


def build_ruler_vt(name: str, seed: int, size: int | None = None) -> Built:
    length = name.rsplit("_", 1)[1]
    target = RULER_LENGTHS[length]
    samples = ruler_vt.generate(seed, name, target, size or RULER_SIZE)
    rows = [{
        "id": f"vt-{length}-{s['index']:04d}",
        "prompt": s["prompt"],
        "answer": " ".join(s["vars"]),
        "aliases": [],
        "meta": {"source_index": s["index"], "vars": s["vars"], "value": s["value"], "num_noises": s["num_noises"],
                 "target_tokens": target, "num_chains": ruler_vt.NUM_CHAINS, "num_hops": ruler_vt.NUM_HOPS},
    } for s in samples]
    return Built(
        rows, "ruler_vt", "ruler_vt_recall",
        source={"kind": "generator", "repo": "https://github.com/NVIDIA/RULER",
                "commit": "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a",
                "files": ["scripts/data/synthetic/variable_tracking.py", "scripts/data/synthetic/constants.py",
                          "scripts/synthetic.yaml (vt)", "scripts/eval/synthetic/constants.py"],
                "config": ruler_vt.CONFIG, "target_tokens": target},
        source_split=f"synthetic, target {target} prompt tokens",
        sampling={"method": "generated; sample i uses random.Random(derive_int(seed, set, 'sample', i))",
                  "n": len(rows)},
        stratum=lambda r: "all", stratified_by="none (single stratum)",
        extra={"token_bucket": [target - RULER_BUCKET_SLACK, target]},
    )


BUILDERS: dict[str, Callable] = {
    "omanic_heldout": build_omanic,
    "musique_ans_dev": build_musique,
    "2wiki_dev": build_2wiki,
    "hotpotqa_distractor_dev": build_hotpotqa,
    **{f"babilong_{t}_{l}": build_babilong for t in BABILONG_TASKS for l in BABILONG_LENGTHS},
    **{f"ruler_vt_{l}": build_ruler_vt for l in RULER_LENGTHS},
}
SET_NAMES = list(BUILDERS)


# ------------------------------------------------------------------------------------------------ assembly

def build(name: str, seed: int = SEED, size: int | None = None) -> tuple[bytes, bytes]:
    """Return (data.jsonl bytes, manifest.json bytes) for one set. `size` truncates/limits (tests only)."""
    b = BUILDERS[name](name, seed, size)
    ids = [r["id"] for r in b.rows]
    assert len(set(ids)) == len(ids), f"{name}: duplicate ids"

    if b.stratum is None:
        split_of = {i: "dev" for i in ids}
        split_rule = {"rule": "none: all dev", "note": b.extra.pop("split_note", "")}
    else:
        split_of, per_stratum = common.assign_splits(b.rows, stratum=b.stratum, item_id=lambda r: r["id"],
                                                     seed=seed, set_name=name)
        split_rule = {"rule": "80% dev / 20% final; n_final = round_half_up(N/5), apportioned over strata by largest "
                              "remainder; lowest sha256(seed|split|set|id) in each stratum go to final",
                      "final_fraction": str(common.FINAL_FRACTION), "stratified_by": b.stratified_by,
                      "per_stratum": per_stratum}

    lines_by_split = {"dev": [], "final": []}
    lines = []
    tokens = []
    for r in b.rows:
        n_tok = common.prompt_tokens(r["prompt"])
        tokens.append(n_tok)
        task = {"id": r["id"], "set": name, "split": split_of[r["id"]], "prompt": r["prompt"], "answer": r["answer"],
                "aliases": r["aliases"], "scorer": b.scorer, "meta": {**r["meta"], "prompt_tokens": n_tok}}
        assert task["prompt"] == task["prompt"].strip()
        line = (dumps_line(task) + "\n").encode()
        lines.append(line)
        lines_by_split[task["split"]].append(line)
    data = b"".join(lines)

    adapter = get_adapter(b.adapter)
    manifest = {
        "name": name,
        "schema_version": common.SCHEMA_VERSION,
        "sha256": sha256_bytes(data),
        "size": len(lines),
        "seed": seed,
        "splits": {s: {"n": len(v), "sha256": sha256_bytes(b"".join(v))} for s, v in lines_by_split.items()},
        "split_rule": split_rule,
        "source": b.source,
        "source_split": b.source_split,
        "sampling": b.sampling,
        "scorer": b.scorer,
        "prompt_adapter": adapter.manifest(),
        "generator": {"command": f"python -m harness_rl.evals.build --sets {name}",
                      "files": {rel(p): file_hash(p) for p in GENERATOR_FILES}},
        "tokenizer": {"id": common.TOKENIZER_ID, "revision": common.TOKENIZER_REVISION,
                      "counted": "Qwen3 chat template, one user message, add_generation_prompt, enable_thinking=True"},
        "prompt_tokens": {"min": min(tokens), "mean": round(statistics.fmean(tokens), 1), "max": max(tokens)},
        "file_format": "one JSON object per line: id, set, split, prompt, answer, aliases, scorer, meta; "
                       "sorted keys, separators (',', ':'), ensure_ascii=False, '\\n' line ends",
        **b.extra,
    }
    if size is not None:
        manifest["reduced_size"] = size
    return data, dumps_manifest(manifest).encode()


def write(name: str, out_root: Path, seed: int = SEED) -> dict:
    data, manifest = build(name, seed)
    d = Path(out_root) / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "data.jsonl").write_bytes(data)
    (d / "manifest.json").write_bytes(manifest)
    return json.loads(manifest)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="*", default=SET_NAMES, choices=SET_NAMES)
    ap.add_argument("--out", type=Path, default=SETS_DIR)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--check", action="store_true", help="compare a fresh build with the files in --out")
    args = ap.parse_args(argv)
    bad = 0
    for name in args.sets:
        if args.check:
            data, manifest = build(name, args.seed)
            d = args.out / name
            same = (d / "data.jsonl").read_bytes() == data and (d / "manifest.json").read_bytes() == manifest
            bad += not same
            print(f"{name:28s} {'identical' if same else 'DIFFERENT'}")
        else:
            m = write(name, args.out, args.seed)
            print(f"{name:28s} dev {m['splits']['dev']['n']:4d} final {m['splits']['final']['n']:4d} "
                  f"tokens mean {m['prompt_tokens']['mean']:7.1f} max {m['prompt_tokens']['max']:5d} "
                  f"sha256 {m['sha256'][:12]}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
