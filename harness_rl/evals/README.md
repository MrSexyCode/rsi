# Eval suite (milestone 0.6): frozen sets, prompt adapters, scorers

```
evals/
  sets/<name>/data.jsonl     one task per line (frozen)
  sets/<name>/manifest.json  hashes, source + revision, sampling, split, adapter template, tokenizer, token stats
  build.py                   builds every set from the local sources (deterministic)
  common.py                  canonical JSON, hash-ranked sampling and dev/final split, token counting
  prompts.py                 prompt adapters (versioned templates)
  ruler_vt.py                RULER variable-tracking generator
  scoring.py                 extract_answer + scorers
  loader.py                  load_set / load_manifest / set_hash / list_sets
  run.py                     runner (separate milestone; not built here)
```

## Using a set

```python
from harness_rl.evals import load_set, set_hash
from harness_rl.evals.scoring import score

tasks = load_set("musique_ans_dev")                      # dev only (default)
tasks = load_set("musique_ans_dev", ("dev", "final"), allow_final=True)   # runner --final only
eval_set_hash = set_hash("musique_ans_dev")              # sha256 of data.jsonl, for every eval record
result = score(task, decoded_response)                   # {"score": ..., components...}
```

`load_set` raises `FinalSplitLocked` (a `PermissionError`) if `final` is requested without `allow_final=True`,
and `ValueError` if `data.jsonl` does not match the manifest's sha256.

## Sets

Prompt tokens = Qwen3-8B tokens of the chat-rendered prompt (one user message, `enable_thinking=True`), which is
what the engine receives.

| Set | dev / final | Source @ revision | mean tok | max tok |
|---|---|---|---|---|
| `omanic_heldout` | 967 / 0 | `~/data/omanic/test_shuffled.parquet` (li-lab/Omanic @ edbe032a, OmanicBench) | 143 | 234 |
| `musique_ans_dev` | 400 / 100 | HF `dgslibisey/MuSiQue` @ c8f4f8c9, `musique_ans_v1.0_dev.jsonl` | 2645 | 4773 |
| `2wiki_dev` | 400 / 100 | official `data_ids_april7.zip` (Dropbox link in Alab-NII/2wikimultihop README @ 13800e5b) | 1094 | 4452 |
| `hotpotqa_distractor_dev` | 400 / 100 | HF `hotpotqa/hotpot_qa` @ 1908d6af, `distractor/validation` | 1496 | 2852 |
| `babilong_qa{1..5}_0k` | 80 / 20 each | HF `RMT-team/babilong` @ ee0d5887, `data/qaK/0k.json` | 189-628 | 1192 |
| `babilong_qa{1..5}_4k` | 80 / 20 each | same, `4k.json` | 3707-3881 | 4731 |
| `babilong_qa{1..5}_8k` | 80 / 20 each | same, `8k.json` | 7554-7749 | 8616 |
| `ruler_vt_4k` | 160 / 40 | generated, NVIDIA/RULER @ c3f5e3b4 spec | 4088 | 4096 |
| `ruler_vt_8k` | 160 / 40 | generated, NVIDIA/RULER @ c3f5e3b4 spec | 8176 | 8186 |

Seed 0 for every set. Per-set numbers and hashes: each `manifest.json`.

- **omanic_heldout** is the Phase 0 "synthetic" held-out split: the existing environment's `test_shuffled`
  parquet, prompts verbatim. It is the training distribution (closed-book), not a natural set, so the dev/final
  rule does not apply and every item is `dev`.
- **MuSiQue / 2Wiki / HotpotQA**: 500 sampled, stratified by hop type (MuSiQue id prefix: 2hop, 3hop1, 3hop2,
  4hop1-3) or question type (2Wiki: compositional, comparison, bridge_comparison, inference; HotpotQA: bridge,
  comparison). All context paragraphs (gold and distractors) in source order. 2Wiki aliases follow the official
  evaluation (`aliases + demonyms` of `answer_id` from `id_aliases.json`); MuSiQue uses `answer_aliases`; HotpotQA
  has none.
- **BABILong**: each `(task, length)` split has exactly 100 samples, so all are used; 15 sets.
- **RULER VT**: 200 per length, see `ruler_vt.py` for the spec mirrored and the deliberate differences.

## Rebuilding

```bash
cd /home/ubuntu/rsi
PY=/home/ubuntu/playground/verl/.venv/bin/python
$PY -m harness_rl.evals.build                    # rebuild all sets in place (~40 s, CPU only)
$PY -m harness_rl.evals.build --check            # rebuild in memory and compare bytes with the committed files
$PY -m harness_rl.evals.build --sets ruler_vt_4k --out /tmp/sets
HARNESS_RL_SLOW=1 $PY -m pytest harness_rl/tests/test_evals.py   # includes full byte-identical rebuilds
```

Same seed gives byte-identical `data.jsonl` and `manifest.json`: sorted keys, separators `(",", ":")`,
`ensure_ascii=False`, no timestamps. The manifest records the sha256 of every generator file, so editing
`build.py`/`common.py`/`prompts.py`/`ruler_vt.py` makes the tests flag the sets as stale until they are rebuilt
(and a changed `data.jsonl` hash means a new `eval_set_hash`).

Sources (read from fixed paths, treated as untrusted data; nothing from them is executed; the BABILong and RULER
code was only read, and the tests compare our copies of their constants with `ast.literal_eval`):

| Path | How it was fetched |
|---|---|
| `/home/ubuntu/data/omanic/test_shuffled.parquet` | existing (`playground/omanic/preprocess.py`) |
| `/home/ubuntu/playground/data/musique/musique_ans_v1.0_dev.jsonl` | existing (HF `dgslibisey/MuSiQue` @ c8f4f8c9) |
| `/home/ubuntu/data/harness_rl_raw/2wiki/data_ids_april7.zip` (+ `extracted/{dev,id_aliases}.json`) | `curl -L https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?dl=1`, members extracted with `zipfile` |
| `/home/ubuntu/data/harness_rl_raw/2wiki/repo` | `git clone https://github.com/Alab-NII/2wikimultihop` @ 13800e5b (eval script reference) |
| `/home/ubuntu/data/harness_rl_raw/hotpotqa/distractor/validation-00000-of-00001.parquet` | `hf_hub_download("hotpotqa/hotpot_qa", ..., revision="1908d6af...")` |
| `/home/ubuntu/data/harness_rl_raw/babilong/data/qa{1..5}/{0k,4k,8k}.json` | `hf_hub_download("RMT-team/babilong", ..., revision="ee0d5887...")` |
| `/home/ubuntu/data/harness_rl_raw/babilong_code/repo` | `git clone https://github.com/booydar/babilong` @ 7a6efee2 (prompts, metric) |
| `/home/ubuntu/data/harness_rl_raw/ruler/repo` | `git clone https://github.com/NVIDIA/RULER` @ c3f5e3b4 (VT spec, template, metric) |

## Task format

```json
{"id": "...", "set": "musique_ans_dev", "split": "dev", "prompt": "...", "answer": "...", "aliases": ["..."],
 "scorer": "qa_em_f1", "meta": {"source_index": 17, "prompt_tokens": 2311, "question": "...", ...}}
```

`prompt` is the single user message (no system message), and always equals `prompt.strip()` (Pi strips stdin
whitespace). `meta` is set-specific (hop type, question type, supporting paragraphs, BABILong task/length, RULER
vars/value/haystack size, Omanic answer text and motif).

## Dev/final split

Every natural set (all except `omanic_heldout`) is split 80% `dev` / 20% `final`:
`n_final = round_half_up(N / 5)`, apportioned over strata (the sampling strata; BABILong: target answer; RULER:
one stratum) by largest remainder, and inside each stratum the items with the smallest
`sha256("<seed>|split|<set>|<id>")` go to `final`. No id, prompt or question is in both splits (tested).
`final` is for the end-of-project report only: `load_set` refuses it without `allow_final=True`.

## Prompts

All prompts end with the training environment's answer instruction, so extraction is the same everywhere:

```
Reason step by step. Then end your response with exactly this line:
Answer: <...>
```

| Adapter (v1) | Sets | Template |
|---|---|---|
| `omanic_mc` | omanic_heldout | Omanic `INSTRUCTION` verbatim (passthrough; checked against the template) |
| `musique_ctx`, `2wiki_ctx`, `hotpotqa_ctx` | the three QA sets | `QA_TEMPLATE` below |
| `babilong_qa1..5` | babilong_* | BABILong `DEFAULT_TEMPLATE` with the task's instruction, examples, post_prompt; then the answer instruction with `Answer: <your answer, in the format given above>` |
| `ruler_vt` | ruler_vt_* | RULER ICL example + `\n\n` + RULER VT task template; then the answer instruction with `Answer: <the names of all the variables, separated by spaces>` |

QA_TEMPLATE (paragraphs as in `playground/musique/preprocess.py`: `[P1] Title\ntext`, blank line between):

```
Answer the following question using the paragraphs below. Answering it requires several reasoning steps.

{paragraphs}

Question: {question}

Reason step by step. Then end your response with exactly this line:
Answer: <the short final answer only: an entity, date, or number>
```

Each manifest stores the adapter's exact template text, its sha256 and version.

## Scorers

`extract_answer(response)`: the text after the last `</think>` (the whole response if there is none, exactly as
the training rewards do), scanned bottom-up for the last line matching `^\s*\**\s*answer\s*\**\s*:\s*(.*?)\s*$`
(case-insensitive); the answer is stripped of whitespace and `*`; empty means no answer. Same rule as
`playground/musique/reward.py::parse`.

| Scorer | Sets | Primary `score` | Components |
|---|---|---|---|
| `omanic_letter` | omanic_heldout | `playground/omanic/reward.py::compute_score`, imported by path, unchanged (letter on the last Answer line; hedges score 0) | `acc`, `format_ok`, `has_math_hop`, `pred` |
| `qa_em_f1` | musique, 2wiki, hotpotqa | max over gold + aliases of SQuAD token F1 | `em` (max SQuAD EM), `f1`, `format_ok`, `pred` |
| `babilong_match` | babilong_* | official `compare_answers` on the extracted answer: lower-case, first sentence only, the target must be the only task label present, ignoring labels that occur in the question | `format_ok`, `pred` |
| `ruler_vt_recall` | ruler_vt_* | RULER `string_match_all` on the extracted answer: fraction of the 5 chain variables contained in it (case-insensitive substring), as RULER reports VT | `all_match` (all 5 found), `exact_set` (the set of 5-letter capitalized names equals the gold set: no extras), `n_pred`, `format_ok`, `pred` |

SQuAD normalization: lower-case, remove punctuation, remove articles a/an/the, collapse whitespace. F1 also
applies the HotpotQA/2Wiki official rule: if either side normalizes to yes/no/noanswer and they differ, F1 = 0.
No answer line means score 0 for every scorer. `pred` is the only non-numeric field.

## Deviations and open points

- The QA answer line says "an entity, date, or number", as specified, but 45/500 2Wiki and 31/500 HotpotQA golds
  are yes/no. The scorer accepts "yes"/"no"; the instruction does not mention them.
- BABILong keeps its post_prompt ("Do not write anything else after that", qa5 also "Do not explain your answer")
  before our "Reason step by step" line; with thinking on, reasoning goes into `<think>` anyway.
- RULER: no answer prefix (cannot prefill in thinking mode), per-sample length fit to the Qwen3 chat-rendered
  prompt (no tokens reserved for generation), collision-free names; see `ruler_vt.py`.
- BABILong's "0k/4k/8k" are its own context budgets; with Qwen3 the full prompts measure up to 1.2k / 4.7k / 8.6k
  tokens. The tests check them against the buckets `(0, 2048]`, `(2048, 5120]`, `(5120, 9216]`; RULER prompts
  are in `(L - 64, L]`.
