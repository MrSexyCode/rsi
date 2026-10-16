"""Deterministic RULER variable-tracking (VT) generator, length-targeted with the Qwen3 tokenizer.

Mirrors NVIDIA/RULER@c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a scripts/data/synthetic/variable_tracking.py with
the `vt` config of scripts/synthetic.yaml (type_haystack=noise, num_chains=1, num_hops=4) and the 'base' model
template:

- Chain: "VAR {X0} = {5-digit value}", then "VAR {Xj+1} = VAR {Xj} " (trailing space kept, as in RULER).
  Names are 5 random upper-case letters (3 for the ICL example); the value is uniform in [10000, 99998]
  (np.random.randint(10000, 99999)).
- Haystack: num_noises copies of the noise sentence; each chain's lines are inserted in order at sorted random
  positions (RULER's insert(pos + j) loop); lines joined with "\\n", then ". \\n" -> ".\\n".
- One in-context example, generated once per set exactly like RULER's (3-letter names, value 12345, longest
  haystack with <= 500 tokens), then randomized per sample (RULER's randomize_icl: the 5 answer names get new
  random names of the same length, 12345 gets a new 5-digit value).
- Answer: all num_hops + 1 names of the chain (the query asks for every variable assigned the value).

Deliberate differences (all documented in README.md):
- RNG: Python's random.Random with a per-sample seed derived from (seed, set, index) instead of the global
  random/np.random streams, so each sample is a pure function of the seed (RULER's exact byte stream is not
  reproducible anyway because its lengths depend on the tokenizer).
- Length: RULER binary-searches one haystack size per dataset and shrinks it per sample until
  tokens + tokens_to_generate <= max_seq_length. Here each sample takes the largest haystack whose full prompt,
  rendered with the Qwen3 chat template (thinking on), is <= the target (4096 / 8192); no tokens are reserved
  for generation because the response length is set by the runner's thinking budget.
- Names are drawn without duplicates (RULER's list can, rarely, repeat a name within a chain), never equal "VAR",
  and the randomized ICL names never collide with the names they replace, so str.replace cannot corrupt the
  example. The ICL value is redrawn if it equals the query value.
- The query's answer prefix is not part of the prompt (see prompts.RULER_VT_TEMPLATE).
"""

from __future__ import annotations

import random
import string

from harness_rl.evals import common
from harness_rl.evals.prompts import RULER_VT_ANSWER_PREFIX, RULER_VT_TASK_TEMPLATE, _fill, get_adapter

NOISE = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
NUM_CHAINS = 1
NUM_HOPS = 4
ICL_MAX_TOKENS = 500
ICL_VALUE = "12345"
CONFIG = {"type_haystack": "noise", "num_chains": NUM_CHAINS, "num_hops": NUM_HOPS, "icl_max_tokens": ICL_MAX_TOKENS,
          "noise": NOISE}


def _name(rng: random.Random, k: int, taken: set[str]) -> str:
    while True:
        v = "".join(rng.choices(string.ascii_uppercase, k=k))
        if v not in taken and v != "VAR":
            taken.add(v)
            return v


def generate_chains(rng: random.Random, num_chains: int = NUM_CHAINS, num_hops: int = NUM_HOPS, is_icl: bool = False):
    k = 3 if is_icl else 5
    hops = min(10, num_hops) if is_icl else num_hops
    taken: set[str] = set()
    names = [_name(rng, k, taken) for _ in range((hops + 1) * num_chains)]
    vars_ret, chains = [], []
    for i in range(0, len(names), hops + 1):
        this = names[i:i + hops + 1]
        value = ICL_VALUE if is_icl else str(rng.randint(10000, 99998))
        chain = [f"VAR {this[0]} = {value}"] + [f"VAR {this[j + 1]} = VAR {this[j]} " for j in range(hops)]
        vars_ret.append(this)
        chains.append(chain)
    return vars_ret, chains


def build_context(rng: random.Random, chains: list[list[str]], num_noises: int) -> str:
    sentences = [NOISE] * num_noises
    for chain in chains:
        positions = sorted(rng.sample(range(len(sentences)), len(chain)))
        for insert_pi, j in zip(positions, range(len(chain))):
            sentences.insert(insert_pi + j, chain[j])
    return "\n".join(sentences).replace(". \n", ".\n")


def _largest_fitting(count, budget: int, start: int, lo: int = 5) -> int:
    """Largest n >= lo with count(n) <= budget, by local search from an estimate (count is ~linear in n)."""
    n = max(lo, start)
    if count(n) <= budget:
        while count(n + 1) <= budget:
            n += 1
    else:
        while n > lo and count(n) > budget:
            n -= 1
        if count(n) > budget:
            raise ValueError("budget too small for the minimum haystack")
    return n


def _estimate(count, budget: int, a: int = 20, b: int = 60) -> int:
    ta, tb = count(a), count(b)
    per = (tb - ta) / (b - a)
    return a + int((budget - ta) / per)


def make_icl_example(seed: int, set_name: str) -> tuple[str, list[str]]:
    """RULER's ICL example: its task template + answer prefix for a 3-letter chain with value 12345, haystack as
    long as fits in 500 tokens, followed by " " + the answer names."""
    rng = random.Random(common.derive_int(seed, set_name, "icl"))
    vars_, chains = generate_chains(rng, is_icl=True)

    def text(n: int) -> str:
        ctx = build_context(random.Random(common.derive_int(seed, set_name, "icl-pos", n)), chains, n)
        return (_fill(RULER_VT_TASK_TEMPLATE, context=ctx, query=ICL_VALUE)
                + _fill(RULER_VT_ANSWER_PREFIX, num_v=str(len(vars_[0])), query=ICL_VALUE))

    def count(n: int) -> int:
        return common.text_tokens(text(n))

    n = _largest_fitting(count, ICL_MAX_TOKENS, _estimate(count, ICL_MAX_TOKENS, 5, 15))
    return text(n) + " " + " ".join(vars_[0]), vars_[0]


def randomize_icl(icl_example: str, rng: random.Random, avoid_value: str) -> str:
    """RULER's randomize_icl, made collision-free."""
    targets = icl_example.strip().split()[-NUM_HOPS - 1:]
    taken = set(targets)
    for item in targets:
        icl_example = icl_example.replace(item, _name(rng, len(item), taken))
    while (new_value := str(rng.randint(10000, 99998))) == avoid_value:
        pass
    return icl_example.replace(ICL_VALUE, new_value)


def generate(seed: int, set_name: str, target_tokens: int, n: int) -> list[dict]:
    """n VT samples whose chat-rendered prompts are as long as possible but <= target_tokens."""
    adapter = get_adapter("ruler_vt")
    icl_base, _ = make_icl_example(seed, set_name)
    out = []
    for i in range(n):
        rng = random.Random(common.derive_int(seed, set_name, "sample", i))
        vars_, chains = generate_chains(rng)
        value = chains[0][0].split("=")[-1].strip()
        icl = randomize_icl(icl_base, rng, value)

        def render(k: int) -> str:
            ctx = build_context(random.Random(common.derive_int(seed, set_name, "pos", i, k)), chains, k)
            return adapter.render(icl_example=icl, context=ctx, query=value)

        cache: dict[int, int] = {}

        def count(k: int) -> int:
            if k not in cache:
                cache[k] = common.prompt_tokens(render(k))
            return cache[k]

        k = _largest_fitting(count, target_tokens, _estimate(count, target_tokens, 50, 150))
        out.append({"index": i, "prompt": render(k), "vars": vars_[0], "value": value, "num_noises": k,
                    "prompt_tokens": count(k)})
    return out
