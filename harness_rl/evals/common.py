"""Shared helpers for building frozen eval sets: canonical JSON, hash-based deterministic sampling and splits,
and prompt token counting.

Determinism: nothing here uses a stateful RNG. Sampling and dev/final assignment rank items by
sha256("<seed>|<salt>|<set>|<id>"), so the result depends only on the seed, the item ids and the stratum sizes,
never on Python/NumPy RNG implementations or iteration order.
"""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, Sequence

from harness_rl.core.env_paths import MODEL_ID

EVALS_DIR = Path(__file__).resolve().parent
SETS_DIR = EVALS_DIR / "sets"
REPO_ROOT = EVALS_DIR.parents[1]
RAW_DIR = Path("/home/ubuntu/data/harness_rl_raw")  # downloaded sources, one directory per source (untrusted data)

# Tokenizer used for length targeting and the token statistics in manifests (pinned HF snapshot).
TOKENIZER_ID = MODEL_ID
TOKENIZER_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"

FINAL_FRACTION = Fraction(1, 5)  # 80% dev / 20% final for every natural set
SPLITS = ("dev", "final")
SCHEMA_VERSION = 1


def dumps_line(obj) -> str:
    """Canonical one-line JSON: sorted keys, fixed separators, UTF-8 kept as is."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def dumps_manifest(obj) -> str:
    return json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def rank_key(seed: int, salt: str, set_name: str, item_id: str) -> str:
    return sha256_bytes(f"{seed}|{salt}|{set_name}|{item_id}".encode())


def derive_int(seed: int, *parts) -> int:
    """A 64-bit integer seed derived from (seed, parts), for per-item random.Random instances."""
    return int(sha256_bytes("|".join(map(str, (seed, *parts))).encode())[:16], 16)


def allocate(counts: dict[str, int], total: int) -> dict[str, int]:
    """Largest-remainder apportionment of `total` over strata proportional to `counts`.

    Sums to `total` exactly; ties in the remainder go to the lexicographically smaller stratum key.
    """
    n = sum(counts.values())
    assert 0 <= total <= n, (total, n)
    quotas = {k: Fraction(c * total, n) for k, c in counts.items()}
    alloc = {k: int(q) for k, q in quotas.items()}
    rest = total - sum(alloc.values())
    order = sorted(quotas, key=lambda k: (-(quotas[k] - alloc[k]), k))
    for k in order[:rest]:
        alloc[k] += 1
    assert all(alloc[k] <= counts[k] for k in counts)
    return alloc


def round_half_up(x: Fraction) -> int:
    return int(x + Fraction(1, 2))


def group(items: Iterable, key: Callable) -> dict[str, list]:
    out: dict[str, list] = {}
    for it in items:
        out.setdefault(str(key(it)), []).append(it)
    return out


def stratified_sample(items: Sequence, n: int, *, stratum: Callable, item_id: Callable, seed: int,
                      set_name: str) -> tuple[list, dict[str, int]]:
    """Pick n items, allocated over strata by largest remainder, lowest rank_key first within each stratum.

    Returns the picked items in their original order and the per-stratum counts.
    """
    if n >= len(items):
        return list(items), {k: len(v) for k, v in sorted(group(items, stratum).items())}
    groups = group(items, stratum)
    alloc = allocate({k: len(v) for k, v in groups.items()}, n)
    picked = set()
    for k, members in groups.items():
        ranked = sorted(members, key=lambda it: rank_key(seed, "sample", set_name, item_id(it)))
        picked.update(item_id(it) for it in ranked[:alloc[k]])
    return [it for it in items if item_id(it) in picked], dict(sorted(alloc.items()))


def assign_splits(items: Sequence, *, stratum: Callable, item_id: Callable, seed: int, set_name: str,
                  final_fraction: Fraction = FINAL_FRACTION) -> tuple[dict[str, str], dict]:
    """Deterministic stratified dev/final assignment.

    The final count is round_half_up(N * final_fraction) exactly; it is apportioned over strata by largest
    remainder, and within a stratum the items with the lowest rank_key(seed, "split", ...) go to final.
    """
    groups = group(items, stratum)
    n_final = round_half_up(len(items) * final_fraction)
    alloc = allocate({k: len(v) for k, v in groups.items()}, n_final)
    split = {}
    for k, members in groups.items():
        ranked = sorted(members, key=lambda it: rank_key(seed, "split", set_name, item_id(it)))
        for i, it in enumerate(ranked):
            split[item_id(it)] = "final" if i < alloc[k] else "dev"
    per_stratum = {k: {"dev": len(groups[k]) - alloc[k], "final": alloc[k]} for k in sorted(groups)}
    return split, per_stratum


@lru_cache(maxsize=1)
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER_ID, revision=TOKENIZER_REVISION, trust_remote_code=False)


def chat_text(prompt: str) -> str:
    """The prompt as the model sees it: one user message, Qwen3 chat template, thinking on (see docs/inventory.md)."""
    return tokenizer().apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=True)


def prompt_tokens(prompt: str) -> int:
    """Number of prompt tokens the engine receives for this prompt (chat template included)."""
    return len(tokenizer().encode(chat_text(prompt), add_special_tokens=False))


def text_tokens(text: str) -> int:
    return len(tokenizer().encode(text, add_special_tokens=False))


def rel(path: Path) -> str:
    """Repo-relative path for files inside the repo, absolute otherwise."""
    path = Path(path).resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)
