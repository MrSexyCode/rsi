"""Fictional entity names from a syllable generator, one style per entity type (plan 1.2).

A name is one or two *core* words built from (onset) vowel (onset vowel)* (coda) syllables, plus a fixed type noun
for organizations, artifacts and events ("Varnel Guild", "Osk Codex"... ). Persons are "First Last", cities a single
word. Core words are rejected when they are on the blocklist (data/blocklist.txt: English words, the top first and
last names, country and city names; see data/README.md), and within one world every core word is unique, at least
4 letters, at edit distance >= 2 from every other core word, and never a substring (so never a prefix) of another.
Everything is drawn from the caller's random.Random, so names are deterministic by seed.
"""

from __future__ import annotations

import random
import re
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
BLOCKLIST_FILE = DATA_DIR / "blocklist.txt"

TYPES = ("Person", "City", "Organization", "Artifact", "Event")

ONSETS = ("b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z",
          "br", "dr", "fr", "gr", "kr", "tr", "vr", "bl", "fl", "kl", "pl", "sk", "st", "th", "sh")
VOWELS = ("a", "e", "i", "o", "u", "a", "e", "i", "o", "ai", "ei", "ou")  # plain vowels twice as likely
CODAS = ("n", "r", "l", "s", "k", "m", "d", "x", "th", "nd", "rn", "sk")
MIN_LEN = 4

# Type nouns appended to core words. They are ordinary English words on purpose (they mark the type).
TYPE_NOUNS = {
    "Organization": ("Group", "Labs", "Guild", "Institute", "Holdings", "Works", "Foundation", "Company", "Society",
                     "Collective"),
    "Artifact": ("Codex", "Lantern", "Astrolabe", "Tapestry", "Chalice", "Compass", "Mosaic", "Clock", "Atlas",
                 "Harp"),
    "Event": ("Festival", "Summit", "Fair", "Regatta", "Expo", "Gala", "Congress", "Games", "Carnival", "Symposium"),
}

# (syllables lo, hi, coda probability) per core word slot
STYLE = {
    "Person": ((2, 2, 0.5), (2, 3, 0.6)),
    "City": ((2, 3, 0.6),),
    "Organization": ((2, 2, 0.5),),
    "Artifact": ((2, 2, 0.5),),
    "Event": ((2, 2, 0.5),),
}


def _alt(xs) -> str:
    return "(?:" + "|".join(sorted(set(xs), key=lambda s: (-len(s), s))) + ")"


# Every core word the generator can produce matches this (lower-case); used to shrink the stored blocklist.
GRAMMAR = re.compile(f"^{_alt(ONSETS)}?{_alt(VOWELS)}(?:{_alt(ONSETS)}{_alt(VOWELS)})*{_alt(CODAS)}?$")


@lru_cache(maxsize=1)
def blocklist() -> frozenset[str]:
    return frozenset(BLOCKLIST_FILE.read_text().split())


def core_word(rng: random.Random, lo: int, hi: int, coda_p: float) -> str:
    n = rng.randint(lo, hi)
    w = (rng.choice(ONSETS) if rng.random() < 0.8 else "") + rng.choice(VOWELS)
    for _ in range(n - 1):
        w += rng.choice(ONSETS) + rng.choice(VOWELS)
    if rng.random() < coda_p:
        w += rng.choice(CODAS)
    return w


def close(a: str, b: str) -> bool:
    """True if two lower-case words collide: one contains the other, or edit distance < 2."""
    if a in b or b in a:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        return sum(x != y for x, y in zip(a, b)) <= 1
    s, t = (a, b) if la < lb else (b, a)
    return any(t[:i] + t[i + 1:] == s for i in range(len(t)))


class NameGen:
    """Draws unique, mutually distant names for one world. Collision checks are set lookups, not pairwise scans."""

    def __init__(self, rng: random.Random, max_tries: int = 2000):
        self.rng = rng
        self.max_tries = max_tries
        self.block = blocklist()
        self.cores: set[str] = set()
        self._joined = "|"  # "|core|core|...", for "is t a substring of an existing core"
        self._del_pos: set[tuple[int, str]] = set()  # (i, w without char i): same key <=> one substitution apart
        self._del: set[str] = set()  # w without one char: t in it <=> t is w minus one char
        self.names: dict[str, str] = {}  # name -> type

    def _collides(self, t: str) -> bool:
        if t in self._joined:  # t is (a substring of) an existing core
            return True
        n = len(t)
        if any(t[i:j] in self.cores for i in range(n) for j in range(i + MIN_LEN, n + 1)):  # contains a core
            return True
        dels = [t[:i] + t[i + 1:] for i in range(n)]
        if any((i, d) in self._del_pos for i, d in enumerate(dels)):  # one substitution
            return True
        return any(d in self.cores for d in dels) or t in self._del  # one insertion / deletion

    def _ok(self, t: str) -> bool:
        return len(t) >= MIN_LEN and bool(GRAMMAR.match(t)) and t not in self.block and not self._collides(t)

    def _commit(self, t: str) -> None:
        self.cores.add(t)
        self._joined += t + "|"
        for i in range(len(t)):
            d = t[:i] + t[i + 1:]
            self._del_pos.add((i, d))
            self._del.add(d)

    def new(self, etype: str) -> str:
        for _ in range(self.max_tries):
            words = [core_word(self.rng, *slot) for slot in STYLE[etype]]
            if not all(self._ok(w) for w in words) or (len(words) == 2 and close(*words)):
                continue
            for w in words:
                self._commit(w)
            name = " ".join(w.capitalize() for w in words)
            if etype in TYPE_NOUNS:
                name += " " + self.rng.choice(TYPE_NOUNS[etype])
            self.names[name] = etype
            return name
        raise RuntimeError(f"could not draw a fresh {etype} name in {self.max_tries} tries")


def core_words(name: str) -> list[str]:
    """The generated (non type-noun) words of a name, lower-case."""
    words = name.split()
    if words[-1] in {n for ns in TYPE_NOUNS.values() for n in ns}:
        words = words[:-1]
    return [w.lower() for w in words]
