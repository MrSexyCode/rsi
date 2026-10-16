"""Build data/blocklist.txt from the raw name and word lists (read as data only; nothing in them is executed).

    python -m harness_rl.adapters.multihop.build_blocklist [--check]

Sources (URLs and sha256 in data/README.md; the sha256 of every file is checked before use):
  english words      dwyl/english-words words_alpha.txt (all 370k words)
  first names        top 10,000 of hadley/data-baby-names (summed yearly share, 1880-2008) + all Census 1990 first names
  last names         top 10,000 of the US Census 2010 surname table
  countries, cities  GeoNames countryInfo.txt (country + capital) and cities15000 (name, asciiname, alternate names)
Multi-word entries contribute every word. Only lower-case words the syllable grammar (names.GRAMMAR) can produce
and of length >= names.MIN_LEN are stored, which keeps the file small without changing what is blocked.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from collections import defaultdict
from pathlib import Path

from harness_rl.adapters.multihop.names import BLOCKLIST_FILE, GRAMMAR, MIN_LEN

RAW = Path("/home/ubuntu/data/harness_rl_raw")
SOURCES = {
    "english_words/words_alpha.txt": "3ed0c94610d8bcf7c11bbb49c56aa49c7234d32b66824df91f554169e572da48",
    "baby_names_hadley/baby-names.csv": "1259523fa76e5c18151a4c7612b854b22605d07127c596126940e2941ba15d3c",
    "census1990_firstnames/dist.female.first": "bd2f310fc4e5d5e5ea122c9d4342c9821145823118eb20db1647f305ec77b358",
    "census1990_firstnames/dist.male.first": "0a5078ef6effe3b483d15b0f7f95047662126c9bfb624ecd5e5b978fc0f2470b",
    "census_surnames/names.zip": "117c41cb4668727b7627b2845b6df3f83eb2a22a1813f42c0ff4bdcab86de135",
    "geonames/countryInfo.txt": "c55f638cfaafb4fe45a9aa15120a0f1f78e8a34a8e1847d7b9320cf740fbc9ab",
    "geonames/cities15000.zip": "da8eca618fb5a0aff0150b2c2caa89b9bf6fb279e6720ac4c287823049cf2069",
}
TOP_N = 10_000
_WORD = re.compile(r"[A-Za-z]+")


def _check_sources() -> None:
    for rel, sha in SOURCES.items():
        got = hashlib.sha256((RAW / rel).read_bytes()).hexdigest()
        if got != sha:
            raise ValueError(f"{RAW / rel}: sha256 {got} != recorded {sha}")


def _words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def _zip_member(rel: str, member: str) -> str:
    import zipfile

    with zipfile.ZipFile(RAW / rel) as z:
        return z.read(member).decode("utf-8")


def raw_lists() -> dict[str, set[str]]:
    _check_sources()
    english = set((RAW / "english_words/words_alpha.txt").read_text().split())

    share: dict[str, float] = defaultdict(float)
    with open(RAW / "baby_names_hadley/baby-names.csv", newline="") as f:
        for row in csv.DictReader(f):
            share[row["name"].lower()] += float(row["percent"])
    first = set(sorted(share, key=lambda n: (-share[n], n))[:TOP_N])
    for g in ("female", "male"):
        first |= {line.split()[0].lower() for line in (RAW / f"census1990_firstnames/dist.{g}.first").read_text()
                  .splitlines() if line.strip()}

    rows = list(csv.DictReader(_zip_member("census_surnames/names.zip", "Names_2010Census.csv").splitlines()))
    ranked = sorted((r for r in rows if r["rank"].isdigit()), key=lambda r: int(r["rank"]))
    last = {r["name"].lower() for r in ranked[:TOP_N]}

    places: set[str] = set()
    for line in (RAW / "geonames/countryInfo.txt").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            cols = line.split("\t")
            places.update(_words(cols[4]) + _words(cols[5]))
    for line in _zip_member("geonames/cities15000.zip", "cities15000.txt").splitlines():
        cols = line.split("\t")
        places.update(_words(cols[1]) + _words(cols[2]))
        places.update(w for alt in cols[3].split(",") if alt.isascii() for w in _words(alt))
    return {"english": english, "first": first, "last": last, "places": places}


def blocklist_text(lists: dict[str, set[str]]) -> str:
    words = set().union(*lists.values())
    keep = sorted(w for w in words if len(w) >= MIN_LEN and GRAMMAR.match(w))
    return "\n".join(keep) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="compare with the stored file instead of writing it")
    args = ap.parse_args(argv)
    lists = raw_lists()
    text = blocklist_text(lists)
    stats = {k: len(v) for k, v in lists.items()} | {"stored": text.count("\n")}
    print(stats)
    if args.check:
        same = BLOCKLIST_FILE.read_text() == text
        print("blocklist up to date" if same else "blocklist STALE")
        return 0 if same else 1
    BLOCKLIST_FILE.write_text(text)
    print(f"wrote {BLOCKLIST_FILE} ({len(text)} bytes, sha256 {hashlib.sha256(text.encode()).hexdigest()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
