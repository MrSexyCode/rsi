# multihop adapter data

## `blocklist.txt`

Lower-case words a generated name core may not equal (`names.py`). Built by
`python -m harness_rl.adapters.multihop.build_blocklist` (`--check` compares with this file), which verifies each
source's sha256 before reading it. Sources are read as data only; nothing from them is executed.

| List | Local path (`/home/ubuntu/data/harness_rl_raw/...`) | URL | sha256 | Used |
|---|---|---|---|---|
| English words | `english_words/words_alpha.txt` | https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt | `3ed0c94610d8bcf7c11bbb49c56aa49c7234d32b66824df91f554169e572da48` | all 370,105 |
| First names | `baby_names_hadley/baby-names.csv` | https://raw.githubusercontent.com/hadley/data-baby-names/master/baby-names.csv | `1259523fa76e5c18151a4c7612b854b22605d07127c596126940e2941ba15d3c` | top 10,000 by summed yearly share (only 6,782 distinct exist) |
| First names | `census1990_firstnames/dist.{female,male}.first` | https://www2.census.gov/topics/genealogy/1990surnames/dist.female.first, `.../dist.male.first` | `bd2f310fc4e5d5e5ea122c9d4342c9821145823118eb20db1647f305ec77b358`, `0a5078ef6effe3b483d15b0f7f95047662126c9bfb624ecd5e5b978fc0f2470b` | all (5,163 distinct) |
| Last names | `census_surnames/names.zip` (`Names_2010Census.csv`) | https://www2.census.gov/topics/genealogy/2010surnames/names.zip | `117c41cb4668727b7627b2845b6df3f83eb2a22a1813f42c0ff4bdcab86de135` | top 10,000 by rank |
| Countries | `geonames/countryInfo.txt` | https://download.geonames.org/export/dump/countryInfo.txt | `c55f638cfaafb4fe45a9aa15120a0f1f78e8a34a8e1847d7b9320cf740fbc9ab` | country and capital names |
| Cities | `geonames/cities15000.zip` (`cities15000.txt`) | https://download.geonames.org/export/dump/cities15000.zip | `da8eca618fb5a0aff0150b2c2caa89b9bf6fb279e6720ac4c287823049cf2069` | name, ASCII name, ASCII alternate names |

Downloaded 2026-10-16 (GeoNames dumps change daily; the hashes pin the copies used). The SSA baby-name archive
(https://www.ssa.gov/oact/babynames/names.zip) refused scripted downloads, so the first-name list is the union of
the two sources above: 8,077 names. Multi-word entries contribute every word. Union: 522,021 words; the file keeps only
the 39,320 the syllable grammar (`names.GRAMMAR`) can produce with at least 4 letters, which blocks exactly the same
names in 313 KB.

## Built artifacts

- Held-out eval grid: `harness_rl/evals/sets/multihop_grid/` (`python -m harness_rl.adapters.multihop.build_sets --eval`).
- Training pool: `/home/ubuntu/data/harness_rl/multihop_pool/{train,val}.parquet` + `manifest.json`
  (`python -m harness_rl.adapters.multihop.build_sets --pool --train 20000 --val 500 --seed 0`).
- Probe table: `python -m harness_rl.adapters.multihop.probe_table --n 4000 --workers 135 --csv ...`.

Any change in this package changes the adapter version (its content hash); the tests then flag the eval grid and
the pool as stale until they are rebuilt.

## Changes

### v2 (`dsl.ADAPTER_RELEASE`)

After the 1.5 calibration of v1 (base Qwen3-8B: every cell 0.61-1.00, many abstentions on answerable tasks):

- Template audit. Every relation and attribute has a family of key words (`schema.Relation.keys`); each of its
  fact templates and question phrases contains one of its own keys and no other relation's key (tested). Removed
  the loose paraphrases the calibration flagged (per-template residual error vs. the cell mean on answerable
  tasks): `grew_up_in` ("the hometown of", "spent their childhood in"; replaced by `lives_in`),
  `studied_at` "was trained at" / "completed their studies at", `led_by` "is run by" / "is the head of",
  `treasure` "The pride of ... is the artifact", `largest_employer` (its key "employer" collided with works_for;
  replaced by `official_partner`), the "{v} people live in" / "is home to ... residents" population and
  "Work on ... ended" completion phrasings, and `founded_year` (now `established_year`, so "founded" only means
  founded_by). The worst v1 question phrases were the intersection verb phrases with the object in the middle
  ("treasures above all its other artifacts {y}", "has as its signature festival {y}"): intersections now use
  relative clauses ending in the object ("whose flagship product is ...", "that is headquartered in ...").
- Question wording: a chain question whose outer noun phrase already names the answer type is asked as
  "What is the name of the artifact that ..." instead of "Which artifact is the artifact that ...".
- Knobs: hops {2, 3, 4, 5, 6, 8, 10, 12}; a `very_high` distractor level (3x the near-miss insertions of high,
  all four types, plus more decoys: 9 decoy chains for chain, 4 per side for comparison, a 4x4 intersection grid).
  Full grid 8 x 3 x 4 x 3 = 288 cells. Eval grid: hops {2, 4, 6, 8, 12} x 3 ops x 4 levels, shuffled, 30 per
  cell = 1,800.
