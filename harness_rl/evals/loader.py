"""Read frozen eval sets. `final` items are only returned when explicitly allowed (the runner's --final)."""

from __future__ import annotations

import json
from pathlib import Path

from harness_rl.core.versioning import file_hash
from harness_rl.evals.common import SETS_DIR, SPLITS


class FinalSplitLocked(PermissionError):
    """Raised when the `final` split is requested without allow_final=True."""


def list_sets(root: Path = SETS_DIR) -> list[str]:
    return sorted(p.parent.name for p in Path(root).glob("*/manifest.json"))


def load_manifest(name: str, root: Path = SETS_DIR) -> dict:
    return json.loads((Path(root) / name / "manifest.json").read_text())


def set_hash(name: str, root: Path = SETS_DIR) -> str:
    """eval_set_hash: sha256 of the set's data.jsonl (recorded in the manifest as `sha256`)."""
    return load_manifest(name, root)["sha256"]


def load_set(name: str, splits=("dev",), allow_final: bool = False, root: Path = SETS_DIR,
             verify: bool = True) -> list[dict]:
    """Tasks of the set `name` whose split is in `splits`, in file order.

    Refuses `final` unless allow_final=True. With verify=True (default) the data file must match the manifest's
    sha256, so a run can never silently score an edited set.
    """
    if isinstance(splits, str):
        splits = (splits,)
    splits = tuple(splits)
    unknown = set(splits) - set(SPLITS)
    if unknown:
        raise ValueError(f"unknown splits {sorted(unknown)}; expected a subset of {SPLITS}")
    if "final" in splits and not allow_final:
        raise FinalSplitLocked(f"{name}: the 'final' split is reserved for the end-of-project report; "
                               "pass allow_final=True (runner: --final) to load it")
    path = Path(root) / name / "data.jsonl"
    if verify:
        expected = load_manifest(name, root)["sha256"]
        if file_hash(path) != expected:
            raise ValueError(f"{path}: sha256 does not match manifest ({expected})")
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return [r for r in rows if r["split"] in splits]
