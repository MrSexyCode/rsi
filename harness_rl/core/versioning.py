"""Content hashes that identify a harness package, an adapter package or an eval set.

version = sha256 over the sorted relative file paths and their bytes, so renaming, adding, removing or editing any
file changes it, and nothing else (mtimes, permissions, the directory's location) does. Caches are skipped.
"""

import hashlib
from pathlib import Path

_SKIP_DIRS = {"__pycache__", ".pytest_cache", "node_modules", ".git"}
_SKIP_SUFFIXES = {".pyc"}


def package_files(root: Path) -> list[Path]:
    root = Path(root)
    files = [p for p in root.rglob("*") if p.is_file()
             and not _SKIP_DIRS.intersection(p.relative_to(root).parts) and p.suffix not in _SKIP_SUFFIXES]
    return sorted(files, key=lambda p: p.relative_to(root).as_posix())


def content_hash(root: Path) -> str:
    """sha256 hex digest of a directory's files: for each file, its relative path, NUL, its length, NUL, its bytes."""
    h = hashlib.sha256()
    for p in package_files(root):
        data = p.read_bytes()
        h.update(p.relative_to(root).as_posix().encode())
        h.update(b"\0%d\0" % len(data))
        h.update(data)
    return h.hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
