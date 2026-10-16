"""Sandbox for program-output tasks: whitelist AST validation, then a resource-limited subprocess.

Two layers, both always applied:
  - validate() admits only the generator's tiny language: int literals, one-level names, lists, arithmetic,
    comparisons, bounded `for` over range(<literal>) or a variable, if/else, a few pure builtins, and exactly one
    print() as the last statement. Everything else (imports, attributes, strings, defs, lambdas, while, with, try,
    comprehensions, keywords, starred args, ** ...) is not on the list, so it is rejected without being named.
  - run() executes only validated code, in `python -I -S` with an empty env, a fresh temp cwd and a wall timeout.
    A bootstrap sets CPU / address-space / file-size / process / fd rlimits inside the child before compiling the
    program (instead of preexec_fn, which can deadlock when the caller is multi-threaded, e.g. a reward pool).

Results are cached by (source, limits): execution is deterministic for this language, so a task's answer is
computed once per process and verify() never executes anything.
"""

from __future__ import annotations

import ast
import builtins
import keyword
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from functools import lru_cache

ALLOWED_BUILTINS = frozenset({"print", "range", "len", "sum", "min", "max", "abs", "sorted"})
_ONE_ARG = frozenset({"len", "sum", "abs", "sorted"})

ALLOWED_NODES = frozenset({
    ast.Module, ast.Assign, ast.AugAssign, ast.For, ast.If, ast.Expr,
    ast.Name, ast.Load, ast.Store, ast.Constant, ast.List, ast.Subscript, ast.Slice, ast.Call,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
    ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod, ast.USub, ast.Not, ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
})

MAX_SOURCE_CHARS = 4000
MAX_NODES = 1500
MAX_CONST = 1000
MAX_RANGE = 20
MAX_LOOP_DEPTH = 3

TIMEOUT_S = 5.0
CPU_S = 2
MEM_BYTES = 256 * 2**20
MAX_OUTPUT_CHARS = 4096

_NAME_RE = re.compile(r"[a-z][a-z0-9_]{0,15}\Z")
_BUILTIN_NAMES = frozenset(dir(builtins))


def validate(code: str) -> list[str]:
    """Reasons the program is outside the allowed language; empty means it may be executed."""
    if len(code) > MAX_SOURCE_CHARS:
        return [f"source longer than {MAX_SOURCE_CHARS} chars"]
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"syntax error: {e.msg}"]
    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_NODES:
        return [f"more than {MAX_NODES} AST nodes"]
    errs: list[str] = []
    call_funcs = {id(n.func) for n in nodes if isinstance(n, ast.Call)}
    for n in nodes:
        if type(n) not in ALLOWED_NODES:
            errs.append(f"node {type(n).__name__} not allowed")
        elif isinstance(n, ast.Constant):
            if type(n.value) is not int or abs(n.value) > MAX_CONST:
                errs.append(f"constant {n.value!r} not allowed (ints up to {MAX_CONST} only)")
        elif isinstance(n, ast.Name):
            if not _NAME_RE.match(n.id) or keyword.iskeyword(n.id):
                errs.append(f"name {n.id!r} not allowed")
            elif n.id in _BUILTIN_NAMES and (n.id not in ALLOWED_BUILTINS or id(n) not in call_funcs):
                errs.append(f"builtin {n.id!r} only allowed as a direct call to one of {sorted(ALLOWED_BUILTINS)}")
        elif isinstance(n, ast.Call):
            errs += _check_call(n)
        elif isinstance(n, (ast.Assign, ast.AugAssign)):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            if len(targets) != 1 or not isinstance(targets[0], (ast.Name, ast.Subscript)):
                errs.append("assignment target must be one name or one subscript")
        elif isinstance(n, ast.For):
            if not isinstance(n.target, ast.Name) or n.orelse:
                errs.append("for: target must be a name, no else")
            it = n.iter
            if not (isinstance(it, ast.Name) or (isinstance(it, ast.Call) and isinstance(it.func, ast.Name)
                                                 and it.func.id == "range")):
                errs.append("for: iterate over range(...) or a variable only")
    body = tree.body
    prints = [n for n in nodes if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "print"]
    last_is_print = bool(body) and isinstance(body[-1], ast.Expr) and any(body[-1].value is p for p in prints)
    if len(prints) != 1 or not last_is_print:
        errs.append("exactly one print(...) call, as the last top-level statement")
    if any(isinstance(n, ast.Expr) and n is not body[-1] for n in nodes):
        errs.append("bare expression statements not allowed")
    if (d := _loop_depth(body)) > MAX_LOOP_DEPTH:
        errs.append(f"loop depth {d} > {MAX_LOOP_DEPTH}")
    return errs


def _check_call(n: ast.Call) -> list[str]:
    if not isinstance(n.func, ast.Name) or n.func.id not in ALLOWED_BUILTINS:
        return ["call target must be an allowed builtin"]
    if n.keywords or any(isinstance(a, ast.Starred) for a in n.args):
        return [f"{n.func.id}: keyword or starred arguments not allowed"]
    f, k = n.func.id, len(n.args)
    if f == "range":
        if not 1 <= k <= 3 or not all(isinstance(a, ast.Constant) and type(a.value) is int for a in n.args):
            return ["range: 1-3 int literal arguments only"]
        if len(range(*(a.value for a in n.args))) > MAX_RANGE:
            return [f"range longer than {MAX_RANGE}"]
        return []
    if (f in _ONE_ARG or f == "print") and k != 1 or f in ("min", "max") and k not in (1, 2):
        return [f"{f}: wrong number of arguments"]
    return []


def _loop_depth(stmts: list[ast.stmt]) -> int:
    depth = 0
    for s in stmts:
        if isinstance(s, ast.For):
            depth = max(depth, 1 + _loop_depth(s.body))
        elif isinstance(s, ast.If):
            depth = max(depth, _loop_depth(s.body), _loop_depth(s.orelse))
    return depth


@dataclass(frozen=True)
class ExecResult:
    status: str  # ok | error | timeout | rejected
    stdout: str = ""
    stderr: str = ""
    returncode: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


_BOOT = """\
import resource as r, sys
for k, v in (("CPU", {cpu}), ("AS", {mem}), ("FSIZE", 0), ("NPROC", 0), ("NOFILE", 16), ("CORE", 0)):
    r.setrlimit(getattr(r, "RLIMIT_" + k), (v, v))
src = sys.stdin.read()
del r, sys, k, v
exec(compile(src, "<program>", "exec"), {{"__name__": "__main__"}})
"""


def execute(code: str, timeout: float = TIMEOUT_S, cpu_s: int = CPU_S, mem_bytes: int = MEM_BYTES) -> ExecResult:
    """Validate, then run in a fresh limited subprocess (uncached)."""
    if errs := validate(code):
        return ExecResult("rejected", stderr="; ".join(errs))
    with tempfile.TemporaryDirectory(prefix="progout-") as cwd:
        try:
            p = subprocess.run([sys.executable, "-I", "-S", "-c", _BOOT.format(cpu=cpu_s, mem=mem_bytes)],
                               input=code, capture_output=True, text=True, encoding="utf-8", errors="replace",
                               env={}, cwd=cwd, timeout=timeout, start_new_session=True)
        except subprocess.TimeoutExpired:
            return ExecResult("timeout")
    out, err = p.stdout, p.stderr[-MAX_OUTPUT_CHARS:]
    if len(out) > MAX_OUTPUT_CHARS:
        return ExecResult("error", stderr=f"output longer than {MAX_OUTPUT_CHARS} chars", returncode=p.returncode)
    return ExecResult("ok" if p.returncode == 0 else "error", out, err, p.returncode)


@lru_cache(maxsize=1 << 16)
def run(code: str) -> ExecResult:
    """execute() with the default limits, cached by source."""
    return execute(code)
