"""Grammar-based generator of small Python programs: int variables, one list, bounded for loops, if/else, one print.

generate(spec) is pure (no execution), so signature(spec) and twins rebuild a program cheaply; the adapter executes
it in the sandbox. Shape of every program:

    <list> = [L digits]; two ints = digits          # 3 init statements
    <body>                                           # stmts - 3 statements, max for-nesting exactly `depth`
    print(<expr over modified variables>)

Statements are counted as AST statements (a for/if header counts 1, plus its body). Values stay small by
construction: on entering a top-level loop nest the generator fixes a 'mutable' subset of variables; inside the nest
only those are assigned, and accumulating right-hand sides read only literals, loop variables and variables the nest
never assigns, so growth is at most linear in the iteration count. List length changes only at top level outside
if-branches, so every index is statically in range. The adapter additionally rejects outputs outside [-999, 999].

Twins are applied to the parsed program: `perturb` = [i, v] sets the i-th int literal (source order) to v, and is
only legal on 'value' literals (not range bounds, indices, slice bounds or % / // divisors); `rename` = a seed that
maps every variable to a fresh name from the same pool.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from harness_rl.core.adapter import rng_for

SCALARS = ("a", "b", "c", "d", "e", "f", "g", "h", "k", "m", "n", "p", "q", "r", "s", "t", "u", "w", "x", "y", "z")
LISTS = ("xs", "ys", "zs", "nums", "vals", "arr", "row", "seq", "data", "items")
MAX_INTS = 5


@dataclass
class Program:
    code: str
    kinds: list[str]  # statement kinds in source order (init_*, new_int, assign, aug, mod, list_*, for_*, if*, print)
    literals: list[int]  # int literals in source order
    perturbable: list[int]  # indices into `literals` a perturbation twin may change


class _Gen:
    def __init__(self, spec: dict):
        self.rng = rng_for("progout", "gen", spec["stmts"], spec["depth"], spec["list_len"], spec["seed"])
        self.D = spec["depth"]
        self.pool = self.rng.sample(SCALARS, len(SCALARS))
        self.lst = self.rng.choice(LISTS)
        self.len = spec["list_len"]
        self.max_len = spec["list_len"] + 2
        self.ints: list[str] = []
        self.modified: list[str] = []
        self.deep: list[str] = []  # variables assigned inside loops nested `depth` deep
        self.kinds: list[str] = []

    # ---- helpers
    def fresh(self) -> str:
        return self.pool.pop()

    def lit(self, lo: int = 1, hi: int = 9) -> str:
        return str(self.rng.randint(lo, hi))

    def mark(self, target: str, lv: list) -> None:
        for seen in [self.modified] + [self.deep] * (self.D > 0 and len(lv) == self.D):
            if target not in seen:
                seen.append(target)

    def index(self, lv: list) -> str:
        """A list index that is statically in range: a literal, or a range-loop variable bounded by the length."""
        ok = [name for name, bound in lv if bound is not None and bound <= self.len]
        if ok and self.rng.random() < 0.5:
            return self.rng.choice(ok)
        return str(self.rng.randrange(self.len))

    def atom(self, lv: list, mut: set | None, lit: bool = True, avoid: str = "") -> str:
        """A value: literal, readable int, loop variable or list element. In a loop nest only non-mutable sources."""
        opts = [self.lit()] if lit else []
        opts += [v for v in self.ints if mut is None or v not in mut]
        opts += [name for name, _ in lv]
        if mut is None or self.lst not in mut:
            opts.append(f"{self.lst}[{self.index(lv)}]")
        opts = [o for o in opts if o != avoid]
        return self.rng.choice(opts) if opts else self.lit()

    def binary(self, lv: list, mut: set | None) -> str:
        """`x op y` with distinct operands, not both literals (no constant or self-cancelling expressions)."""
        x = self.atom(lv, mut)
        y = self.atom(lv, mut, lit=not x.isdigit(), avoid=x)
        return f"{x} {self.rng.choice('+-')} {y}"

    def cond(self, lv: list) -> str:
        a, b = self.rng.sample(self.ints, 2) if len(self.ints) >= 2 else (self.ints[0], self.lit())
        opts = [f"{a} > {b}", f"{a} < {self.lit()}", f"{a} % 2 == {self.rng.randint(0, 1)}",
                f"{self.lst}[{self.index(lv)}] > {a}"]
        opts += [f"{name} > {self.lit(0, 5)}" for name, _ in lv]
        opts += [f"{name} % 2 == 0" for name, b in lv if b is not None]
        return self.rng.choice(opts)

    # ---- statements
    def simple(self, lv: list, mut: set | None, in_if: bool) -> str:
        """One simple statement (cost 1)."""
        rng, xs = self.rng, self.lst
        ints = [v for v in self.ints if mut is None or v in mut]
        list_ok = mut is None or xs in mut
        opts = []
        if ints:
            opts += ["aug", "aug", "mod"] + (["assign", "assign"] if mut is None else ["assign"])
        if list_ok:
            opts += ["list_set", "list_aug"]
            if mut is None:
                opts += ["list_sort", "list_reverse"]
                if not in_if:
                    opts += ["list_extend"] * (self.len < self.max_len) + ["list_drop"] * (self.len > 2)
        if mut is None and not in_if and len(self.ints) < MAX_INTS:
            opts += ["new_int"]
        kind = rng.choice(opts)
        self.kinds.append(kind)
        a = lambda: self.atom(lv, mut)  # noqa: E731
        if kind in ("aug", "mod", "assign"):
            t = rng.choice(ints)
            self.mark(t, lv)
            if kind == "aug":
                op = rng.choice(["+=", "+=", "-="] + (["*="] if mut is None else []))
                return f"{t} {op} {self.lit(2, 3) if op == '*=' else a()}"
            if kind == "mod":
                return f"{t} = {t} % {self.lit(2, 9)}"
            if mut is None and rng.random() < 0.3:
                return f"{t} = {self.atom(lv, mut, lit=False)} * {self.lit(2, 3)}"
            return f"{t} = {self.binary(lv, mut)}"
        if kind == "new_int":
            t = self.fresh()
            expr = rng.choice([self.binary(lv, mut), f"sum({xs})", f"max({xs})", f"min({xs})",
                               f"{xs}[{self.index(lv)}] * {self.lit(2, 3)}"])
            self.ints.append(t)
            self.mark(t, lv)
            return f"{t} = {expr}"
        self.mark(xs, lv)
        if kind == "list_set":
            return f"{xs}[{self.index(lv)}] = {self.atom(lv, mut, lit=False)} {rng.choice('+-')} {self.lit()}"
        if kind == "list_aug":
            return f"{xs}[{self.index(lv)}] {rng.choice(['+=', '+=', '-='])} {a()}"
        if kind == "list_sort":
            return f"{xs} = sorted({xs})"
        if kind == "list_reverse":
            return f"{xs} = {xs}[::-1]"
        if kind == "list_extend":
            line = f"{xs} = {xs} + [{a()}]"
            self.len += 1
            return line
        self.len -= 1  # list_drop
        return f"{xs} = {xs}[1:]"

    def block(self, budget: int, need: int, depth: int, lv: list, mut: set | None, in_if: bool) -> list[str]:
        """`budget` statements; if need > 0, one for-chain of exactly `need` nested loops is placed among them."""
        out: list[str] = []
        free = budget - (need + 1 if need else 0)
        chain_at = self.rng.randint(0, free) if need else -1
        used = 0
        while free > 0 or need:
            if need and used >= chain_at:
                extra = self.rng.randint(0, min(free, 2))
                out += self.loop(need + extra, need - 1, depth, lv, mut, in_if)
                free -= extra
                need = 0
                continue
            opts = ["simple"] * 3
            if free >= 2:
                opts += ["if"] + ["for"] * (depth < self.D)
            kind = self.rng.choice(opts)
            if kind == "simple":
                out.append(self.simple(lv, mut, in_if))
                cost = 1
            else:
                cost = self.rng.randint(2, min(free, 4))
                if kind == "for":
                    out += self.loop(cost - 1, 0, depth, lv, mut, in_if)
                else:
                    out += self.if_stmt(cost - 1, depth, lv, mut)
            free -= cost
            used += cost
        return out

    def loop(self, body: int, need: int, depth: int, lv: list, mut: set | None, in_if: bool) -> list[str]:
        if mut is None:  # entering a top-level nest: fix what it may assign
            mut = set(self.rng.sample(self.ints, self.rng.randint(1, min(2, len(self.ints)))))
            if self.rng.random() < 0.5:
                mut.add(self.lst)
        name = self.fresh()
        if self.rng.random() < 0.5:
            bound = self.rng.randint(2, 4)
            header, inner_lv = f"for {name} in range({bound}):", lv + [(name, bound)]
            self.kinds.append("for_range")
        else:
            header, inner_lv = f"for {name} in {self.lst}:", lv + [(name, None)]
            self.kinds.append("for_list")
        return [header] + _indent(self.block(body, need, depth + 1, inner_lv, mut, in_if))

    def if_stmt(self, body: int, depth: int, lv: list, mut: set | None) -> list[str]:
        lines = [f"if {self.cond(lv)}:"]
        if body >= 2 and self.rng.random() < 0.6:
            self.kinds.append("if_else")
            b1 = self.rng.randint(1, body - 1)
            lines += _indent(self.block(b1, 0, depth, lv, mut, True))
            lines += ["else:"] + _indent(self.block(body - b1, 0, depth, lv, mut, True))
        else:
            self.kinds.append("if")
            lines += _indent(self.block(body, 0, depth, lv, mut, True))
        return lines

    def print_stmt(self) -> str:
        xs, rng = self.lst, self.rng
        # mostly print something the deepest loop computes, so the depth knob reaches the answer
        mods = self.deep if self.deep and rng.random() < 0.8 else self.modified
        ints = [v for v in mods if v != xs]
        opts = [f"print({v})" for v in ints]
        if ints:
            t = rng.choice(ints)
            opts.append(f"print({t} {rng.choice('+-')} {rng.choice([v for v in self.ints if v != t])})")
        if xs in mods:
            opts += [f"print({xs})", f"print(sum({xs}))", f"print(max({xs}) - min({xs}))"]
        if not opts:
            opts = [f"print(sum({xs}) + {self.ints[0]})"]
        self.kinds.append("print")
        return rng.choice(opts)

    def program(self, stmts: int) -> str:
        lines = [f"{self.lst} = [{', '.join(self.lit(0, 9) for _ in range(self.len))}]"]
        self.kinds.append("init_list")
        for _ in range(2):
            t = self.fresh()
            self.ints.append(t)
            lines.append(f"{t} = {self.lit()}")
            self.kinds.append("init_int")
        lines += self.block(stmts - 3, self.D, 0, [], None, False)
        lines.append(self.print_stmt())
        return "\n".join(lines) + "\n"


def _indent(lines: list[str]) -> list[str]:
    return ["    " + s for s in lines]


def min_stmts(depth: int) -> int:
    return 3 + depth + 1


def int_constants(tree: ast.AST) -> list[ast.Constant]:
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and type(n.value) is int]
    return sorted(nodes, key=lambda n: (n.lineno, n.col_offset))


def _fixed_constants(tree: ast.AST) -> set[int]:
    """ids of literals a perturbation must not touch: range bounds, indices, slice bounds, % and // divisors."""
    fixed: set[int] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "range":
            fixed |= {id(c) for a in n.args for c in ast.walk(a)}
        elif isinstance(n, ast.Subscript):
            fixed |= {id(c) for c in ast.walk(n.slice)}
        elif isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Mod, ast.FloorDiv)):
            fixed |= {id(c) for c in ast.walk(n.right)}
    return fixed


class _Rename(ast.NodeTransformer):
    def __init__(self, mapping: dict[str, str]):
        self.mapping = mapping

    def visit_Name(self, node: ast.Name) -> ast.Name:
        node.id = self.mapping.get(node.id, node.id)
        return node


def generate(spec: dict) -> Program:
    if spec["stmts"] < min_stmts(spec["depth"]):
        raise ValueError(f"stmts={spec['stmts']} < {min_stmts(spec['depth'])} needed for depth={spec['depth']}")
    g = _Gen(spec)
    tree = ast.parse(g.program(spec["stmts"]))
    if spec.get("perturb") is not None:
        i, v = spec["perturb"]
        node = int_constants(tree)[i]
        if id(node) in _fixed_constants(tree):
            raise ValueError(f"literal {i} is structural and cannot be perturbed")
        node.value = v
    if spec.get("rename"):
        used = list(dict.fromkeys(n.id for n in ast.walk(tree) if isinstance(n, ast.Name)
                                  and (n.id in SCALARS or n.id in LISTS)))
        rng = rng_for("progout", "rename", spec["rename"])
        free_s = rng.sample([s for s in SCALARS if s not in used], len([s for s in SCALARS if s not in used]))
        free_l = rng.sample([s for s in LISTS if s not in used], len([s for s in LISTS if s not in used]))
        tree = _Rename({u: (free_l if u in LISTS else free_s).pop() for u in used}).visit(tree)
    consts = int_constants(tree)
    fixed = _fixed_constants(tree)
    return Program(code=ast.unparse(tree), kinds=g.kinds, literals=[c.value for c in consts],
                   perturbable=[i for i, c in enumerate(consts) if id(c) not in fixed])
