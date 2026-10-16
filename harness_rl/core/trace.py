"""Rollout trace schema: one RolloutTrace per rollout, holding one CallRecord per model call.

A CallRecord is what the model proxy captured for one chat-completion request: the exact token ids sent to and
returned by the engine (the training data), plus everything needed to audit them (rendered prompt, normalized
messages, what the harness asked for vs. what was applied, timings). `calls` is a list so multi-call trajectories
need no schema change. Domain-free: nothing here knows about tasks, rewards or adapters.

Lossless split: decode(response_ids minus one trailing eos token, skip_special_tokens=False)
== reasoning_raw + content_raw.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable, Iterator


@dataclass
class CallRecord:
    rollout_id: str
    call_index: int
    prompt_ids: list[int] = field(default_factory=list)
    response_ids: list[int] = field(default_factory=list)
    response_logprobs: list[float] | None = None
    rendered_prompt: str = ""
    messages_in: list[dict] = field(default_factory=list)  # normalized messages that were rendered
    tools_in: list[dict] | None = None
    requested: dict[str, Any] = field(default_factory=dict)  # non-message request fields, as the harness sent them
    applied_sampling: dict[str, Any] = field(default_factory=dict)  # what went to the engine (incl. max_tokens)
    finish_reason: str | None = None  # stop | length | tool_calls | error
    reasoning_raw: str = ""  # decoded text up to and including the first </think> (all of it if never closed)
    content_raw: str = ""  # the rest of the decoded text
    tool_calls: list[dict] = field(default_factory=list)  # OpenAI-format tool calls returned to the harness
    timings: dict[str, float] = field(default_factory=dict)  # t_recv, t_render_done, t_gen_start, t_gen_end, t_resp_done
    engine_meta: dict[str, Any] = field(default_factory=dict)  # stop_reason, num_preempted, ...
    error: str | None = None  # set when the call failed (bad request, engine abort, backend exception)

    @property
    def decoded_text(self) -> str:
        return self.reasoning_raw + self.content_raw

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> CallRecord:
        return cls(**{f.name: d[f.name] for f in fields(cls) if f.name in d})


@dataclass
class RolloutTrace:
    rollout_id: str
    calls: list[CallRecord] = field(default_factory=list)
    harness_version: str | None = None
    adapter_version: str | None = None
    eval_set_hash: str | None = None
    checkpoint_id: str | None = None
    seed: int | None = None
    task_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> RolloutTrace:
        kw = {f.name: d[f.name] for f in fields(cls) if f.name in d}
        kw["calls"] = [CallRecord.from_dict(c) for c in d.get("calls", [])]
        return cls(**kw)

    def to_json(self) -> str:
        return to_json(self)

    @classmethod
    def from_json(cls, s: str) -> RolloutTrace:
        return cls.from_dict(json.loads(s))


def to_json(obj: CallRecord | RolloutTrace) -> str:
    return json.dumps(obj.to_dict(), ensure_ascii=False, separators=(",", ":"))


def from_json(s: str) -> RolloutTrace:
    return RolloutTrace.from_json(s)


def write_jsonl(path: str | Path, traces: Iterable[RolloutTrace], append: bool = True) -> int:
    """Write one trace per line; each line goes out in a single write() so concurrent appenders don't interleave."""
    n = 0
    with open(path, "a" if append else "w", encoding="utf-8") as f:
        for t in traces:
            f.write(to_json(t) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[RolloutTrace]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield RolloutTrace.from_json(line)
