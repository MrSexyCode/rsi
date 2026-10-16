"""Turn Qwen3 response ids into OpenAI fields: lossless reasoning/content split, tool calls, finish_reason.

Split rule, on text = decode(response_ids minus one trailing eos, skip_special_tokens=False):
- `</think>` present: reasoning_raw = text up to and including the FIRST `</think>`; content_raw = the rest.
  (First, like vLLM's qwen3 parser; the Omanic reward parses after the LAST `</think>`, which gives the same
  answer from either text or content.)
- no `</think>`: everything is reasoning_raw in thinking mode (e.g. truncated mid-thought), else content_raw.
Reconstruction: text == reasoning_raw + content_raw, always.

API strings (what the harness sees), mirroring the conventional Qwen3 markup `<think>\\n{r}\\n</think>\\n\\n{c}`:
- reasoning_content = reasoning_raw minus a leading `<think>` and a trailing `</think>`, then .strip("\\n")
- content = content_raw.lstrip("\\n"), minus parsed `<tool_call>` blocks (then .strip()) when tools were offered.
The template re-renders an assistant turn as `<think>\\n` + r.strip("\\n") + `\\n</think>\\n\\n` + c.lstrip("\\n"),
so these strings round-trip through the template for well-formed outputs.
"""

from __future__ import annotations

import json
import re

THINK_START, THINK_END = "<think>", "</think>"
_TOOL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def split_reasoning(text: str, thinking: bool) -> tuple[str, str]:
    i = text.find(THINK_END)
    if i >= 0:
        j = i + len(THINK_END)
        return text[:j], text[j:]
    return (text, "") if thinking else ("", text)


def api_reasoning(reasoning_raw: str) -> str:
    r = reasoning_raw
    if r.startswith(THINK_START):
        r = r[len(THINK_START):]
    if r.endswith(THINK_END):
        r = r[: -len(THINK_END)]
    return r.strip("\n")


def api_content(content_raw: str) -> str:
    return content_raw.lstrip("\n")


def parse_tool_calls(content: str, id_prefix: str = "call_") -> tuple[str, list[dict]]:
    """Extract Qwen3 `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` blocks.

    Blocks whose body is not a JSON object with a string `name` stay in the content. Returns (content without
    the parsed blocks, OpenAI tool_calls); content is unchanged when nothing parsed.
    """
    calls, kept, pos = [], [], 0
    for m in _TOOL_RE.finditer(content):
        try:
            obj = json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
            continue
        args = obj.get("arguments", {})
        kept.append(content[pos:m.start()])
        pos = m.end()
        calls.append({"id": f"{id_prefix}{len(calls)}", "type": "function",
                      "function": {"name": obj["name"],
                                   "arguments": args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)}})
    if not calls:
        return content, []
    kept.append(content[pos:])
    return "".join(kept).strip(), calls


def decode_response(tokenizer, response_ids: list[int], eos_ids: set[int]) -> tuple[str, bool]:
    """(decoded text without one trailing eos token, whether the response ended with eos)."""
    ended = bool(response_ids) and response_ids[-1] in eos_ids
    ids = response_ids[:-1] if ended else response_ids
    return tokenizer.decode(ids, skip_special_tokens=False), ended


def finish_reason(ended_by_eos: bool, has_tool_calls: bool) -> str:
    """eos -> stop (tool_calls if any were parsed); no eos -> length.

    No eos means the engine stopped on its token budget: max_tokens, or less when the engine clamps to the context
    window (max_model_len - prompt). Both are `length` in OpenAI terms; the server records which in engine_meta.
    """
    if not ended_by_eos:
        return "length"
    return "tool_calls" if has_tool_calls else "stop"
