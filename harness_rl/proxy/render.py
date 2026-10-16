"""Message normalization and Qwen3 rendering for the proxy, token-identical to verl's direct path.

Pi sends OpenAI chat messages (text parts arrays, `developer` role, extra keys); the Qwen3 template silently
renders non-string content as "" and ignores unknown roles, so we normalize first and reject what we can't render.
Token ids come from verl's own Continuous Token builder (`build_initial_tokens`, the path SingleTurnAgentLoop
uses) when verl is importable, else from `tokenizer.apply_chat_template(tokenize=True)`, which is what that
builder calls. The rendered text is kept for the trace.
"""

from __future__ import annotations

from typing import Any

KEPT_KEYS = {"role", "content", "reasoning_content", "tool_calls", "tool_call_id", "name"}
ROLES = {"system", "user", "assistant", "tool"}


class BadRequest(ValueError):
    pass


def _flatten(content: Any, where: str) -> str | None:
    if content is None or isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise BadRequest(f"{where}: content must be a string or a parts array")
    out = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") != "text" or not isinstance(part.get("text"), str):
            kind = part.get("type") if isinstance(part, dict) else type(part).__name__
            raise BadRequest(f"{where}: unsupported content part {kind!r} (text only)")
        out.append(part["text"])
    return "".join(out)


def normalize_messages(messages: Any) -> tuple[list[dict], list[dict]]:
    """Return (normalized messages, dropped keys as [{"index", "key", "value"}])."""
    if not isinstance(messages, list) or not messages:
        raise BadRequest("messages must be a non-empty list")
    out, dropped = [], []
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            raise BadRequest(f"messages[{i}] is not an object")
        role = "system" if m.get("role") == "developer" else m.get("role")
        if role not in ROLES:
            raise BadRequest(f"messages[{i}]: unsupported role {m.get('role')!r}")
        n = {"role": role}
        for k, v in m.items():
            if k == "role":
                continue
            if k not in KEPT_KEYS:
                dropped.append({"index": i, "key": k, "value": v})
            elif k == "content":
                n[k] = _flatten(v, f"messages[{i}]")
            elif k == "reasoning_content" and role != "assistant":
                dropped.append({"index": i, "key": k, "value": v})
            else:
                n[k] = v
        out.append(n)
    return out, dropped


def normalize_ids(x: Any) -> list[int]:
    if isinstance(x, dict) or hasattr(x, "input_ids"):
        x = x["input_ids"]
    if hasattr(x, "tolist"):
        x = x.tolist()
    x = list(x)
    if len(x) == 1 and isinstance(x[0], (list, tuple)):
        x = list(x[0])
    return [int(t) for t in x]


class Renderer:
    """Renders messages to (prompt_ids, text). One verl builder is cached per distinct chat_template_kwargs."""

    def __init__(self, tokenizer, hf_model_type: str | None = "qwen3", use_verl: bool = True):
        self.tokenizer = tokenizer
        self.hf_model_type = hf_model_type
        self._builders: dict[str, Any] = {}
        self._create = None
        if use_verl:
            try:
                from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder
                self._create = create_continuous_token_builder
            except ImportError:
                pass

    @property
    def uses_verl(self) -> bool:
        return self._create is not None

    def _builder(self, kw: dict):
        key = repr(sorted(kw.items()))
        if key not in self._builders:
            self._builders[key] = self._create(self.tokenizer, hf_model_type=self.hf_model_type, chat_template_kwargs=dict(kw))
        return self._builders[key]

    def render_ids(self, messages: list[dict], tools: list[dict] | None, chat_template_kwargs: dict) -> list[int]:
        if self._create is not None:
            return normalize_ids(self._builder(chat_template_kwargs).build_initial_tokens(messages, tools=tools or None))
        return normalize_ids(self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, tools=tools or None, **chat_template_kwargs))

    def render_text(self, messages: list[dict], tools: list[dict] | None, chat_template_kwargs: dict) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, tools=tools or None, **chat_template_kwargs)

    def render(self, messages: list[dict], tools: list[dict] | None, chat_template_kwargs: dict) -> tuple[list[int], str]:
        return self.render_ids(messages, tools, chat_template_kwargs), self.render_text(messages, tools, chat_template_kwargs)
