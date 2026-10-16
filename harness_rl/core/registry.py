"""Adapter registry. Adapters call register() at import; core never imports adapters (see core/adapter.py).

`import harness_rl.adapters` registers every built-in adapter.
"""

from harness_rl.core.adapter import Adapter

_REGISTRY: dict[str, Adapter] = {}


def register(adapter: Adapter) -> Adapter:
    if not isinstance(adapter, Adapter):
        raise TypeError(f"{adapter!r} does not implement the Adapter protocol")
    if adapter.name in _REGISTRY and _REGISTRY[adapter.name] is not adapter:
        raise ValueError(f"adapter {adapter.name!r} already registered")
    _REGISTRY[adapter.name] = adapter
    return adapter


def get(name: str) -> Adapter:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown adapter {name!r}; registered: {sorted(_REGISTRY)}") from None


def names() -> list[str]:
    return sorted(_REGISTRY)


def all_adapters() -> list[Adapter]:
    return [_REGISTRY[n] for n in names()]
