"""In-context multi-hop world compiler (plan 1.2-1.4). Importing this package registers the adapter."""

from harness_rl.adapters.multihop.adapter import MultihopAdapter
from harness_rl.core import registry

ADAPTER = registry.register(MultihopAdapter())
