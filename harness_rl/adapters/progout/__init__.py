"""Program-output prediction adapter (plan 1.6). Importing this package registers it."""

from harness_rl.adapters.progout.adapter import ProgoutAdapter
from harness_rl.core import registry

ADAPTER = registry.register(ProgoutAdapter())
