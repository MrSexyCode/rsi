"""Fixed locations of the existing environment and runtimes that harness_rl wraps (see docs/inventory.md)."""

from pathlib import Path

PLAYGROUND = Path("/home/ubuntu/playground")
VERL_DIR = PLAYGROUND / "verl"
VERL_PYTHON = VERL_DIR / ".venv/bin/python"
OMANIC_REWARD = PLAYGROUND / "omanic/reward.py"
OMANIC_DATA = Path.home() / "data/omanic"

MODEL_ID = "Qwen/Qwen3-8B"

HARNESS_RL = Path(__file__).resolve().parents[1]
PI_RUNTIME = HARNESS_RL / "pi_runtime"
PI_BIN = PI_RUNTIME / "node_modules/.bin/pi"
