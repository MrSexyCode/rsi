"""Run Pi once for one rollout, isolated, and return its final assistant message (docs/pi_spike.md).

A harness package (harness_rl/harness/<name>/) holds harness.json, settings.json, provider.json, extension.ts and any
files the extension reads. Per rollout we materialize a throwaway HOME whose agent dir has the package's settings and
a models.json with the rollout's literal proxy URL (Pi does not interpolate env vars in baseUrl), plus an empty cwd.
Pi then runs with discovery off, tools off, offline, a cleared environment, and the prompt on stdin.

JsonSubprocessRunner is the plan's default mode. Pi exits 0 even when the provider call fails, so success is read
from the final message's stopReason, never from the exit code.
"""

import asyncio
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from harness_rl.core.env_paths import HARNESS_RL, PI_BIN
from harness_rl.core.versioning import content_hash

HARNESS_ROOT = HARNESS_RL / "harness"
LOCKDOWN_FLAGS = ["--no-session", "-ne", "-ns", "-np", "--no-themes", "-nc", "-na", "--no-mcp", "-nt", "--offline"]
_MESSAGE_END = b'{"type":"message_end"'


@dataclass(frozen=True)
class HarnessPackage:
    root: Path
    manifest: dict
    version: str

    @classmethod
    def load(cls, name_or_path: str | Path) -> "HarnessPackage":
        root = Path(name_or_path)
        if not root.is_absolute() and not root.exists():
            root = HARNESS_ROOT / str(name_or_path)
        root = root.resolve()
        manifest = json.loads((root / "harness.json").read_text())
        for f in ["settings.json", "provider.json", "extension.ts"]:
            if not (root / f).is_file():
                raise FileNotFoundError(f"harness package {root} lacks {f}")
        return cls(root=root, manifest=manifest, version=content_hash(root))

    @property
    def name(self) -> str:
        return self.manifest["name"]


@dataclass
class PiResult:
    final: dict | None  # last assistant message (Pi's AgentMessage), None if Pi produced none
    ok: bool  # final message exists and stopReason is "stop" or "length" (a truncated response is a valid rollout)
    error: str | None
    returncode: int | None
    stderr_tail: str
    t_start: float
    t_end: float
    assistant_messages: int = 0  # assistant message_end events seen; >1 means more than one model call
    meta: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "".join(c.get("text", "") for c in (self.final or {}).get("content", []) if c.get("type") == "text")

    @property
    def thinking(self) -> str:
        return "".join(c.get("thinking", "") for c in (self.final or {}).get("content", []) if c.get("type") == "thinking")


def materialize(harness: HarnessPackage, rollout_dir: Path, base_url: str) -> tuple[Path, Path]:
    """Create <rollout_dir>/home/.pi/agent/{settings,models}.json and an empty <rollout_dir>/cwd. Returns (home, cwd)."""
    home, cwd = rollout_dir / "home", rollout_dir / "cwd"
    agent = home / ".pi" / "agent"
    agent.mkdir(parents=True)
    cwd.mkdir()
    shutil.copyfile(harness.root / "settings.json", agent / "settings.json")
    provider = json.loads((harness.root / "provider.json").read_text())
    provider["baseUrl"] = base_url
    provider_name = harness.manifest["model"].split("/", 1)[0]
    (agent / "models.json").write_text(json.dumps({"providers": {provider_name: provider}}, indent=1))
    return home, cwd


def pi_command(harness: HarnessPackage) -> list[str]:
    return [str(PI_BIN), "-p", "--mode", "json", *LOCKDOWN_FLAGS, "-e", str(harness.root / "extension.ts"),
            "--thinking", harness.manifest.get("thinking", "off"), "--model", harness.manifest["model"]]


def pi_env(home: Path, rollout_id: str) -> dict[str, str]:
    return {"PATH": "/usr/bin:/bin", "HOME": str(home), "PI_OFFLINE": "1", "PI_TELEMETRY": "0",
            "PI_SKIP_VERSION_CHECK": "1", "ROLLOUT_ID": rollout_id}


class JsonSubprocessRunner:
    """One `pi -p --mode json` subprocess per rollout."""

    def __init__(self, harness: HarnessPackage, work_root: str | Path | None = None, timeout_s: float = 3600.0,
                 keep_dirs: bool = False):
        self.harness = harness
        self.work_root = Path(work_root or tempfile.gettempdir()) / "harness_rl_rollouts"
        self.work_root.mkdir(parents=True, exist_ok=True)
        self.timeout_s = timeout_s
        self.keep_dirs = keep_dirs
        self._cmd = pi_command(harness)

    async def run(self, prompt: str, rollout_id: str, base_url: str) -> PiResult:
        if prompt != prompt.strip():
            raise ValueError("Pi strips surrounding whitespace from a stdin prompt; pass a stripped prompt")
        if not prompt:
            raise ValueError("empty prompt")
        rollout_dir = Path(tempfile.mkdtemp(prefix=f"{rollout_id}_", dir=self.work_root))
        t_start = time.time()
        proc = None
        try:
            home, cwd = materialize(self.harness, rollout_dir, base_url)
            proc = await asyncio.create_subprocess_exec(
                *self._cmd, cwd=cwd, env=pi_env(home, rollout_id), stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, limit=1 << 26)
            out, err = await asyncio.wait_for(proc.communicate(prompt.encode()), timeout=self.timeout_s)
        except asyncio.TimeoutError:
            if proc is not None and proc.returncode is None:
                proc.kill()
                await proc.wait()
            return PiResult(None, False, f"timeout after {self.timeout_s}s", None, "", t_start, time.time())
        finally:
            if not self.keep_dirs:
                shutil.rmtree(rollout_dir, ignore_errors=True)
        t_end = time.time()
        ends = [json.loads(line) for line in out.splitlines() if line.startswith(_MESSAGE_END)]
        assistants = [e["message"] for e in ends if e["message"].get("role") == "assistant"]
        final = assistants[-1] if assistants else None
        stop = final.get("stopReason") if final else None
        error = None
        if final is None:
            error = f"no assistant message (rc={proc.returncode})"
        elif stop not in ("stop", "length"):
            error = f"stopReason={stop}: {final.get('errorMessage', '')}"
        return PiResult(final=final, ok=error is None, error=error, returncode=proc.returncode,
                        stderr_tail=err.decode(errors="replace")[-2000:], t_start=t_start, t_end=t_end,
                        assistant_messages=len(assistants))
