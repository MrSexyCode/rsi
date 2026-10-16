"""0.2.6 spike: Pi wall-time overhead per rollout, excluding model time, for three invocation modes.

    json  one `pi -p --mode json` subprocess per rollout, prompt on stdin
    rpc   a pool of C long-lived `pi --mode rpc` processes; each rollout = new_session + prompt
    sdk   G long-lived Node gateways (sdk_gateway.mjs) running C concurrent SDK sessions between them

A stub endpoint streams `--chunks` SSE events over `--delay` seconds (simulated model time) and logs, per
request, arrival and stream-end times. Per-rollout overhead = client wall time - stub (arrival -> end) time,
i.e. everything Pi adds: process start, config load, request build, stream parsing, shutdown.

Usage: python bench_overhead.py --mode json --concurrency 64 --rollouts 256 --delay 5 --chunks 3000 --out results.jsonl
"""

import argparse
import asyncio
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PI_BIN = HERE.parent / "pi_runtime/node_modules/.bin/pi"
STUB = HERE / "stub_server.py"
GATEWAY = HERE / "sdk_gateway.mjs"
PY = "/home/ubuntu/playground/verl/.venv/bin/python"

PROMPT = (
    "Answer the following multiple-choice question. Answering it requires several reasoning steps.\n\n"
    "Question: The city where the founder of the company that built the first steam locomotive in the region "
    "was born had what population in the census taken in the year the company was founded?\n\nOptions:\n"
    "A. 12,000\nB. 54,000\nC. 120,500\nD. 2,300\n\n"
    "Reason step by step. Then end your response with exactly this line:\nAnswer: <the letter of the correct option>"
)
FORCE_EXT = """export default function (pi: any) {
\tpi.on("before_agent_start", () => ({ systemPrompt: "" }));
}
"""
CLI_FLAGS = ["--no-session", "-ne", "-ns", "-np", "--no-themes", "-nc", "-na", "--no-mcp", "-nt", "--offline",
             "--thinking", "off", "--model", "rlproxy/qwen3-8b"]


def write_agent_dir(root: Path, port: int, header_from_env: bool) -> Path:
    agent = root / ".pi/agent"
    agent.mkdir(parents=True, exist_ok=True)
    provider = {"baseUrl": f"http://127.0.0.1:{port}/v1", "api": "openai-completions",
                "apiKey": "${ROLLOUT_ID}" if header_from_env else "sdk",
                "compat": {"supportsDeveloperRole": False, "supportsStore": False},
                "models": [{"id": "qwen3-8b", "reasoning": True, "contextWindow": 40960, "maxTokens": 8192}]}
    if header_from_env:
        provider["headers"] = {"X-Rollout-Id": "${ROLLOUT_ID}"}
    (agent / "models.json").write_text(json.dumps({"providers": {"rlproxy": provider}}))
    return agent


def iso_env(home: Path, rollout_id: str) -> dict:
    return {"PATH": "/usr/bin:/bin", "HOME": str(home), "PI_OFFLINE": "1", "PI_TELEMETRY": "0",
            "PI_SKIP_VERSION_CHECK": "1", "ROLLOUT_ID": rollout_id}


async def read_jsonl_until(stream, stop):
    """Read LF-delimited JSON records until stop(record) is true; return all records."""
    out = []
    while True:
        line = await stream.readline()
        if not line:
            return out
        rec = json.loads(line)
        out.append(rec)
        if stop(rec):
            return out


def final_assistant(records):
    ends = [r["message"] for r in records if r.get("type") == "message_end" and r["message"].get("role") == "assistant"]
    return ends[-1] if ends else None


async def run_json(args, root, home, cwd):
    sem = asyncio.Semaphore(args.concurrency)
    results = []

    async def one(i):
        rid = f"j{i}"
        async with sem:
            t0 = time.time()
            proc = await asyncio.create_subprocess_exec(
                str(PI_BIN), "-p", "--mode", "json", *CLI_FLAGS, "-e", str(root / "force_prompt.ts"),
                cwd=cwd, env=iso_env(home, rid), stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, limit=1 << 26)
            out, err = await proc.communicate(PROMPT.encode())
            t1 = time.time()
            # Like PiAgentLoop: only the final message_end records matter; skip parsing the per-delta events.
            msg = final_assistant([json.loads(l) for l in out.splitlines() if l.startswith(b'{"type":"message_end"')])
            results.append({"id": rid, "t_start": t0, "t_end": t1, "stopReason": msg and msg.get("stopReason"),
                            "rc": proc.returncode, "err": err.decode()[-300:] if proc.returncode else ""})

    await asyncio.gather(*(one(i) for i in range(args.rollouts)))
    return results


async def run_rpc(args, root, home, cwd):
    queue = asyncio.Queue()
    for i in range(args.rollouts):
        queue.put_nowait(i)
    results = []

    async def worker(w):
        rid = f"w{w}"
        proc = await asyncio.create_subprocess_exec(
            str(PI_BIN), "--mode", "rpc", *CLI_FLAGS, "-e", str(root / "force_prompt.ts"),
            cwd=cwd, env=iso_env(home, rid), stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL, limit=1 << 26)
        seq = 0
        while True:
            try:
                i = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            t0 = time.time()
            proc.stdin.write((json.dumps({"id": f"n{i}", "type": "new_session"}) + "\n").encode())
            await read_jsonl_until(proc.stdout, lambda r: r.get("type") == "response" and r.get("id") == f"n{i}")
            proc.stdin.write((json.dumps({"id": f"p{i}", "type": "prompt", "message": PROMPT}) + "\n").encode())
            recs = await read_jsonl_until(proc.stdout, lambda r: r.get("type") == "agent_settled")
            t1 = time.time()
            msg = final_assistant(recs)
            results.append({"id": f"{rid}#{seq}", "t_start": t0, "t_end": t1, "stopReason": msg and msg.get("stopReason")})
            seq += 1
        proc.stdin.close()
        await proc.wait()

    await asyncio.gather(*(worker(w) for w in range(args.concurrency)))
    return results


async def run_sdk(args, root, home, cwd):
    n_gw = args.gateways
    gws = []
    for g in range(n_gw):
        env = iso_env(home, "unused") | {"PI_AGENT_DIR": str(home / ".pi/agent"), "PI_CWD": str(cwd)}
        proc = await asyncio.create_subprocess_exec(
            "node", str(GATEWAY), cwd=cwd, env=env, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL, limit=1 << 26)
        ready = json.loads(await proc.stdout.readline())
        assert ready.get("ready"), ready
        gws.append(proc)
    sem = asyncio.Semaphore(args.concurrency)
    waiters = {}
    results = []

    async def reader(proc):
        while line := await proc.stdout.readline():
            rec = json.loads(line)
            waiters.pop(rec["id"]).set_result(rec)

    readers = [asyncio.create_task(reader(p)) for p in gws]

    async def one(i):
        rid = f"s{i}"
        async with sem:
            fut = asyncio.get_running_loop().create_future()
            waiters[rid] = fut
            t0 = time.time()
            gws[i % n_gw].stdin.write((json.dumps({"id": rid, "prompt": PROMPT}) + "\n").encode())
            rec = await fut
            results.append({"id": rid, "t_start": t0, "t_end": time.time(), "stopReason": rec.get("stopReason"),
                            "err": rec.get("error") or rec.get("errorMessage")})

    await asyncio.gather(*(one(i) for i in range(args.rollouts)))
    for p in gws:
        p.stdin.close()
        await p.wait()
    for r in readers:
        r.cancel()
    return results


def match_stub(results, stub_recs, mode):
    """Attach the stub's (t_end - t) to each result. json/sdk: by rollout id. rpc: by worker id + order."""
    by_id = {}
    for r in stub_recs:
        by_id.setdefault(r.get("rollout_id"), []).append(r)
    for lst in by_id.values():
        lst.sort(key=lambda r: r["t"])
    for res in results:
        if mode == "rpc":
            w, seq = res["id"].split("#")
            lst = by_id.get(w, [])
            s = lst[int(seq)] if int(seq) < len(lst) else None
        else:
            lst = by_id.get(res["id"], [])
            s = lst[0] if len(lst) == 1 else None
        res["model_s"] = (s["t_end"] - s["t"]) if s else None
        res["wall_s"] = res["t_end"] - res["t_start"]
        res["overhead_s"] = res["wall_s"] - res["model_s"] if s else None
    return results


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["json", "rpc", "sdk"], required=True)
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--rollouts", type=int, required=True)
    ap.add_argument("--delay", type=float, default=5.0)
    ap.add_argument("--chunks", type=int, default=3000)
    ap.add_argument("--gateways", type=int, default=1)
    ap.add_argument("--burst-chars", type=int, default=0,
                    help="proxy-like stream: wait --delay, then send this many reasoning chars in 3 chunks (overrides --chunks)")
    ap.add_argument("--port", type=int, default=18190)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(tempfile.mkdtemp(prefix="pibench_"))
    home, cwd = root / "home", root / "cwd"
    cwd.mkdir(parents=True)
    write_agent_dir(home, args.port, header_from_env=args.mode != "sdk")
    (root / "force_prompt.ts").write_text(FORCE_EXT)
    stub_log = root / "stub.jsonl"
    stub = subprocess.Popen([PY, str(STUB), "--port", str(args.port), "--log", str(stub_log), "--content", "Answer: B",
                             *(["--reasoning", "w" * args.burst_chars] if args.burst_chars else ["--chunks", str(args.chunks)]),
                             "--delay", str(args.delay), "--no-body-log",
                             "--timing-only-on-end"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.5)
    try:
        t0 = time.time()
        results = asyncio.run({"json": run_json, "rpc": run_rpc, "sdk": run_sdk}[args.mode](args, root, home, cwd))
        total = time.time() - t0
    finally:
        stub.terminate()
        stub.wait()
    stub_recs = [json.loads(l) for l in open(stub_log)] if stub_log.exists() else []
    results = match_stub(results, stub_recs, args.mode)
    ok = [r for r in results if r.get("stopReason") == "stop" and r["overhead_s"] is not None]
    ov = [r["overhead_s"] for r in ok]
    model = [r["model_s"] for r in ok]
    summary = {"mode": args.mode, "concurrency": args.concurrency, "rollouts": args.rollouts, "gateways": args.gateways,
               "delay": args.delay, "chunks": args.chunks, "burst_chars": args.burst_chars, "ok": len(ok), "failed": len(results) - len(ok),
               "overhead_median_s": statistics.median(ov) if ov else None, "overhead_p90_s": pct(ov, 0.9) if ov else None,
               "overhead_p99_s": pct(ov, 0.99) if ov else None, "overhead_max_s": max(ov) if ov else None,
               "model_median_s": statistics.median(model) if model else None, "batch_wall_s": total,
               "first_failure": next((r for r in results if r not in ok), None)}
    with open(args.out, "a") as f:
        f.write(json.dumps(summary) + "\n")
    print(json.dumps(summary))
    shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
