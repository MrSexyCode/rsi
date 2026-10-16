"""0.6 eval runner: score a checkpoint on frozen eval sets, through Pi (a harness package) or directly.

Both paths share everything but Pi: the same proxy renders and tokenizes the prompt, the same vLLM engine samples with
the same per-sample seed, and the same scorer reads decode(response_ids, skip_special_tokens=True), the text verl's
reward manager sees. --direct posts the task's single user message to the proxy itself instead of running Pi, so a
direct-vs-Pi difference isolates what Pi adds.

    python -m harness_rl.evals.run --checkpoint Qwen/Qwen3-8B --harness vanilla --sets dev:natural --samples 4
    python -m harness_rl.evals.run ... --direct
Engine: --vllm-url URL of a running `vllm serve` (served model name via --served-model), or --launch to start one
here (--gpus 0,1,...; data-parallel over them, TP1). Sets: names, or dev:all / dev:natural / dev:babilong / dev:ruler.

Writes <out>/<run_name>/{results.jsonl (one line per task x sample), summary.json} and appends one row per set to
docs/evals.md. Per set: mean@k = mean over tasks of the mean score over k samples; stderr = std of task means / sqrt(n).
"""

import argparse
import asyncio
import hashlib
import json
import math
import os
import signal
import socket
import statistics
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
from transformers import AutoTokenizer

from harness_rl.core.env_paths import HARNESS_RL, VERL_DIR
from harness_rl.evals.loader import list_sets, load_manifest, load_set, set_hash
from harness_rl.evals.scoring import extract_answer, score
from harness_rl.pi_loop.runner import HarnessPackage, JsonSubprocessRunner
from harness_rl.proxy.backends import VLLMHTTPBackend
from harness_rl.proxy.server import ModelProxy

EVALS_MD = HARNESS_RL / "docs" / "evals.md"
OUT_ROOT = HARNESS_RL / "outputs" / "evals"
# Qwen3 thinking-mode sampling (generation_config.json), as in the playground's existing evals.
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
MAX_NEW_TOKENS = 16384  # thinking budget; prompts reach ~8.7k tokens, max_model_len below covers both
MAX_MODEL_LEN = 40960
GROUPS = {
    "natural": lambda n: n.startswith(("musique", "2wiki", "hotpotqa", "babilong", "ruler")),
    "babilong": lambda n: n.startswith("babilong"),
    "ruler": lambda n: n.startswith("ruler"),
    "all": lambda n: True,
}


def resolve_sets(specs: list[str]) -> list[str]:
    names = []
    for s in specs:
        if s.startswith("dev:"):
            names += [n for n in list_sets() if GROUPS[s[4:]](n)]
        else:
            names.append(s)
    return list(dict.fromkeys(names))


def sample_seed(base_seed: int, set_name: str, task_id: str, k: int) -> int:
    h = hashlib.sha256(f"{base_seed}|{set_name}|{task_id}|{k}".encode()).hexdigest()
    return int(h[:8], 16)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def launch_vllm(checkpoint: str, gpus: str, log_path: Path) -> tuple[subprocess.Popen, str]:
    port = free_port()
    n = len(gpus.split(","))
    cmd = [str(VERL_DIR / ".venv/bin/vllm"), "serve", checkpoint, "--port", str(port), "--served-model-name", "evalmodel",
           "--max-model-len", str(MAX_MODEL_LEN), "--gpu-memory-utilization", "0.9", "--data-parallel-size", str(n),
           "--enable-prefix-caching"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus)
    proc = subprocess.Popen(cmd, env=env, stdout=open(log_path, "w"), stderr=subprocess.STDOUT, start_new_session=True)
    url = f"http://127.0.0.1:{port}"
    return proc, url


async def wait_ready(url: str, proc: subprocess.Popen | None, timeout_s: float = 1800):
    t0 = time.time()
    async with aiohttp.ClientSession() as s:
        while time.time() - t0 < timeout_s:
            if proc is not None and proc.poll() is not None:
                raise RuntimeError(f"vLLM exited with {proc.returncode}")
            try:
                async with s.get(f"{url}/v1/models") as r:
                    if r.status == 200:
                        return
            except aiohttp.ClientError:
                pass
            await asyncio.sleep(5)
    raise TimeoutError("vLLM did not become ready")


async def run_direct(proxy: ModelProxy, http: aiohttp.ClientSession, rid: str, prompt: str) -> dict:
    """The direct path: the same single user message Pi would send, posted to the proxy without Pi."""
    body = {"model": proxy.model_name, "messages": [{"role": "user", "content": prompt}], "stream": False}
    async with http.post(f"{proxy.rollout_base_url(rid)}/chat/completions", json=body) as r:
        await r.read()
        return {"ok": r.status == 200, "error": None if r.status == 200 else f"HTTP {r.status}", "assistant_messages": 1}


async def evaluate(args) -> dict:
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.checkpoint)
    sets = resolve_sets(args.sets)
    splits = ("dev", "final") if args.final else ("dev",)
    tasks = {name: load_set(name, splits=splits, allow_final=args.final)[: args.limit] for name in sets}

    vllm_proc = None
    if args.launch:
        log = args.run_dir / "vllm.log"
        vllm_proc, url = launch_vllm(args.checkpoint, args.gpus, log)
        served = "evalmodel"
    else:
        url, served = args.vllm_url, args.served_model
    try:
        await wait_ready(url, vllm_proc)
        backend = VLLMHTTPBackend(url, served)
        proxy = ModelProxy(tok, backend, render_workers=1)
        await proxy.start()
        harness = None if args.direct else HarnessPackage.load(args.harness)
        runner = None if args.direct else JsonSubprocessRunner(harness)
        sem = asyncio.Semaphore(args.concurrency)
        results_path = args.run_dir / "results.jsonl"
        out_f = open(results_path, "w")
        done = [0]
        total = sum(len(v) for v in tasks.values()) * args.samples
        t_start = time.time()

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=7200)) as http:
            async def one(set_name: str, task: dict, k: int):
                async with sem:
                    rid = f"e{uuid.uuid4().hex}"
                    seed = sample_seed(args.seed, set_name, task["id"], k)
                    proxy.register_rollout(rid, sampling_params={**SAMPLING, "seed": seed, "logprobs": False},
                                           max_tokens=args.max_new_tokens,
                                           chat_template_kwargs={"enable_thinking": True})
                    t0 = time.time()
                    try:
                        if args.direct:
                            res = await run_direct(proxy, http, rid, task["prompt"])
                        else:
                            r = await runner.run(task["prompt"], rid, proxy.rollout_base_url(rid))
                            res = {"ok": r.ok, "error": r.error, "assistant_messages": r.assistant_messages}
                    finally:
                        calls = proxy.collect(rid)
                    valid = res["ok"] and len(calls) == 1 and res["assistant_messages"] == 1
                    text = tok.decode(calls[0].response_ids, skip_special_tokens=True) if valid else ""
                    sc = score(task, text) if valid else {"score": 0.0}
                    rec = {"set": set_name, "id": task["id"], "split": task["split"], "sample": k, "seed": seed,
                           "valid": valid, "error": res["error"] if not valid else None, "num_calls": len(calls),
                           "score": sc["score"], "answer_found": valid and extract_answer(text) is not None,
                           "components": {kk: v for kk, v in sc.items() if kk != "score"},
                           "finish_reason": calls[0].finish_reason if calls else None,
                           "n_prompt_tokens": len(calls[0].prompt_ids) if calls else None,
                           "n_response_tokens": len(calls[0].response_ids) if calls else None,
                           "wall_s": time.time() - t0, "response": text}
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    done[0] += 1
                    if done[0] % 200 == 0 or done[0] == total:
                        out_f.flush()
                        print(f"[{done[0]}/{total}] {time.time() - t_start:.0f}s", flush=True)
                    return rec

            jobs = [one(n, t, k) for n, ts in tasks.items() for t in ts for k in range(args.samples)]
            recs = await asyncio.gather(*jobs)
        out_f.close()
        await proxy.stop()
        await backend.close()
    finally:
        if vllm_proc is not None:
            os.killpg(vllm_proc.pid, signal.SIGTERM)
            vllm_proc.wait()

    per_set = {}
    for name in sets:
        rs = [r for r in recs if r["set"] == name]
        by_task: dict[str, list[float]] = {}
        for r in rs:
            by_task.setdefault(r["id"], []).append(r["score"])
        means = [statistics.mean(v) for v in by_task.values()]
        n = len(means)
        per_set[name] = {
            "n_tasks": n, "samples": args.samples, "mean": statistics.mean(means) if n else None,
            "stderr": statistics.stdev(means) / math.sqrt(n) if n > 1 else None,
            "invalid": sum(not r["valid"] for r in rs),
            "truncated": sum(r["finish_reason"] == "length" for r in rs) / max(1, len(rs)),
            "answer_found": sum(r["answer_found"] for r in rs) / max(1, len(rs)),
            "mean_response_tokens": statistics.mean(r["n_response_tokens"] or 0 for r in rs) if rs else None,
            "eval_set_hash": set_hash(name), "splits": list(splits),
        }
    return {
        "run_name": args.run_name, "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": args.checkpoint, "mode": "direct" if args.direct else "pi",
        "harness": None if args.direct else harness.name,
        "harness_version": None if args.direct else harness.version,
        "seed": args.seed, "sampling": SAMPLING, "max_new_tokens": args.max_new_tokens, "samples": args.samples,
        "sets": per_set, "wall_s": time.time() - t_start,
    }


def append_evals_md(summary: dict):
    new = not EVALS_MD.exists()
    with open(EVALS_MD, "a") as f:
        if new:
            f.write("# Eval results\n\nOne row per (run, set). mean@k ± stderr over tasks. Written by `evals/run.py`.\n\n"
                    "| date | run | checkpoint | mode | harness@version | set | n | mean@k | ± | invalid | trunc | resp tok |\n"
                    "|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        hv = f"{summary['harness']}@{summary['harness_version'][:8]}" if summary["harness"] else "—"
        for name, s in summary["sets"].items():
            f.write(f"| {summary['time'][:10]} | {summary['run_name']} | {summary['checkpoint']} | {summary['mode']} | {hv} "
                    f"| {name} | {s['n_tasks']} | {s['mean']:.4f} | {s['stderr'] or 0:.4f} | {s['invalid']} "
                    f"| {s['truncated']:.3f} | {s['mean_response_tokens']:.0f} |\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="HF id or local HF-format model dir")
    ap.add_argument("--tokenizer", default=None, help="defaults to --checkpoint")
    ap.add_argument("--harness", default="vanilla")
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--direct", action="store_true", help="bypass Pi (parity reference)")
    ap.add_argument("--final", action="store_true", help="also evaluate the reserved final split")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--concurrency", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None, help="first N tasks per set (smoke tests)")
    ap.add_argument("--vllm-url", default=None)
    ap.add_argument("--served-model", default="evalmodel")
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--no-md", action="store_true", help="do not append to docs/evals.md")
    args = ap.parse_args()
    if not args.launch and not args.vllm_url:
        ap.error("give --vllm-url or --launch")
    mode = "direct" if args.direct else args.harness
    args.run_name = args.run_name or f"{datetime.now():%m%d_%H%M}_{Path(args.checkpoint).name}_{mode}"
    args.run_dir = OUT_ROOT / args.run_name
    args.run_dir.mkdir(parents=True, exist_ok=True)

    summary = asyncio.run(evaluate(args))
    (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    if not args.no_md:
        append_evals_md(summary)
    for name, s in summary["sets"].items():
        print(f"{name:28s} n={s['n_tasks']:4d} mean@{args.samples}={s['mean']:.4f} ± {s['stderr'] or 0:.4f} "
              f"invalid={s['invalid']} trunc={s['truncated']:.3f}")


if __name__ == "__main__":
    main()
