"""0.2.6 denominator: median wall time of one direct rollout under training-like load.

Sends N Omanic train prompts (thinking on, T=1.0, top_p=1, max 8192 new tokens, as in the DAPO config) to a
vLLM server with C requests in flight, and records each request's wall time and completion length.
verl runs 4 TP2 replicas with 256 x 5 = 1280 rollouts per step, i.e. ~320 in flight per replica.

Usage: python measure_rollout_time.py --url http://127.0.0.1:18300 --n 640 --concurrency 320 --out rt.json
"""

import argparse
import asyncio
import json
import statistics
import time

import aiohttp
import pandas as pd


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--n", type=int, default=640)
    ap.add_argument("--concurrency", type=int, default=320)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_parquet("/home/ubuntu/data/omanic/train.parquet").sample(n=args.n // 5, random_state=args.seed)
    prompts = [list(p) for p in df.prompt for _ in range(5)]  # n=5 samples per prompt, as rollout.n
    sem = asyncio.Semaphore(args.concurrency)
    recs = []

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600)) as http:
        async def one(i, msgs):
            async with sem:
                body = {"model": "qwen3-8b", "messages": [{"role": m["role"], "content": m["content"]} for m in msgs],
                        "temperature": 1.0, "top_p": 1.0, "max_tokens": 8192, "seed": args.seed * 100000 + i,
                        "chat_template_kwargs": {"enable_thinking": True}}
                t0 = time.time()
                async with http.post(f"{args.url}/v1/chat/completions", json=body) as r:
                    out = await r.json()
                recs.append({"i": i, "wall_s": time.time() - t0, "completion_tokens": out["usage"]["completion_tokens"],
                             "finish": out["choices"][0]["finish_reason"]})

        t0 = time.time()
        await asyncio.gather(*(one(i, m) for i, m in enumerate(prompts)))
        total = time.time() - t0

    walls = sorted(r["wall_s"] for r in recs)
    summary = {"n": len(recs), "concurrency": args.concurrency, "batch_wall_s": total,
               "wall_median_s": statistics.median(walls), "wall_p10_s": walls[len(walls) // 10],
               "wall_p90_s": walls[9 * len(walls) // 10], "wall_max_s": walls[-1],
               "tokens_median": statistics.median(r["completion_tokens"] for r in recs),
               "tokens_mean": statistics.mean(r["completion_tokens"] for r in recs),
               "truncated_frac": sum(r["finish"] == "length" for r in recs) / len(recs)}
    json.dump({"summary": summary, "records": recs}, open(args.out, "w"))
    print(json.dumps(summary))


if __name__ == "__main__":
    asyncio.run(main())
