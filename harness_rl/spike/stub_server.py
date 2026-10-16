"""Logging stub of an OpenAI-compatible chat endpoint, for the 0.2 Pi spike.

Every request to any path is appended to --log as one JSON line (path, headers, body). Chat completions
get a canned reply, streamed as SSE when the request asks for it. The reply can carry Qwen3-style
`reasoning_content` so we can see how Pi handles it. An optional --delay simulates model time.

With --chunks N the reasoning is split into N SSE events sent in bursts every --burst-ms over --delay seconds,
to mimic a token-by-token vLLM stream. --no-body-log keeps only timing fields (for load tests).

Usage: python stub_server.py --port 18080 --log /tmp/x.jsonl [--reasoning "..."] [--content "..."] [--delay 0]
"""

import argparse
import asyncio
import json
import time

from aiohttp import web


def make_handler(args):
    async def handle(request: web.Request) -> web.StreamResponse:
        raw = await request.read()
        try:
            body = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            body = raw.decode(errors="replace")
        t_arrival = time.time()
        rec = {"t": t_arrival, "method": request.method, "path": request.path_qs,
               "headers": {k: v for k, v in request.headers.items() if k.lower() != "authorization"},
               "has_authorization": "authorization" in {k.lower() for k in request.headers}, "body": body}
        if args.no_body_log:
            rec.pop("body")
            rec.pop("headers")
            rec["rollout_id"] = request.headers.get("X-Rollout-Id")
        if not args.timing_only_on_end:
            with open(args.log, "a") as f:
                f.write(json.dumps(rec) + "\n")

        if request.path.endswith("/models"):
            return web.json_response({"object": "list", "data": [{"id": "qwen3-8b", "object": "model"}]})
        if not request.path.endswith("/chat/completions"):
            return web.json_response({"error": "not found"}, status=404)

        if args.delay and not args.chunks:
            await asyncio.sleep(args.delay)
        model = (body or {}).get("model", "qwen3-8b")
        created = int(time.time())
        cid = "chatcmpl-stub"
        usage = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}

        if isinstance(body, dict) and body.get("stream"):
            resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
            await resp.prepare(request)

            async def send(delta, finish=None, extra=None):
                chunk = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                         "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                if extra:
                    chunk.update(extra)
                await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())

            await send({"role": "assistant", "content": ""})
            if args.chunks:
                pieces = [args.chunk_text] * args.chunks
                n_bursts = max(1, int(args.delay * 1000 / args.burst_ms)) if args.delay else 1
                per = -(-len(pieces) // n_bursts)
                for i in range(0, len(pieces), per):
                    for piece in pieces[i:i + per]:
                        await send({args.reasoning_field: piece})
                    if args.delay:
                        await asyncio.sleep(args.burst_ms / 1000)
            elif args.reasoning:
                for piece in _pieces(args.reasoning):
                    await send({args.reasoning_field: piece})
            for piece in _pieces(args.content):
                await send({"content": piece})
            await send({}, finish="stop")
            if (body.get("stream_options") or {}).get("include_usage"):
                chunk = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                         "choices": [], "usage": usage}
                await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
            await resp.write(b"data: [DONE]\n\n")
            await resp.write_eof()
            if args.timing_only_on_end:
                rec["t_end"] = time.time()
                with open(args.log, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            return resp

        msg = {"role": "assistant", "content": args.content}
        if args.reasoning:
            msg[args.reasoning_field] = args.reasoning
        return web.json_response({"id": cid, "object": "chat.completion", "created": created, "model": model,
                                  "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}], "usage": usage})

    return handle


def _pieces(text, n=3):
    k = max(1, len(text) // n)
    return [text[i:i + k] for i in range(0, len(text), k)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--content", default="Answer: B")
    ap.add_argument("--reasoning", default="")
    ap.add_argument("--reasoning-field", default="reasoning_content")
    ap.add_argument("--delay", type=float, default=0.0)
    ap.add_argument("--chunks", type=int, default=0)
    ap.add_argument("--chunk-text", default="word")
    ap.add_argument("--burst-ms", type=float, default=50.0)
    ap.add_argument("--no-body-log", action="store_true")
    ap.add_argument("--timing-only-on-end", action="store_true", help="log once per request, after the stream ends")
    args = ap.parse_args()
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.router.add_route("*", "/{tail:.*}", make_handler(args))
    web.run_app(app, host="127.0.0.1", port=args.port, print=None, access_log=None)


if __name__ == "__main__":
    main()
