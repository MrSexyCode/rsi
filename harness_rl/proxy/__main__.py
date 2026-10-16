"""Standalone proxy: `python -m harness_rl.proxy --backend vllm --vllm-url http://127.0.0.1:8000 --port 18000`.

Outside verl there is no in-process caller, so the admin routes are on: POST /admin/register/{rollout_id}
with {"sampling_params", "max_tokens", "chat_template_kwargs"}, then POST /admin/collect/{rollout_id} for the records.
`--backend fake` answers every call with --fake-text, for smoke tests without a GPU.
"""

import argparse
import asyncio

from transformers import AutoTokenizer

from harness_rl.core.env_paths import MODEL_ID
from harness_rl.proxy.backends import FakeBackend, VLLMHTTPBackend, qwen3_script
from harness_rl.proxy.server import ModelProxy


async def main(args) -> None:
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if args.backend == "vllm":
        backend = VLLMHTTPBackend(args.vllm_url, args.vllm_model or args.tokenizer)
    else:
        backend = FakeBackend(qwen3_script(tok, args.fake_text), delay_s=args.fake_delay)
    proxy = ModelProxy(tok, backend, host=args.host, port=args.port, model_name=args.model_name,
                       keepalive_s=args.keepalive, admin=True)
    await proxy.start()
    print(f"proxy listening on {proxy.base_url}", flush=True)
    try:
        await asyncio.Event().wait()
    finally:
        await proxy.stop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["vllm", "fake"], default="vllm")
    ap.add_argument("--vllm-url", default="http://127.0.0.1:8000")
    ap.add_argument("--vllm-model", default=None, help="served model name (default: --tokenizer)")
    ap.add_argument("--tokenizer", default=MODEL_ID)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--model-name", default="qwen3-8b", help="model id reported to the harness")
    ap.add_argument("--keepalive", type=float, default=15.0)
    ap.add_argument("--fake-text", default="<think>\nfake reasoning\n</think>\n\nAnswer: A")
    ap.add_argument("--fake-delay", type=float, default=0.0)
    try:
        asyncio.run(main(ap.parse_args()))
    except KeyboardInterrupt:
        pass
