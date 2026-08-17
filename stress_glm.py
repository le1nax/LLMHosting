#!/usr/bin/env python3
"""Minimal stress test for the vLLM OpenAI endpoint (GLM-5.2).

Fires NUM_PARALLEL requests continuously with random prompts and thinking on
max, until the user stops it with Ctrl-C. Each request may generate up to
MAX_TOKENS tokens but stops early at the end token (ignore_eos=False).

    python stress_glm.py                 # 200 parallel, up to 128000 tokens
    NUM_PARALLEL=50 MAX_TOKENS=4096 python stress_glm.py
"""
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

HOST = os.environ.get("VLLM_HOST", "http://134.61.53.224:52223")   # head-node IP
MODEL = os.environ.get("VLLM_MODEL", "glm-5.2-fp8")
NUM_PARALLEL = int(os.environ.get("NUM_PARALLEL", "200"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "127000"))  # < 128000 ctx (input+output)
PRINT_OUTPUT = os.environ.get("PRINT_OUTPUT", "True").lower() in ("1", "true", "yes")

TOPICS = [
    "the history of computing", "quantum entanglement", "the French Revolution",
    "how neural networks learn", "the migration of monarch butterflies",
    "the economics of renewable energy", "the philosophy of consciousness",
    "the formation of galaxies", "the evolution of human language",
    "the chemistry of fermentation", "deep ocean ecosystems",
    "the mathematics of prime numbers", "the Roman aqueduct system",
    "climate feedback loops", "the design of distributed systems",
    "the biology of aging", "the art of medieval cathedrals",
    "the future of space exploration", "the psychology of decision making",
    "the cryptography behind blockchains",
]
TEMPLATES = [
    "Write a long, detailed essay about {}.",
    "Explain {} step by step, in great depth.",
    "Give a comprehensive technical overview of {}.",
    "Discuss {} from multiple perspectives and reason carefully.",
    "Solve a hard problem related to {} and show all your reasoning.",
]

_lock = threading.Lock()
_stats = {"done": 0, "errors": 0, "tokens": 0, "tokens_seen": 0}
_stop = threading.Event()

# Response-length buckets (by actual completion tokens).
BUCKETS = [500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, float("inf")]
_hist = {b: 0 for b in BUCKETS}


def record_length(tok):
    for b in BUCKETS:
        if tok < b:
            _hist[b] += 1
            break


def fmt_hist():
    labels = {500: "<500", 1000: "<1k", 2000: "<2k", 4000: "<4k", 8000: "<8k",
              16000: "<16k", 32000: "<32k", 64000: "<64k", float("inf"): ">=64k"}
    return "  ".join(f"{labels[b]}={_hist[b]}" for b in BUCKETS if _hist[b])


# Most requests are short (128/512); a few are huge (up to ~100k) to stress the KV cache.
MAXTOK_CHOICES = [128, 512, 1024, 4096, 100000]
MAXTOK_WEIGHTS = [40, 40, 10, 7, 3]


def random_prompt():
    return random.choice(TEMPLATES).format(random.choice(TOPICS))


def random_max_tokens():
    mt = random.choices(MAXTOK_CHOICES, weights=MAXTOK_WEIGHTS, k=1)[0]
    return min(mt, MAX_TOKENS)


def one_request():
    """Stream one completion; count tokens live, return total completion tokens."""
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": random_prompt()}],
        "max_tokens": random_max_tokens(),
        "ignore_eos": False,                              # stop at end token
        "chat_template_kwargs": {"enable_thinking": True},  # thinking on max
        "temperature": 0.7,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    tok = 0
    parts = []
    with requests.post(f"{HOST}/v1/chat/completions", json=body,
                       timeout=3600, stream=True) as r:
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
        for line in r.iter_lines(decode_unicode=True):
            if _stop.is_set():
                break
            if not line or not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("usage"):
                tok = chunk["usage"]["completion_tokens"]
            for ch in chunk.get("choices", []):
                delta = ch.get("delta", {})
                piece = (delta.get("reasoning") or delta.get("reasoning_content")
                         or delta.get("content") or "")
                if piece:
                    with _lock:
                        _stats["tokens_seen"] += 1
                    if PRINT_OUTPUT:
                        parts.append(piece)
    if PRINT_OUTPUT and parts:
        with _lock:
            print("\n" + "=" * 80)
            print("".join(parts))
            print("=" * 80, flush=True)
    return tok


def worker():
    while not _stop.is_set():
        try:
            tok = one_request()
            with _lock:
                _stats["done"] += 1
                _stats["tokens"] += tok
                record_length(tok)
        except Exception as e:  # noqa: BLE001
            with _lock:
                _stats["errors"] += 1
            if not _stop.is_set():
                print(f"  request failed: {e}", file=sys.stderr)
        time.sleep(0.1)


def main():
    print(f"host={HOST}  model={MODEL}  parallel={NUM_PARALLEL}  "
          f"max_tokens={MAX_TOKENS}  (Ctrl-C to stop)")
    t0 = time.perf_counter()
    ex = ThreadPoolExecutor(max_workers=NUM_PARALLEL)
    for _ in range(NUM_PARALLEL):
        ex.submit(worker)
    try:
        while True:
            time.sleep(2)
            wall = time.perf_counter() - t0
            with _lock:
                done, errs, seen = _stats["done"], _stats["errors"], _stats["tokens_seen"]
                avg = _stats["tokens"] / done if done else 0
                hist = fmt_hist()
            print(f"\n[{wall:7.1f}s] done={done:<6} errors={errs:<4} "
                  f"streamed_tok={seen:<9} throughput={seen / wall:8.1f} tok/s  "
                  f"avg_len={avg:6.0f}  [{hist}]",
                  file=sys.stderr, flush=True)
    except KeyboardInterrupt:
        print("\nstopping...", file=sys.stderr)
        _stop.set()
        ex.shutdown(wait=False, cancel_futures=True)
        wall = time.perf_counter() - t0
        avg = _stats["tokens"] / _stats["done"] if _stats["done"] else 0
        print(f"final: done={_stats['done']} errors={_stats['errors']} "
              f"streamed_tok={_stats['tokens_seen']} "
              f"throughput={_stats['tokens_seen'] / wall:.1f} tok/s  "
              f"avg_len={avg:.0f}  [{fmt_hist()}]", file=sys.stderr)


if __name__ == "__main__":
    main()
