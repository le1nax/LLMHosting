#!/usr/bin/env python3
"""Throughput benchmark for the vLLM OpenAI endpoint (GLM-5.2).

Measures output-token throughput at one or more concurrency levels by firing
requests in parallel and dividing total generated tokens by wall-clock time.

Each request generates exactly --max-tokens tokens (ignore_eos) with thinking
disabled, so the numbers are stable and comparable.

Examples:
    python benchmark_glm.py                      # single run, 8 parallel
    python benchmark_glm.py -c 16 -n 64 -t 256   # 64 requests, 16 parallel, 256 tokens each
    python benchmark_glm.py --sweep 1,4,16,64    # compare several concurrency levels
"""
import argparse
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

HOST = os.environ.get("VLLM_HOST", "http://134.61.53.224:52223")   # head-node IP (see "Head:" in the log)
MODEL = os.environ.get("VLLM_MODEL", "glm-5.2-fp8")
PROMPT = os.environ.get("BENCH_PROMPT", "Write a long, detailed essay about the history of computing.")


def one_request(max_tokens):
    """Send one request; return (latency_s, completion_tokens, prompt_tokens)."""
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "ignore_eos": True,                              # force exactly max_tokens
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0.7,
    }
    t0 = time.perf_counter()
    r = requests.post(f"{HOST}/v1/chat/completions", json=body, timeout=3600)
    dt = time.perf_counter() - t0
    r.raise_for_status()
    u = r.json()["usage"]
    return dt, u["completion_tokens"], u["prompt_tokens"]


def run(concurrency, n_requests, max_tokens):
    results, errors = [], 0
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(one_request, max_tokens) for _ in range(n_requests)]
        for f in futs:
            try:
                results.append(f.result())
            except Exception as e:  # noqa: BLE001
                errors += 1
                print(f"  request failed: {e}", file=sys.stderr)
    wall = time.perf_counter() - t0
    if not results:
        return None
    lat = sorted(r[0] for r in results)
    out_tok = sum(r[1] for r in results)
    return {
        "concurrency": concurrency,
        "requests": len(results),
        "errors": errors,
        "wall": wall,
        "out_tok": out_tok,
        "out_tps": out_tok / wall,
        "req_s": len(results) / wall,
        "lat_mean": statistics.mean(lat),
        "lat_p50": lat[len(lat) // 2],
        "lat_p95": lat[min(len(lat) - 1, int(len(lat) * 0.95))],
        "per_req_tps": statistics.mean([r[1] / r[0] for r in results]),
    }


def fmt(s):
    line = (f"c={s['concurrency']:<4} reqs={s['requests']:<4} out_tok={s['out_tok']:<7} "
            f"wall={s['wall']:6.1f}s  THROUGHPUT {s['out_tps']:8.1f} tok/s  "
            f"req/s {s['req_s']:5.2f}  lat mean/p50/p95 "
            f"{s['lat_mean']:5.1f}/{s['lat_p50']:5.1f}/{s['lat_p95']:5.1f}s  "
            f"per-req {s['per_req_tps']:5.1f} tok/s")
    if s["errors"]:
        line += f"  ERRORS={s['errors']}"
    return line


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--concurrency", type=int, default=8, help="parallel requests (default 8)")
    ap.add_argument("-n", "--requests", type=int, default=0, help="total requests (default: 4x concurrency)")
    ap.add_argument("-t", "--max-tokens", type=int, default=256, help="output tokens per request (default 256)")
    ap.add_argument("--sweep", type=str, default="", help="comma-separated concurrency levels, e.g. 1,4,16,64")
    args = ap.parse_args()

    print(f"host={HOST}  model={MODEL}  max_tokens={args.max_tokens}")

    # Warm up / fail fast if the server is unreachable.
    try:
        one_request(8)
    except Exception as e:  # noqa: BLE001
        print(f"cannot reach server: {e}", file=sys.stderr)
        sys.exit(1)

    levels = [int(x) for x in args.sweep.split(",")] if args.sweep else [args.concurrency]
    for c in levels:
        n = args.requests or c * 4
        s = run(c, n, args.max_tokens)
        if s:
            print(fmt(s))


if __name__ == "__main__":
    main()
