#!/usr/bin/env python3
"""Minimal terminal chat client for the vLLM OpenAI endpoint (GLM-5.2).

Streams the model's reply live, keeps the conversation history, and shows the
thinking (reasoning) in a dimmed color before the actual answer.

    python chat_glm.py                       # thinking on
    THINK=0 python chat_glm.py               # thinking off

Commands inside the chat:
    /reset   clear the conversation history
    /think   toggle thinking on/off
    /exit    quit (or Ctrl-C / Ctrl-D)
"""
import json
import os
import sys

import requests

HOST = os.environ.get("VLLM_HOST", "http://134.61.53.224:52223")   # head-node IP
MODEL = os.environ.get("VLLM_MODEL", "glm-5.2-fp8")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "32000"))
THINK = os.environ.get("THINK", "1").lower() in ("1", "true", "yes")

DIM, CYAN, BOLD, RESET = "\033[2m", "\033[36m", "\033[1m", "\033[0m"
SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", "You are a helpful assistant.")


def stream_reply(messages, think):
    """Stream one assistant reply; print it live; return the answer text."""
    body = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": think},
    }
    r = requests.post(f"{HOST}/v1/chat/completions", json=body, timeout=3600, stream=True)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")

    answer, in_think, started = [], False, False
    with r:
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            for ch in json.loads(payload).get("choices", []):
                delta = ch.get("delta", {})
                think_piece = delta.get("reasoning") or delta.get("reasoning_content")
                content = delta.get("content")
                if think_piece:
                    if not in_think:
                        sys.stdout.write(f"{DIM}[thinking] ")
                        in_think = True
                    sys.stdout.write(think_piece)
                    sys.stdout.flush()
                if content:
                    if in_think:
                        sys.stdout.write(f"{RESET}\n\n")
                        in_think = False
                    if not started:
                        sys.stdout.write(f"{CYAN}")
                        started = True
                    answer.append(content)
                    sys.stdout.write(content)
                    sys.stdout.flush()
    if in_think:
        sys.stdout.write(RESET)
    sys.stdout.write(f"{RESET}\n")
    sys.stdout.flush()
    return "".join(answer)


def main():
    think = THINK
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print(f"{BOLD}Chat with {MODEL}{RESET}  (host={HOST}, thinking={'on' if think else 'off'})")
    print("Commands: /reset  /think  /exit\n")

    while True:
        try:
            user = input(f"{BOLD}you>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break

        if not user:
            continue
        if user in ("/exit", "/quit"):
            print("bye")
            break
        if user == "/reset":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("(history cleared)\n")
            continue
        if user == "/think":
            think = not think
            print(f"(thinking {'on' if think else 'off'})\n")
            continue

        messages.append({"role": "user", "content": user})
        sys.stdout.write(f"{BOLD}bot>{RESET} ")
        sys.stdout.flush()
        try:
            answer = stream_reply(messages, think)
        except KeyboardInterrupt:
            print(f"{RESET}\n(interrupted)\n")
            messages.pop()                       # drop the unanswered turn
            continue
        except Exception as e:  # noqa: BLE001
            print(f"{RESET}\n  request failed: {e}\n", file=sys.stderr)
            messages.pop()
            continue
        messages.append({"role": "assistant", "content": answer})
        print()


if __name__ == "__main__":
    main()
