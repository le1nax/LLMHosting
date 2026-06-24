#!/usr/bin/env python3
"""Minimal client for the vLLM OpenAI endpoint (GLM-5.2).

Run on the SLURM job's head node (vLLM listens there on port 8000):
    python query_glm.py "Your question here"
"""
import os
import sys
import requests

HOST = os.environ.get("VLLM_HOST", "http://134.61.53.224:8000")   # head-node IP (changes per job: see "Head:" in the log)
MODEL = os.environ.get("VLLM_MODEL", "glm-5.2-fp8")               # = --served-model-name in the sbatch script

# Reasoning effort for GLM-5.2: "off" = no thinking (direct answer),
# "high" = medium effort, "max" = full effort.
REASONING = os.environ.get("REASONING", "off")
_REASONING_KWARGS = {
    "off":  {"enable_thinking": False},
    "high": {"reasoning_effort": "high"},
    "max":  {"reasoning_effort": "max"},
}[REASONING]

prompt = " ".join(sys.argv[1:]) or "Hello, who are you? Answer short."

resp = requests.post(
    f"{HOST}/v1/chat/completions",
    json={
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
        "chat_template_kwargs": _REASONING_KWARGS,
    },
    timeout=600,
)
resp.raise_for_status()
print(resp.json()["choices"][0]["message"]["content"])
