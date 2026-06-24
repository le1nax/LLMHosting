#!/usr/bin/env python3
"""Minimaler Client fuer den vLLM-OpenAI-Endpoint (GLM-5.2).

Auf dem Head-Node des SLURM-Jobs ausfuehren (dort lauscht vLLM auf Port 8000):
    python query_glm.py "Deine Frage hier"
"""
import os
import sys
import requests

HOST = os.environ.get("VLLM_HOST", "http://localhost:8000")   # ggf. Head-Node-IP statt localhost
MODEL = os.environ.get("VLLM_MODEL", "glm-5.2-fp8")           # = --served-model-name im sbatch-Skript

prompt = " ".join(sys.argv[1:]) or "Hallo, wer bist du?"

resp = requests.post(
    f"{HOST}/v1/chat/completions",
    json={
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
    },
    timeout=600,
)
resp.raise_for_status()
print(resp.json()["choices"][0]["message"]["content"])
