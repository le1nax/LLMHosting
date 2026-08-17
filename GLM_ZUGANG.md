# GLM-Zugang & Label-Extraktion — Anleitung (from scratch)

How to reach the GLM-5.2 vLLM server on the RWTH HPC from `hawking` and run
label extraction. Follow top to bottom. Copy-paste the commands.

---

## 0. Das Bild (topology) — WICHTIG

Three different machines. The tunnel must live on **hawking**, because that's
where the pipeline, the data, and the conda env are.

```
  your laptop            hawking                 RWTH login node          GLM compute node
(dgeiger-Nitro)   (radcluster, where       (login23-1.hpc...)         (i25g0003)
                    Claude + pipeline run)                             134.61.53.224:52223
      │                    │                        │                        │
      │  ssh (you do this) │                        │                        │
      └───────────────────►│                        │                        │
                           │   ssh -L tunnel        │      forwards to       │
                           └───────────────────────►│───────────────────────►│
                              localhost:52223  ─────────────────────────►  GLM API
```

**Rule of thumb:** whatever runs the `python main.py` must be able to reach
`localhost:52223`. That is hawking. So the tunnel is opened **on hawking**.
(Opening it on the laptop = useless, that was our first mistake.)

---

## 1. Voraussetzungen (prerequisites)

- The GLM vLLM job is **running on the HPC** (your "Claude on HPC" starts it via
  sbatch). You need its **compute-node IP** and **port**.
  - Current: node `i25g0003` = `134.61.53.224`, port `52223`.
  - ⚠️ The IP **changes every time the job restarts**. Get the fresh one from the
    job log (`Head:` line) or `squeue` on the HPC.
- Your RWTH HPC login: user `cs088267`, password + **2-factor code**.
- One-time: the login-node host key must be trusted on hawking (see step 3, first run).

---

## 2. Auf hawking einloggen (real terminal!)

From your laptop, open a normal terminal and SSH into hawking **the way you
normally do**:

```bash
ssh dgeiger@hawking
```

> Why a real terminal and not Claude's `!` prompt? Because the next step needs
> you to type a **password + 2FA**, and the `!` prompt has no terminal for that.

---

## 3. Den Tunnel öffnen (on hawking)

```bash
ssh -N -f -o ServerAliveInterval=60 -o ExitOnForwardFailure=yes \
    -L 52223:134.61.53.224:52223 cs088267@login23-1.hpc.itc.rwth-aachen.de
```

- `-L 52223:134.61.53.224:52223` → forward hawking's `localhost:52223` to the GLM node.
- `-N` no shell, `-f` go to background **after** you authenticate.
- Enter **password**, then **2-factor code**.

**First time only:** you'll see `Host key verification failed` or a fingerprint
prompt. Verify the fingerprint against RWTH's published keys, then accept. The
trusted keys for `login23-1` are:
```
ED25519  SHA256:YIuc1gYlVbtRUbi6grM+PWEO4syDQj3Yc7nTHJJiKpc
RSA      SHA256:cwztAW+xPte9zdi1yaw8Vr+B8O34YpZg5qpw0YolJpY
```

Check the tunnel is listening:
```bash
ss -tlnp | grep 52223      # expect: LISTEN ... 127.0.0.1:52223 ... ("ssh",...)
```

---

## 4. GLM testen (verify)

```bash
curl -s http://localhost:52223/v1/models
```
Expect JSON with `"id":"glm-5.2-fp8"`. If yes → you're in. ✅

Optional real inference test:
```bash
curl -s http://localhost:52223/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.2-fp8","messages":[{"role":"user","content":"Say OK"}],
       "max_tokens":8,"chat_template_kwargs":{"enable_thinking":false}}'
```

---

## 5. Labeling starten (run extraction)

```bash
cd /home/homesOnMaster/dgeiger/repos/label_extraction
source /home/homesOnMaster/dgeiger/miniforge3/etc/profile.d/conda.sh && conda activate vllm

# graded run: metastasis / cholestasis / cirrhosis on scale  -, +, ++, +++
python main.py \
  --input  data/sampixel_reports.yaml \
  --output data/labels_glm_graded.yaml \
  --url    http://localhost:52223/v1 \
  --model  glm-5.2-fp8 \
  --graded --reasoning off --concurrency 16
```

- `--graded` → ordinal grades instead of true/false. Which labels + the rubric
  live in `config.py` (`GRADED_LABELS`, `GRADE_RUBRIC`).
- `--reasoning off` → GLM answers directly, no thinking preamble.
- **Resume-safe:** checkpoints every 50; re-running the same command skips IDs
  already in the output file. So a dropped tunnel is not fatal — reconnect, re-run.
- ~3 h for the full 3,946 reports at concurrency 16.

Run it detached so it survives logout:
```bash
nohup python main.py ... > data/labels_glm_graded.log 2>&1 &
```

Watch progress:
```bash
grep -ac '200 OK' data/labels_glm_graded.log      # completed requests
grep -c '^- id:' data/labels_glm_graded.yaml       # results written (every 50)
```

---

## 6. Troubleshooting (die Fehler die wir hatten)

| Symptom | Ursache | Fix |
|---|---|---|
| `bind [127.0.0.1]:52223: Address already in use` | tunnel already open (or opened on wrong machine) | reuse it, or `pkill -f 'ssh.*52223'` and reopen — **on hawking** |
| `Host key verification failed` | login-node key not trusted yet | accept fingerprint (step 3) in a real terminal, or `ssh-keyscan` it into `~/.ssh/known_hosts` |
| `Permission denied (publickey,keyboard-interactive)` via `!` | Claude's `!` has no terminal for password/2FA | open the tunnel in a **real terminal**, not the `!` prompt |
| `curl localhost:52223` empty / `connection refused` | tunnel not up on **this** machine | check `ss -tlnp \| grep 52223` on hawking; you may have tunneled on the laptop |
| curl to `134.61.53.224:52223` times out | compute node port is firewalled from hawking | that's expected — you must go through the tunnel, not direct |
| pipeline errors after a while | GLM job ended or IP changed | get new node IP, reopen tunnel with new `-L ...`, re-run (resumes) |

---

## 7. Aufräumen (teardown)

```bash
pkill -f 'ssh.*52223'      # close the tunnel on hawking
```

---

## Spickzettel (cheat sheet)

```bash
# 1. on hawking, open tunnel  (real terminal, enter pw+2FA)
ssh -N -f -o ServerAliveInterval=60 -o ExitOnForwardFailure=yes \
    -L 52223:<GLM_NODE_IP>:52223 cs088267@login23-1.hpc.itc.rwth-aachen.de
# 2. verify
curl -s http://localhost:52223/v1/models
# 3. run
cd /home/homesOnMaster/dgeiger/repos/label_extraction && conda activate vllm
python main.py --input data/sampixel_reports.yaml --output data/labels_glm_graded.yaml \
  --url http://localhost:52223/v1 --model glm-5.2-fp8 --graded --reasoning off --concurrency 16
```
Only `<GLM_NODE_IP>` changes between sessions.
