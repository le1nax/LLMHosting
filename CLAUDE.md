# CLAUDE.md

Guidance for Claude Code (or any agent) working in this repo.

## What this project is

Scripts to self-host large open-weight LLMs (currently GLM-5.2-FP8 and
Kimi-K3) as OpenAI-compatible APIs via **vLLM + Ray**, launched as multi-node
**SLURM** batch jobs, so they can be queried from other pipelines (e.g. label
extraction over medical reports) instead of paying for a hosted API.

Core pattern used by every `run*.sbatch` script:
1. Reserve N nodes exclusively via `sbatch --nodes=N`.
2. Start a Ray cluster across those nodes (head + workers via `srun`).
3. Launch `vllm serve <model>` on the head node with tensor/data/expert
   parallelism sized to the node count, serving on a fixed port.
4. A background watchdog polls `/v1/models` until vLLM is ready, and
   kills+restarts the job (bounded retries) if it never comes up.
5. Model weights are copied off the shared network filesystem onto
   node-local storage before serving starts (see "Lessons learned" below for
   why this is load-bearing, not an optimization).

## Origin: built for RWTH Aachen HPC (`truhnlab` partition)

This repo was originally developed and tested on RWTH Aachen's cluster,
partition `truhnlab` (9 nodes × 4× NVIDIA Hopper/H100 GPUs, AMD EPYC/x86_64
CPUs, 30-day max walltime, NFS-backed `$HPCWORK` shared filesystem).

**Status per script (as of the JUPITER migration):**
- `runGLMStable.sbatch` — **stable**, the production path for GLM-5.2-FP8.
  Reached this state after several rounds of debugging (see RESEARCHLOG.md).
  4 nodes × 4 GPUs, TP/PP sized accordingly, venv copied to node-local NVMe.
- `runGLMTool.sbatch`, `_runGLM.sbatch`, `runGLMTooldebug.sbatch` — earlier/
  variant GLM launchers, superseded by `runGLMStable.sbatch` for real runs.
- `runQWEN.sbatch` — Qwen launcher, less exercised than the GLM path.
- `runKimiK3.sbatch` — **first draft, not yet successfully run.** Targets
  Kimi-K3 (2.8T params, MXFP4, 896 experts/16 active) across 8 nodes × 4 GPUs
  (32 GPUs, TP=4 × DP=8 ⇒ EP=32), served from a vLLM `apptainer` container
  (needed vLLM 0.27.0+ day-0 Kimi-K3 support, not on PyPI yet) with model
  weights (~1.56 TB) aggregated across node-local NVMe via `--beeond`
  (BeeGFS-on-demand) since no single node's local disk holds the full model.
  **Last real run (job 3015142, 2026-08-17) failed after ~55s**: `$BEEOND`
  was set as an env var but `/mnt/beegfs` was never actually mounted at that
  path — `df -h "$BEEOND"` resolved to the node's root filesystem (~31G
  free), so the weight copy hit `No space left on device` almost
  immediately. The script's own `[[ -z "${BEEOND:-}" ]]` check only verifies
  the variable is *set*, not that it's a real mount — needs a
  `mount | grep beegfs`-style check before the copy starts. This bug is
  **not yet fixed** in the script.
- `runGLMOnHeldNodes.sh`, `holdGLMNodes.sbatch` — workaround for iterating
  quickly during debugging: hold a node allocation open with a placeholder
  job (`sleep infinity`) so repeated launch attempts don't each pay SLURM
  queue-wait time.
- `runKimiK3Test.sh`, `benchmark_glm.py`, `stress_glm.py`, `chat_glm.py`,
  `query_glm*.py`, `monitorGLM.sh` — client-side test/benchmark/chat/monitor
  tooling against the served OpenAI-compatible endpoint.
- `GLM_ZUGANG.md` — step-by-step SSH-tunnel instructions for reaching the
  served model from a downstream pipeline machine. RWTH-specific (hostnames,
  login node), will need a JUPITER equivalent.
- `RESEARCHLOG.md` — append-only log of non-obvious bugs and root causes.
  **Read this before debugging a hang or crash** — several failure modes
  here look environmental/random but have a specific, documented cause.

## Migration target: JUPITER (Jülich Supercomputing Centre)

Moving because [reason: capacity/allocation — fill in]. JUPITER is
Europe's first exascale system; docs: https://apps.fz-juelich.de/jsc/hps/jupiter/index.html

**This is not a drop-in port. Read this section before adapting any script.**

### 1. CPU architecture changes: x86_64 → ARM64 (top blocker)

JUPITER Booster nodes use **NVIDIA Grace-Hopper Superchips** — the CPU is
NVIDIA **Grace (ARM64/aarch64)**, not x86_64 like RWTH's AMD EPYC nodes. This
means:
- The existing Kimi-K3 Apptainer image (`vllm/vllm-openai:kimi-k3`, an
  upstream x86_64 build) **will not run** on JUPITER as-is — need an
  aarch64-compatible image/build.
- The `.venv` / `.venv-python313` environments (uv-managed CPython +
  compiled wheels, notably `nvidia-nccl` etc.) are x86_64 builds and
  **will not run** on JUPITER either. Rebuild venvs from scratch on a
  JUPITER login/compute node, don't copy them over.
- Any hand-picked wheel URLs or pinned binary dependencies in
  `requirements.txt` need an aarch64 availability check.

### 2. SLURM differences

| | RWTH `truhnlab` | JUPITER `booster` |
|---|---|---|
| Partition | `truhnlab` | `booster` (only partition, Early Access phase) |
| GPUs/node | 4× Hopper/H100 | 4× GH200 (Grace Hopper) |
| CPU | AMD EPYC, 128 cores/node | NVIDIA Grace, 288 cores/node |
| Mem/node | 745 GiB | 480 GiB |
| Max walltime | 30 days | **12 hours** (default 1h) |
| Account flag | `--account=truhnlab` | `--account=<budget>` (Jülich compute-time budget code — TODO: fill in) |
| MPI launch | `srun` (already used here) | `srun` only — **`mpiexec` is not supported** |
| Node-count granularity | 1 node | 1 node minimum, whole system max |

**The 12h walltime cap is the second major blocker for this project.** Every
`run*.sbatch` script here is written as a long-lived server (watchdog loops,
restart-on-failure, meant to stay up for a labeling run that takes hours).
On JUPITER, a job is killed at 12h regardless. Options to design around
this, not yet decided:
- Accept 12h serving windows and have downstream clients (e.g.
  `GLM_ZUGANG.md`'s tunnel workflow) handle reconnect-with-new-IP across job
  boundaries (the resume-safe checkpointing already used by the labeling
  pipeline helps here).
- Chain jobs with `sbatch --dependency=afterany:<jobid>` to auto-resubmit.
- Ask JSC support about longer-running QoS/reservations for this use case.

### 3. Module system & filesystems

- Modules: EasyBuild, staged by year (`Stages/<year>`), not whatever module
  system RWTH uses — `module load` commands in scripts need re-checking.
- Env vars replacing RWTH's `$HPCWORK`: `$HOME`, `$PROJECT` (project data),
  `$SCRATCH` (working/scratch data), `$DATA` / `$ARCHIVE` (data projects).
  Everywhere this repo currently hardcodes `/rwthfs/rz/cluster/hpcwork/...`
  or reads `$HPCWORK`, swap for the equivalent JUPITER path.
- Apptainer is supported (backwards-compatible with Singularity) — good,
  the Kimi-K3 container approach carries over conceptually, just needs an
  aarch64 image.
- BeeOND (`--beeond`) support on JUPITER is unconfirmed — verify before
  reusing `runKimiK3.sbatch`'s approach, and fix the mount-verification bug
  above regardless of filesystem.

### 4. Access

More than 20 login nodes provide SSH access to JUPITER's system modules.
RWTH-specific access docs (`GLM_ZUGANG.md`) — hostnames, tunnel commands,
host-key fingerprints — need a JUPITER equivalent once login node
hostnames/IPs are known.

## Lessons learned (see RESEARCHLOG.md for full detail)

These are cluster-environment failure modes, not application bugs — worth
re-verifying they still apply (or don't) on JUPITER's different filesystem:

- **SIGBUS crashes from NFS-backed mmap'd `.venv`** (job 1881487): a venv
  living on shared NFS gets demand-paged; if a page-in transiently fails
  (NFS hiccup), the kernel delivers SIGBUS to the process instead of
  retrying, killing a Ray worker and taking down the whole vLLM server.
  Fix: copy the venv to node-local NVMe (`$TMPDIR`) at job start on every
  node. This is why `runGLMStable.sbatch` does that copy — it's not
  optional. On JUPITER, re-evaluate for whatever the equivalent scratch/NFS
  split is.
- **Transient DeepGEMM JIT-compile failures under parallel load**
  (job 1881087): CUDA header reads from a shared/cached filesystem
  (cvmfs on RWTH) can intermittently fail under many concurrent JIT
  compiles, killing workers. vLLM hangs instead of failing fast — hence the
  startup watchdog pattern used throughout this repo's scripts.
- **`--wait=0` required on every `srun` call in multi-node steps** (found in
  job 2509534): this cluster's SLURM has a ~5s default `WaitTime` — once one
  task in an `--ntasks-per-node=1` step finishes (success or failure), srun
  kills the still-running tasks on other nodes after 5s, even if they're
  just running a bit slower (e.g. normal NFS copy-time variance). Likely
  explains earlier "whole step dies because of one node" observations that
  were misdiagnosed as a resource problem.

## Working conventions

- Comments and some docs in this repo are German; code/identifiers are
  English. Match existing style per-file rather than translating wholesale.
- `RESEARCHLOG.md` is append-only — add new entries for surprising bugs, not
  edits to old ones, so the debugging history stays intact.
- Prefer node-local storage for anything mmap'd or hot-read
  (venvs/interpreters, container images) — see "Lessons learned" — and
  treat shared network filesystems as fine for large sequential reads
  (model weight downloads) but risky for repeated small random access under
  parallel load.
