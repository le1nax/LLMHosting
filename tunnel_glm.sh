#!/usr/bin/env bash
#
# Opens an SSH tunnel from your local machine to the running GLM vLLM server.
#
# The server runs on a compute node whose only IP is on the internal cluster
# network (not reachable from the internet). This script finds that node's IP
# automatically (from the "Head:" line in the job log) and forwards a local
# port to it through the login node.
#
# Usage (on your laptop):
#     ./tunnel_glm.sh                 # open tunnel: localhost:52222 -> GLM server
#     ./tunnel_glm.sh --print         # only print the head IP + ssh command
#
# Then, locally:
#     VLLM_HOST=http://localhost:52222 python query_glm.py "Hello"
#
# Config via environment variables (defaults shown):
#     GLM_LOGIN=cu829455@login23-g-1.hpc.itc.rwth-aachen.de
#     GLM_WORKDIR=/hpcwork/cu829455/workspace/runLargeLLMs
#     GLM_JOBNAME=glm52_vllm
#     GLM_LOCAL_PORT=8000
#     GLM_REMOTE_PORT=8000
set -euo pipefail

GLM_LOGIN="${GLM_LOGIN:-cu829455@login23-g-1.hpc.itc.rwth-aachen.de}"
GLM_WORKDIR="${GLM_WORKDIR:-/hpcwork/cu829455/workspace/runLargeLLMs}"
GLM_JOBNAME="${GLM_JOBNAME:-glm52_vllm}"
GLM_LOCAL_PORT="${GLM_LOCAL_PORT:-52222}"
GLM_REMOTE_PORT="${GLM_REMOTE_PORT:-8000}"

# Discovery logic: find the running job's id, then read the head IP from its log.
read -r -d '' DISCOVER <<'REMOTE' || true
cd "$WORKDIR" 2>/dev/null || { echo "NO_WORKDIR" >&2; exit 2; }
jid=$(squeue -u "$USER" -h -n "$JOB" -t RUNNING -o %i | head -1)
[ -z "$jid" ] && { echo "NO_RUNNING_JOB" >&2; exit 3; }
grep -h '^Head:' "log.$jid.txt" 2>/dev/null | grep -oE '([0-9]+[.]){3}[0-9]+' | head -1
REMOTE

get_head_ip() {
    if command -v squeue >/dev/null 2>&1; then
        # Running on the cluster itself.
        WORKDIR="$GLM_WORKDIR" JOB="$GLM_JOBNAME" bash -c "$DISCOVER"
    else
        # Running on a remote machine -> discover via the login node.
        ssh "$GLM_LOGIN" "WORKDIR='$GLM_WORKDIR' JOB='$GLM_JOBNAME' bash -s" <<<"$DISCOVER"
    fi
}

ip=$(get_head_ip) || {
    echo "Could not determine the GLM head-node IP (is the job '$GLM_JOBNAME' running?)." >&2
    exit 1
}

echo "GLM server head node: $ip:$GLM_REMOTE_PORT"

if [[ "${1:-}" == "--print" ]]; then
    echo "Run this on your local machine:"
    echo "    ssh -N -L ${GLM_LOCAL_PORT}:${ip}:${GLM_REMOTE_PORT} ${GLM_LOGIN}"
    exit 0
fi

echo "Forwarding localhost:${GLM_LOCAL_PORT} -> ${ip}:${GLM_REMOTE_PORT} via ${GLM_LOGIN}"
echo "Use it locally with:  VLLM_HOST=http://localhost:${GLM_LOCAL_PORT} python query_glm.py \"Hello\""
echo "Press Ctrl-C to close the tunnel."
exec ssh -N -L "${GLM_LOCAL_PORT}:${ip}:${GLM_REMOTE_PORT}" "${GLM_LOGIN}"
