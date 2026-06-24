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
#     GLM_LOCAL_PORT=52222
#     GLM_REMOTE_PORT=52223
set -euo pipefail

GLM_LOGIN="${GLM_LOGIN:-cu829455@login23-g-1.hpc.itc.rwth-aachen.de}"
GLM_WORKDIR="${GLM_WORKDIR:-/hpcwork/cu829455/workspace/runLargeLLMs}"
GLM_JOBNAME="${GLM_JOBNAME:-glm52_vllm}"
GLM_LOCAL_PORT="${GLM_LOCAL_PORT:-52222}"
GLM_REMOTE_PORT="${GLM_REMOTE_PORT:-52223}"

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

# Run a command on the cluster (locally if on a cluster node, else via the login node).
run_there() {
    if command -v squeue >/dev/null 2>&1; then bash -c "$1"; else ssh "$GLM_LOGIN" "$1"; fi
}

# True if vLLM actually answers on ip:GLM_REMOTE_PORT (SLURM RUNNING != server ready).
check_ready() {
    local code
    code=$(run_there "curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://$1:${GLM_REMOTE_PORT}/v1/models" 2>/dev/null)
    [ "$code" = "200" ]
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

# Bail out early with a clear message if the local port is already taken
# (usually a previous tunnel from this script is still running).
if lsof -nP -iTCP:"$GLM_LOCAL_PORT" -sTCP:LISTEN >/dev/null 2>&1 \
   || ss -tlnH "( sport = :$GLM_LOCAL_PORT )" 2>/dev/null | grep -q .; then
    echo "ERROR: local port ${GLM_LOCAL_PORT} is already in use:" >&2
    lsof -nP -iTCP:"$GLM_LOCAL_PORT" -sTCP:LISTEN 2>/dev/null \
        || ss -tlnp "( sport = :$GLM_LOCAL_PORT )" 2>/dev/null
    echo "Probably an older tunnel. Close it (e.g. pkill -f 'ssh -N -L ${GLM_LOCAL_PORT}')" >&2
    echo "or pick another port:  GLM_LOCAL_PORT=<port> $0" >&2
    exit 1
fi

# SLURM RUNNING != vLLM ready: wait until the server actually answers (model load ~20 min).
if ! check_ready "$ip"; then
    echo "vLLM not serving yet on ${ip}:${GLM_REMOTE_PORT} (still loading). Waiting..."
    for _ in $(seq 1 180); do          # up to ~30 min
        sleep 10
        if check_ready "$ip"; then break; fi
        printf '.'
    done
    echo
    check_ready "$ip" || { echo "Timed out waiting for the server." >&2; exit 1; }
fi
echo "Server is up."

echo "Forwarding localhost:${GLM_LOCAL_PORT} -> ${ip}:${GLM_REMOTE_PORT} via ${GLM_LOGIN}"
echo "Use it locally with:  VLLM_HOST=http://localhost:${GLM_LOCAL_PORT} python query_glm.py \"Hello\""
echo "Press Ctrl-C to close the tunnel."
exec ssh -o ExitOnForwardFailure=yes -N -L "${GLM_LOCAL_PORT}:${ip}:${GLM_REMOTE_PORT}" "${GLM_LOGIN}"
