#!/usr/bin/env bash
#
# remote_tunnel.sh -- run this on a PUBLIC-FACING PC to expose the HPC GLM
# vLLM server to the outside world. The PC becomes the entry point for requests:
#
#   external clients --> THIS_PC:52223 --(ssh)--> login node --> compute node:52223 (GLM)
#
# It auto-discovers the GLM head-node IP (changes per job), opens an SSH tunnel
# bound to ALL interfaces (0.0.0.0) so external clients can reach it, and keeps
# the tunnel alive (auto-reconnect, re-discovering the IP after a job restart).
#
# Usage (on the public PC):
#     ./remote_tunnel.sh            # open the public gateway (stays running)
#     ./remote_tunnel.sh --print    # only show the head IP + ssh command
#
# Clients anywhere then use:
#     VLLM_HOST=http://<THIS_PC_PUBLIC_IP>:52223 python query_glm.py "Hello"
#
# !!! SECURITY -- READ THIS !!!
# This publishes the LLM endpoint on the internet WITHOUT authentication. Anyone
# who can reach THIS_PC:52223 can use (and abuse) the model and your HPC compute
# budget. Strongly recommended: restrict the firewall to known source IPs and/or
# start vLLM with --api-key. Check your HPC center's acceptable-use policy first.
# If the PC is behind a router/NAT, you must also port-forward 52223 to the PC.
#
# Config via environment variables (defaults shown):
#     GLM_LOGIN=cu829455@login23-g-1.hpc.itc.rwth-aachen.de
#     GLM_WORKDIR=/hpcwork/cu829455/workspace/runLargeLLMs
#     GLM_JOBNAME=glm52_vllm
#     LISTEN_ADDR=0.0.0.0      # bind address on this PC (0.0.0.0 = reachable from outside)
#     LISTEN_PORT=52223        # public port on this PC
#     REMOTE_PORT=52223        # vLLM port on the compute node
#     OPEN_FIREWALL=0          # 1 = try to open LISTEN_PORT in the local firewall (needs sudo)
#     RECONNECT=1              # 1 = keep reconnecting; 0 = run once
set -uo pipefail

GLM_LOGIN="${GLM_LOGIN:-cu829455@login23-g-1.hpc.itc.rwth-aachen.de}"
GLM_WORKDIR="${GLM_WORKDIR:-/hpcwork/cu829455/workspace/runLargeLLMs}"
GLM_JOBNAME="${GLM_JOBNAME:-glm52_vllm}"
LISTEN_ADDR="${LISTEN_ADDR:-0.0.0.0}"
LISTEN_PORT="${LISTEN_PORT:-52223}"
REMOTE_PORT="${REMOTE_PORT:-52223}"
OPEN_FIREWALL="${OPEN_FIREWALL:-0}"
RECONNECT="${RECONNECT:-1}"

trap 'echo; echo "[remote_tunnel] stopped."; exit 0' INT TERM

# Discovery: find the running job's id, then read the head IP from its log.
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
        # Running on the public PC -> discover via the login node.
        ssh "$GLM_LOGIN" "WORKDIR='$GLM_WORKDIR' JOB='$GLM_JOBNAME' bash -s" <<<"$DISCOVER"
    fi
}

open_firewall() {
    [ "$OPEN_FIREWALL" = "1" ] || return 0
    echo "[remote_tunnel] opening firewall for ${LISTEN_PORT}/tcp (needs sudo)..."
    if command -v ufw >/dev/null 2>&1; then
        sudo ufw allow "${LISTEN_PORT}/tcp" || true
    elif command -v firewall-cmd >/dev/null 2>&1; then
        sudo firewall-cmd --add-port="${LISTEN_PORT}/tcp" || true
    elif command -v iptables >/dev/null 2>&1; then
        sudo iptables -I INPUT -p tcp --dport "${LISTEN_PORT}" -j ACCEPT || true
    else
        echo "  no known firewall tool found; open ${LISTEN_PORT}/tcp manually if needed." >&2
    fi
}

port_busy() {
    lsof -nP -iTCP:"$LISTEN_PORT" -sTCP:LISTEN >/dev/null 2>&1 \
        || ss -tlnH "( sport = :$LISTEN_PORT )" 2>/dev/null | grep -q .
}

# --- print-only mode ---------------------------------------------------------
if [[ "${1:-}" == "--print" ]]; then
    ip=$(get_head_ip) || { echo "No running '${GLM_JOBNAME}' job found." >&2; exit 1; }
    echo "GLM head node: ${ip}:${REMOTE_PORT}"
    echo "ssh -o ExitOnForwardFailure=yes -N -L ${LISTEN_ADDR}:${LISTEN_PORT}:${ip}:${REMOTE_PORT} ${GLM_LOGIN}"
    exit 0
fi

open_firewall

# --- main loop ---------------------------------------------------------------
while true; do
    if port_busy; then
        echo "ERROR: local port ${LISTEN_PORT} is already in use:" >&2
        lsof -nP -iTCP:"$LISTEN_PORT" -sTCP:LISTEN 2>/dev/null \
            || ss -tlnp "( sport = :$LISTEN_PORT )" 2>/dev/null
        echo "Close it (pkill -f 'ssh .*:${LISTEN_PORT}:') or set LISTEN_PORT=<port>." >&2
        exit 1
    fi

    ip=$(get_head_ip) || ip=""
    if [ -z "$ip" ]; then
        echo "[remote_tunnel] no running '${GLM_JOBNAME}' job found." >&2
        [ "$RECONNECT" = "1" ] || exit 1
        echo "[remote_tunnel] retrying in 30s... (Ctrl-C to stop)"
        sleep 30
        continue
    fi

    echo "[remote_tunnel] GLM head node: ${ip}:${REMOTE_PORT}"
    echo "[remote_tunnel] exposing ${LISTEN_ADDR}:${LISTEN_PORT} -> ${ip}:${REMOTE_PORT} via ${GLM_LOGIN}"
    echo "[remote_tunnel] clients use: VLLM_HOST=http://<this-pc>:${LISTEN_PORT} python query_glm.py \"Hello\""
    echo "[remote_tunnel] press Ctrl-C to stop."

    ssh -o ExitOnForwardFailure=yes \
        -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
        -N -L "${LISTEN_ADDR}:${LISTEN_PORT}:${ip}:${REMOTE_PORT}" "$GLM_LOGIN"
    rc=$?

    echo "[remote_tunnel] tunnel closed (exit ${rc})."
    [ "$RECONNECT" = "1" ] || exit "$rc"
    echo "[remote_tunnel] reconnecting in 5s... (Ctrl-C to stop)"
    sleep 5
done
