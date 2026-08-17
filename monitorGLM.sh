#!/usr/bin/zsh
# monitorGLM.sh -- Haenge-Detektor fuer die laufende vLLM/GLM-Deployment.
#
# Hintergrund: Am 2026-07-15 hing der Server ~5 Minuten lang komplett still
# (kein Fehler, kein Crash, kein Core-Dump), bevor vLLMs eigener interner
# RPC-Timeout (multiproc_executor/shm_broadcast) ihn beendete. Die
# Ray-Session-Logs waren zu diesem Zeitpunkt bereits nutzlos: alle 16 Worker-
# Prozesse liefen (Ray-Heartbeat unauffaellig), aber KEINE Spur davon, WAS
# genau haengt -- weil vLLMs eigentliche Ausfuehrung ueber Shared-Memory
# laeuft, nicht ueber Ray-RPCs, und Ray das schlicht nicht sieht.
#
# Dieses Skript ueberwacht das Log auf Stillstand und macht BEIM ERSTEN
# Anzeichen (default 60s ohne neue Zeile, weit vor vLLMs eigenen ~5 min)
# sofort einen Snapshot von allen 4 Nodes:
#   - py-spy dump (mit --locals) von jedem Ray/vLLM-Worker-Prozess
#     -> zeigt den exakten haengenden Python-Frame + native Traceback
#   - nvidia-smi -q (voller GPU-Zustand: ECC, Clocks, Throttle-Gruende,
#     laufende Prozesse) -- Xid-Codes/dmesg sind auf diesem Cluster als
#     Nicht-Root-User NICHT lesbar (kernel.dmesg_restrict), ECC-Zaehler
#     sind der naechstbeste verfuegbare Hardware-Fehler-Indikator.
#
# Aufruf: ./monitorGLM.sh <HELD_JOBID> <LOGFILE> [STALL_THRESHOLD_S=60] [POLL_S=15]
# Laeuft endlos im Hintergrund; ueberlebt mehrere Versuche/Neustarts der
# eigentlichen Deployment-Schleife, solange dasselbe Logfile weiterwaechst.

set -u

HELD_JOBID=${1:?"Usage: $0 <HELD_JOBID> <LOGFILE> [STALL_THRESHOLD_S] [POLL_S]"}
LOGFILE=${2:?"Usage: $0 <HELD_JOBID> <LOGFILE> [STALL_THRESHOLD_S] [POLL_S]"}
STALL_THRESHOLD=${3:-60}
POLL_INTERVAL=${4:-15}

PROJECT_DIR=$(cd "$(dirname "$0")" && pwd)
DIAG_DIR="$PROJECT_DIR/diagnostics"
mkdir -p "$DIAG_DIR"

echo "[monitor] $(date '+%F %T') Start: Job=$HELD_JOBID Log=$LOGFILE Schwelle=${STALL_THRESHOLD}s Poll=${POLL_INTERVAL}s"

capture_snapshot() {
    local ts=$(date +%Y%m%d_%H%M%S)
    local outdir="$DIAG_DIR/hang_${ts}"
    mkdir -p "$outdir"
    echo "[monitor] $(date '+%F %T') Log seit >= ${STALL_THRESHOLD}s still -- Snapshot nach $outdir"

    local job_info=$(scontrol show job "$HELD_JOBID" -o 2>/dev/null)
    local nodelist=$(echo "$job_info" | grep -oP '(?<![A-Za-z])NodeList=\K\S+')
    if [[ -z "$nodelist" ]]; then
        echo "[monitor] WARNUNG: konnte NodeList von Job $HELD_JOBID nicht ermitteln, ueberspringe Snapshot." >&2
        return
    fi
    local nodes=(${(f)"$(scontrol show hostnames "$nodelist")"})

    for node in $nodes; do
        (
            pids=$(srun --jobid="$HELD_JOBID" --overlap --nodes=1 --ntasks=1 -w "$node" \
                pgrep -f 'ray::RayWorkerProc|EngineCore|vllm serve|APIServer' 2>/dev/null)
            for pid in ${(f)pids}; do
                [[ -z "$pid" ]] && continue
                srun --jobid="$HELD_JOBID" --overlap --nodes=1 --ntasks=1 -w "$node" \
                    "$PROJECT_DIR/.venv/bin/py-spy" dump --pid "$pid" --locals \
                    > "$outdir/${node}_pyspy_${pid}.txt" 2>&1
            done
            srun --jobid="$HELD_JOBID" --overlap --nodes=1 --ntasks=1 -w "$node" \
                nvidia-smi -q > "$outdir/${node}_nvidia-smi.txt" 2>&1
            echo "[monitor] $(date '+%F %T') [$node] Snapshot fertig (PIDs: ${(j:,:)${(f)pids}})"
        ) &
    done
    wait
    echo "[monitor] $(date '+%F %T') Snapshot komplett: $outdir"
}

last_size=-1
stalled_since=0
already_captured=0

while true; do
    if [[ -f "$LOGFILE" ]]; then
        cur_size=$(stat -c %s "$LOGFILE" 2>/dev/null || echo -1)
        if [[ "$cur_size" == "$last_size" ]]; then
            stalled_since=$(( stalled_since + POLL_INTERVAL ))
        else
            if (( stalled_since >= STALL_THRESHOLD )); then
                echo "[monitor] $(date '+%F %T') Log waechst wieder -- Stillstand beendet."
            fi
            stalled_since=0
            already_captured=0
        fi
        last_size=$cur_size

        if (( stalled_since >= STALL_THRESHOLD && already_captured == 0 )); then
            capture_snapshot
            already_captured=1
        fi
    fi
    sleep "$POLL_INTERVAL"
done
