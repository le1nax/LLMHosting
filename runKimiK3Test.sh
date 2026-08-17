#!/usr/bin/zsh
# runKimiK3Test.sh -- orchestrates one full test cycle for the (untested)
# Kimi-K3 deployment: submit -> wait for queue+startup -> run eval suite ->
# ALWAYS tear down afterward (via trap, even on failure/timeout/Ctrl-C), so
# we never leave 8 nodes / 32 GPUs occupied indefinitely after a test run.
#
# Every step is timestamped and logged (both to stdout, captured by the
# caller's redirect, and to per-test result files) specifically so a failure
# -- expected, given this script has never run on real nodes -- leaves a
# clear trace instead of just "it didn't work."
#
# Aufruf: nohup zsh runKimiK3Test.sh > log.kimik3test.$(date +%Y%m%d_%H%M%S).txt 2>&1 &

set -u   # NOT set -e: a failed test step must still reach the teardown trap
PROJECT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$PROJECT_DIR"

TS_FMT='+%Y-%m-%d %H:%M:%S'
log() { echo "[$(date "$TS_FMT")] $*"; }

RESULTS_DIR="$PROJECT_DIR/kimi_k3_test_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"
log "=== runKimiK3Test.sh starting ==="
log "Results directory: $RESULTS_DIR"

JOBID=""

# --- Garantiertes Teardown: laeuft IMMER beim Skript-Ende, egal ob Erfolg,
# Fehler, Timeout oder externer Kill (trap ... EXIT deckt alle drei ab). ---
teardown() {
    log "=== TEARDOWN ==="
    if [[ -n "$JOBID" ]]; then
        log "Cancelling job $JOBID ..."
        scancel "$JOBID" > "$RESULTS_DIR/teardown.log" 2>&1
        sleep 5
        state=$(squeue -j "$JOBID" -h -o "%T" 2>/dev/null)
        log "Job $JOBID state after scancel: '${state:-<gone from queue>}'"
        if [[ -n "$state" ]]; then
            log "WARNING: job $JOBID still shows state '$state' 5s after scancel -- check manually."
        fi
    else
        log "No job was ever submitted -- nothing to tear down."
    fi
    log "=== TEARDOWN COMPLETE -- results in $RESULTS_DIR ==="
}
trap teardown EXIT

# --- 1. Submit ---
log "=== STEP 1: sbatch runKimiK3.sbatch ==="
SUBMIT_OUT=$(sbatch runKimiK3.sbatch 2>&1)
log "sbatch output: $SUBMIT_OUT"
echo "$SUBMIT_OUT" > "$RESULTS_DIR/sbatch_output.log"
JOBID=$(echo "$SUBMIT_OUT" | grep -oP '(?<=Submitted batch job )\d+')
if [[ -z "$JOBID" ]]; then
    log "ERROR: could not parse job ID from sbatch output -- aborting, nothing to tear down."
    JOBID=""
    exit 1
fi
log "Submitted job $JOBID"

# --- 2. Wait for it to actually start running (queue was down to 2/9 idle
# nodes at submit time, so this can be a long wait -- that's expected). ---
log "=== STEP 2: waiting for job $JOBID to leave the queue and start RUNNING ==="
check_num=0
while true; do
    (( check_num++ ))
    state=$(squeue -j "$JOBID" -h -o "%T" 2>/dev/null)
    reason=$(squeue -j "$JOBID" -h -o "%R" 2>/dev/null)
    log "queue check #$check_num: state='${state:-<not in queue>}' reason='${reason:-}'"
    if [[ "$state" == "RUNNING" ]]; then
        break
    fi
    if [[ -z "$state" ]]; then
        log "ERROR: job $JOBID disappeared from the queue before ever running -- aborting."
        exit 1
    fi
    sleep 60
done
log "Job $JOBID is RUNNING."

DEPLOY_LOG="$PROJECT_DIR/log.kimik3.$JOBID.txt"
log "Deployment log file: $DEPLOY_LOG"

# --- 3. Find the head node IP (printed by runKimiK3.sbatch as 'Head: <node> (<ip>)') ---
log "=== STEP 3: waiting for head node IP in $DEPLOY_LOG ==="
HEAD_IP=""
for i in $(seq 1 30); do
    if [[ -f "$DEPLOY_LOG" ]]; then
        HEAD_IP=$(grep -oP '^Head: \S+ \(\K[^)]+' "$DEPLOY_LOG" 2>/dev/null | head -1)
        [[ -n "$HEAD_IP" ]] && break
    fi
    log "  head IP not yet available (attempt $i/30), waiting..."
    sleep 10
done
if [[ -z "$HEAD_IP" ]]; then
    log "ERROR: could not determine head node IP after 5 min -- aborting."
    cp "$DEPLOY_LOG" "$RESULTS_DIR/deployment_log_at_failure.txt" 2>/dev/null
    exit 1
fi
log "Head node IP: $HEAD_IP"
VLLM_URL="http://$HEAD_IP:52223"

# --- 4. Wait for vLLM to actually be healthy. runKimiK3.sbatch itself retries
# up to 3x internally (MAX_RESTARTS) with a 3h STARTUP_TIMEOUT each -- give
# this loop enough headroom to span that worst case, but bail early the
# moment the deployment log says it gave up for good. ---
log "=== STEP 4: waiting for vLLM readiness at $VLLM_URL (health-checked every 30s) ==="
ready=0
for i in $(seq 1 1200); do   # 1200 * 30s = 10h ceiling, well past 3x3h worst case
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "$VLLM_URL/v1/models" 2>/dev/null)
    if (( i % 10 == 0 || code == 200 )); then
        log "health check #$i: http_code=${code:-<no response>}"
    fi
    if [[ "$code" == "200" ]]; then
        ready=1
        break
    fi
    if grep -q "Maximale Anzahl Neustarts erreicht" "$DEPLOY_LOG" 2>/dev/null; then
        log "ERROR: deployment script exhausted all $((3)) restart attempts -- see $DEPLOY_LOG"
        break
    fi
    if [[ -z "$(squeue -j "$JOBID" -h -o "%T" 2>/dev/null)" ]]; then
        log "ERROR: job $JOBID is no longer in the queue (ended on its own) before becoming ready."
        break
    fi
    sleep 30
done

cp "$DEPLOY_LOG" "$RESULTS_DIR/deployment_log_snapshot.txt" 2>/dev/null

if (( ! ready )); then
    log "ERROR: vLLM never became ready -- skipping test suite, proceeding straight to teardown."
    log "Full deployment log saved to $RESULTS_DIR/deployment_log_snapshot.txt for post-mortem."
    exit 1
fi
log "vLLM is ready at $VLLM_URL."

export VLLM_HOST="$VLLM_URL"
export VLLM_MODEL="kimi-k3"

# --- 5. Test 1/3: smoke test ---
log "=== STEP 5: TEST 1/3 -- smoke test ==="
{
    echo "--- GET /v1/models ---"
    curl -s "$VLLM_URL/v1/models"
    echo
    echo "--- POST /v1/chat/completions (minimal) ---"
    curl -s "$VLLM_URL/v1/chat/completions" -H 'Content-Type: application/json' \
        -d '{"model":"kimi-k3","messages":[{"role":"user","content":"Say OK and nothing else."}],"max_tokens":16}'
    echo
} > "$RESULTS_DIR/1_smoke_test.log" 2>&1
log "smoke test done, see $RESULTS_DIR/1_smoke_test.log"
grep -q '"role":"assistant"\|"role": "assistant"' "$RESULTS_DIR/1_smoke_test.log" \
    && log "smoke test: got an assistant response (looks OK)" \
    || log "WARNING: smoke test response did not look like a normal chat completion -- check the log"

# --- 6. Test 2/3: throughput / latency benchmark (reuses benchmark_glm.py,
# which reads VLLM_HOST/VLLM_MODEL from env -- no Kimi-specific copy needed) ---
log "=== STEP 6: TEST 2/3 -- throughput/latency benchmark ==="
.venv/bin/python benchmark_glm.py --sweep 1,4,16 -t 256 > "$RESULTS_DIR/2_benchmark.log" 2>&1
bench_rc=$?
log "benchmark finished, exit code $bench_rc, see $RESULTS_DIR/2_benchmark.log"
while IFS= read -r line; do log "  [bench] $line"; done < "$RESULTS_DIR/2_benchmark.log"

# --- 7. Test 3/3: quality prompts (coding / reasoning / general / long-context) ---
log "=== STEP 7: TEST 3/3 -- quality prompts ==="
.venv/bin/python - "$RESULTS_DIR/3_quality_prompts.log" <<'PYEOF'
import os, sys, time, requests

host = os.environ["VLLM_HOST"]
model = os.environ["VLLM_MODEL"]
outfile = sys.argv[1]

prompts = [
    ("coding_fibonacci", "Write a Python function that returns the nth Fibonacci number using memoization."),
    ("reasoning_sheep", "A farmer has 17 sheep, all but 9 die. How many are left? Explain your reasoning."),
    ("coding_sql", "Write a SQL query to find the second-highest salary from an Employees table with columns id, name, salary."),
    ("general_tcp_udp", "Explain the difference between TCP and UDP in two sentences."),
    ("long_context_summary", "Summarize in one paragraph: " + "The quick brown fox jumps over the lazy dog. " * 200),
]

ok, failed = 0, 0
with open(outfile, "w") as f:
    for name, prompt in prompts:
        shown = prompt[:200] + ("..." if len(prompt) > 200 else "")
        f.write(f"=== {name} ===\nPROMPT: {shown}\n")
        try:
            t0 = time.time()
            r = requests.post(f"{host}/v1/chat/completions", json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1024,
                "temperature": 0.3,
            }, timeout=600)
            dt = time.time() - t0
            r.raise_for_status()
            answer = r.json()["choices"][0]["message"]["content"]
            f.write(f"LATENCY: {dt:.1f}s\nRESPONSE: {answer}\n\n")
            ok += 1
        except Exception as e:
            f.write(f"ERROR: {e}\n\n")
            failed += 1
        f.flush()
print(f"quality prompts: {ok} ok, {failed} failed")
PYEOF
quality_rc=$?
log "quality prompts finished, exit code $quality_rc, see $RESULTS_DIR/3_quality_prompts.log"

log "=== ALL TESTS COMPLETE ==="
log "bench_rc=$bench_rc quality_rc=$quality_rc -- results directory: $RESULTS_DIR"
log "Proceeding to teardown (via trap)."
exit 0
