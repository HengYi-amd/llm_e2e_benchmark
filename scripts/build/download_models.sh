#!/usr/bin/env bash
# Optional: pre-fetch model weights into the project-local HF_HOME, turning a
# mid-sweep network failure into an early one.
set -Eeuo pipefail
# shellcheck disable=SC1091
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)/env.sh"

# Driven by config; the smoke model must stay in step with
# scripts/bench/route_smoke.sh.
MODELS="${E2E_DOWNLOAD_MODELS:-$E2E_MODELS ${SMOKE_MODEL:-Qwen/Qwen3-0.6B}}"
MODELS="$(echo "$MODELS" | tr ' ' '\n' | grep -v '^$' | awk '!seen[$0]++' | tr '\n' ' ')"

HF="${E2E_HF_CLI:-$(command -v hf || command -v huggingface-cli || true)}"
[ -n "$HF" ] || { echo "[FATAL] no hf / huggingface-cli on PATH"; exit 1; }

echo "[download] HF_HOME=$HF_HOME"
echo "[download] models: $MODELS"

# Check reachability once, before starting a large download.
probe="$(echo "$MODELS" | awk '{print $1}')"
code=$(curl -s -o /dev/null -w '%{http_code}' -m 20 -L \
    "https://huggingface.co/api/models/$probe" || echo 000)
echo "[download] huggingface api -> $code"
[ "$code" = "200" ] || { echo "[FATAL] huggingface unreachable"; exit 1; }

for m in $MODELS; do
    echo "[download] === $m ==="
    t0=$(date +%s)
    for attempt in 1 2 3; do
        if "$HF" download "$m" --quiet; then
            break
        fi
        echo "[download] $m attempt $attempt failed, retrying in 30s"
        sleep 30
    done
    t1=$(date +%s)
    echo "[download] $m took $((t1-t0))s"
done

echo "[download] disk usage:"
du -sh "$HF_HOME" 2>/dev/null || true
echo '[download] DONE'
