#!/usr/bin/env bash
# Run the pinned Pi with an isolated environment: clean env, HOME=$PI_HOME, cwd=$PI_CWD, offline, no telemetry.
# Usage: PI_HOME=... PI_CWD=... [PROXY_BASE_URL=...] [ROLLOUT_ID=...] pi_iso.sh [pi args...]
set -euo pipefail
PI_BIN=/home/ubuntu/rsi/harness_rl/pi_runtime/node_modules/.bin/pi
cd "$PI_CWD"
exec env -i PATH=/usr/bin:/bin HOME="$PI_HOME" \
    PROXY_BASE_URL="${PROXY_BASE_URL:-}" ROLLOUT_ID="${ROLLOUT_ID:-none}" \
    PI_OFFLINE=1 PI_TELEMETRY=0 ${PI_EXTRA_ENV:-} \
    "$PI_BIN" "$@" < /dev/null
