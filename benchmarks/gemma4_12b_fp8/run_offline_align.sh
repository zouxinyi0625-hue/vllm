#!/usr/bin/env bash
# Run offline alignment benchmark from a config JSON.
set -euo pipefail
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
cd "$(dirname "$SCRIPT_PATH")"

CONFIG=${1:-configs/26b_e011_mtp.json}
shift || true

python3 bench_offline_align.py --config "$CONFIG" "$@"
