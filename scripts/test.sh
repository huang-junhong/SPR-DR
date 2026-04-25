#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/test_spr_example.yaml}"
python scripts/test_spr.py --config "${CONFIG}"
