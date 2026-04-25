#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/train_spr_example.yaml}"
python scripts/train_spr_gc.py --config "${CONFIG}"
