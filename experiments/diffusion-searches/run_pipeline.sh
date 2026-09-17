#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

bash experiments/setup_data.sh
source venv/bin/activate
python3 experiments/embed_fonts.py
python3 experiments/embed_text.py
echo "PIPELINE COMPLETE"
