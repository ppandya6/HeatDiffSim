#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python -m venv .venv
.venv/bin/python -m pip install -e .
export MPLCONFIGDIR=/tmp/heatdiffsim-mpl
export XDG_CACHE_HOME=/tmp/heatdiffsim-cache
.venv/bin/python -m unittest discover -s tests -v
