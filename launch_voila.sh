#!/usr/bin/env bash
# ==============================================================================
# CoChem-SpycFit: Air-Gapped Voila Dashboard Execution Entrypoint
# ==============================================================================
echo "[CoChem-SpycFit] Launching Voila Air-Gapped Dashboard..."
voila Start_SpycFit.ipynb --theme=dark --no-browser --host 127.0.0.1 --template=airgap
