# ==============================================================================
# CoChem-SpycFit: Air-Gapped Voila Dashboard Execution Entrypoint
# ==============================================================================
Write-Host "[CoChem-SpycFit] Launching Voila Air-Gapped Dashboard..." -ForegroundColor Cyan
voila Start_SpycFit.ipynb --theme=dark --no-browser --host 127.0.0.1 --template=airgap
