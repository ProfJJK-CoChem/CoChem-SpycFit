# ==============================================================================
# CoChem-SPYFY: Air-Gapped Voila Dashboard Execution Entrypoint
# ==============================================================================
Write-Host "[CoChem-SPYFY] Launching Voila Air-Gapped Dashboard..." -ForegroundColor Cyan
voila Start_SPYFY.ipynb --theme=dark --no-browser --host 127.0.0.1 --template=airgap
