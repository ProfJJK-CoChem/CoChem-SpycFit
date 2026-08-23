@echo off
REM ==============================================================================
REM CoChem-SpycFit: Air-Gapped Voila Dashboard Execution Entrypoint
REM ==============================================================================
echo [CoChem-SpycFit] Launching Voila Air-Gapped Dashboard...
voila Start_SpycFit.ipynb --theme=dark --no-browser --host 127.0.0.1 --template=airgap
