@echo off
REM ==============================================================================
REM CoChem-SPYFY: Air-Gapped Voila Dashboard Execution Entrypoint
REM ==============================================================================
echo [CoChem-SPYFY] Launching Voila Air-Gapped Dashboard...
voila Start_SPYFY.ipynb --theme=dark --no-browser --host 127.0.0.1 --template=airgap
