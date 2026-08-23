"""
CoChem-SpycFit Core Engine: Stage 0 Hardware & Precision Gatekeeper and Fit Registry Manager (Root Proxy).
"""

from __future__ import annotations

import logging

_proxy_logger = logging.getLogger("CoChem-SpycFit.CoreEngineProxy")

try:
    from src.cochem_spycfit.core_engine import *  # noqa: F401, F403
except ImportError:
    from cochem_spycfit.core_engine.cochem_fit_registry_manager import *  # noqa: F401, F403
    try:
        from cochem_spycfit.core_engine.cochem_spycfit_init import *  # noqa: F401, F403
    except ImportError as _err:
        _proxy_logger.debug("Root core_engine proxy import resolution completed: %s", _err)
