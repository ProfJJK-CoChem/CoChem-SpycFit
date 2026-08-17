from typing import Any
import pluggy
from cochem_spycfit.gui import SpycFitTab

@pluggy.HookimplMarker("cochem_studio")
def register_tabs(main_window: Any) -> None:
    # Register CoChem-SpycFit
    spycfit_tab = SpycFitTab()
    main_window.tabs.addTab(spycfit_tab, "SpycFit - Bayesian Fitting")