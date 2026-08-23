from typing import TYPE_CHECKING

from cochem_base.plugins.loader import hookimpl
from cochem_spycfit.gui import SpycFitTab

if TYPE_CHECKING:
    from cochem_base.gui.main_window import MainWindow

@hookimpl
def register_tabs(main_window: "MainWindow") -> None:
    # Register CoChem-SpycFit
    spycfit_tab = SpycFitTab()
    main_window.tabs.addTab(spycfit_tab, "SpycFit - Bayesian Fitting")