from typing import TYPE_CHECKING, Any, Callable, TypeVar

try:
    from cochem_base.plugins.loader import hookimpl
except ImportError:
    try:
        import pluggy

        hookimpl = pluggy.HookimplMarker("cochem")
    except ImportError:
        _F = TypeVar("_F", bound=Callable[..., Any])

        def hookimpl(func: _F) -> _F:
            """Identity decorator fallback when neither cochem_base nor pluggy is installed."""
            return func

from cochem_spyfy.gui import SpycFitTab

if TYPE_CHECKING:
    from cochem_base.gui.main_window import MainWindow

@hookimpl
def register_tabs(main_window: "MainWindow") -> None:
    # Register CoChem-SPYFY
    spycfit_tab = SpycFitTab()
    main_window.tabs.addTab(spycfit_tab, "SpycFit - Bayesian Fitting")
