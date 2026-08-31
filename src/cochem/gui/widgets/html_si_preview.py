"""
CoChem GUI Embedded Interactive HTML SI Previewer Widget (REQ-QC-095).

Provenance:
- Method Matrix [M]: UI container for offline WebGL Supporting Information packages
  with local file serving, browser dispatch, and scene reloading actions.
"""

from __future__ import annotations

import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class HTMLSIPreviewState(BaseModel):
    """Data contract representing the interactive HTML SI preview state."""
    model_config = ConfigDict(frozen=True)

    loaded_html_path: Optional[str] = None
    is_loaded: bool = False
    status_message: str = "No HTML package loaded"


class HTMLSIPreviewWidget:
    """
    [REQ-QC-095] GUI Controller for Interactive HTML Supporting Information Preview.
    """

    def __init__(self, html_path: Optional[str] = None) -> None:
        self.html_path = html_path
        if html_path and Path(html_path).is_file():
            self._state = HTMLSIPreviewState(
                loaded_html_path=str(Path(html_path).resolve()),
                is_loaded=True,
                status_message=f"Loaded HTML bundle: {Path(html_path).name}",
            )
        else:
            self._state = HTMLSIPreviewState()

    def load_html_bundle(self, html_path: str) -> HTMLSIPreviewState:
        """Load an interactive HTML Supporting Information file."""
        p = Path(html_path)
        if not p.is_file():
            raise FileNotFoundError(f"HTML bundle file not found at '{p}'")

        self.html_path = str(p.resolve())
        self._state = HTMLSIPreviewState(
            loaded_html_path=self.html_path,
            is_loaded=True,
            status_message=f"Loaded HTML bundle: {p.name}",
        )
        return self._state

    def open_in_external_browser(self) -> bool:
        """Open the loaded interactive HTML Supporting Information file in default system browser."""
        if not self._state.loaded_html_path:
            raise ValueError("No HTML bundle loaded to open.")
        url = f"file:///{Path(self._state.loaded_html_path).resolve().as_posix()}"
        return webbrowser.open(url)

    def reload_scene(self) -> HTMLSIPreviewState:
        """Refresh or reload the HTML bundle state."""
        if not self._state.loaded_html_path:
            return self._state
        self._state = HTMLSIPreviewState(
            loaded_html_path=self._state.loaded_html_path,
            is_loaded=True,
            status_message="Scene refreshed.",
        )
        return self._state

    @property
    def state(self) -> HTMLSIPreviewState:
        return self._state
