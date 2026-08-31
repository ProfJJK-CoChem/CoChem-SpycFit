"""
Asynchronous Telemetry & Viewport Frame Broadcast Engine.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Dict, List, Set

logger = logging.getLogger("web_streaming")


class AsyncTelemetryBroadcaster:
    """
    Non-blocking async event broadcaster for live calculation telemetry frames.
    """

    def __init__(self) -> None:
        self.subscribers: Dict[str, Set[Callable[[Dict[str, Any]], None]]] = {
            "telemetry": set(),
            "coordinates": set(),
            "energy": set(),
        }

    def subscribe(self, channel: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register listener callback for channel."""
        if channel not in self.subscribers:
            self.subscribers[channel] = set()
        self.subscribers[channel].add(callback)

    def unsubscribe(self, channel: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Unregister listener callback from channel."""
        if channel in self.subscribers and callback in self.subscribers[channel]:
            self.subscribers[channel].remove(callback)

    def broadcast(self, channel: str, payload: Dict[str, Any]) -> None:
        """Broadcast payload synchronously to registered channel listeners."""
        if channel in self.subscribers:
            for cb in self.subscribers[channel]:
                try:
                    cb(payload)
                except Exception as exc:
                    logger.debug(f"Subscriber error on channel {channel}: {exc}")

    async def broadcast_async(self, channel: str, payload: Dict[str, Any]) -> None:
        """Asynchronous non-blocking broadcast."""
        self.broadcast(channel, payload)
        await asyncio.sleep(0)
