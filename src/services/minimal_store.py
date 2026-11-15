"""Persistent per-chat toggle for minimal UI mode.

Minimal mode is the **default**: the bot suppresses routine progress/status
messages (queued, downloading, uploading, "Done!") so only the finished post
appears. Failures are still reported. A chat can opt out with ``/minimal off``
to get the status messages back.

Opted-out chat IDs persist to a JSON file (``MINIMAL_MODE_FILE``) as
``{"verbose": [ids]}`` so the setting survives restarts; with no file
configured the store is in-memory only. The legacy format (a bare list of
chats that had turned minimal mode *on*) is read as "no opt-outs", since those
chats already match the new default.
"""

import json
from pathlib import Path
from typing import Optional

from ..utils.logger import get_logger

logger = get_logger()


class MinimalModeStore:
    """Tracks chats that opted out of (default-on) minimal mode, optionally file-backed."""

    def __init__(self, path: str = ""):
        self._path: Optional[Path] = Path(path) if path else None
        self._verbose: set[int] = set()
        self._load()

    def _load(self) -> None:
        if not self._path or not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self._verbose = {int(c) for c in data.get("verbose", [])}
            # A bare list is the legacy "minimal on" set — all default now.
        except (OSError, json.JSONDecodeError, TypeError, ValueError, AttributeError) as e:
            logger.error(f"Could not read minimal-mode file {self._path}: {e}")

    def contains(self, chat_id: int) -> bool:
        """True if minimal mode is on for this chat (the default)."""
        return chat_id not in self._verbose

    def set(self, chat_id: int, enabled: bool) -> None:
        """Enable/disable minimal mode for a chat and persist."""
        if enabled == self.contains(chat_id):
            return
        if enabled:
            self._verbose.discard(chat_id)
        else:
            self._verbose.add(chat_id)
        self._persist()
        logger.info("Minimal UI mode changed", chat_id=chat_id, enabled=enabled)

    def _persist(self) -> None:
        if not self._path:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps({"verbose": sorted(self._verbose)}), encoding="utf-8"
            )
        except OSError as e:
            logger.error(f"Could not write minimal-mode file {self._path}: {e}")


_minimal_store: Optional[MinimalModeStore] = None


def get_minimal_store() -> MinimalModeStore:
    """Get the global minimal-mode store instance."""
    global _minimal_store
    if _minimal_store is None:
        from ..config import get_settings
        _minimal_store = MinimalModeStore(get_settings().minimal_mode_file)
    return _minimal_store
