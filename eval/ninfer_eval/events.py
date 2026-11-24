"""Run progress events and the sink that persists, logs, and redacts them."""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

EVENT_SCHEMA_VERSION = 1


def utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string ending in Z."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class RunEvent:
    """One structured progress event emitted during an evaluation run."""

    kind: str
    run_id: str
    job_id: str | None = None
    phase: str | None = None
    completed: int | None = None
    total: int | None = None
    unit: str | None = None
    message: str = ""
    level: str = "info"
    metrics: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=utc_now)
    schema_version: int = EVENT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return the event serialized as a plain dict."""
        return asdict(self)


class EventRenderer(Protocol):
    """Protocol for objects that display progress events as they arrive."""

    def handle(self, event: RunEvent) -> None:
        """Display one progress event."""
        ...
    def close(self) -> None:
        """Release any resources held by the renderer."""
        ...


class NullRenderer:
    """No-op renderer that accepts and discards events."""

    def handle(self, event: RunEvent) -> None:
        """Discard the event without producing any output."""
        del event

    def close(self) -> None:
        """Do nothing; this renderer holds no resources."""
        pass


class EventSink:
    """Persists redacted events to a JSONL file, logs them, and forwards them to a renderer."""

    def __init__(
        self,
        path: Path,
        logger: logging.Logger,
        renderer: EventRenderer | None = None,
        secrets: list[str] | None = None,
    ):
        """Open path for appended JSONL output and keep logger, renderer, and secret values."""
        self._path = path
        self._logger = logger
        self._renderer = renderer or NullRenderer()
        self._secrets = [value for value in (secrets or []) if value]
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("a", encoding="utf-8")

    def emit(self, event: RunEvent) -> None:
        """Append the redacted event to the JSONL file, log it, and hand it to the renderer."""
        safe = self._redact_event(event)
        with self._lock:
            self._file.write(json.dumps(safe.to_dict(), ensure_ascii=False) + "\n")
            self._file.flush()
            log_method = getattr(self._logger, safe.level, self._logger.info)
            prefix = f"[{safe.job_id}] " if safe.job_id else ""
            progress = ""
            if safe.completed is not None:
                progress = f" {safe.completed}/{safe.total if safe.total is not None else '?'} {safe.unit or ''}"
            log_method("%s%s%s", prefix, safe.message or safe.kind, progress)
            self._renderer.handle(safe)

    def close(self) -> None:
        """Close the renderer and the underlying JSONL file."""
        with self._lock:
            self._renderer.close()
            self._file.close()

    def _redact_event(self, event: RunEvent) -> RunEvent:
        """Return a copy of the event with every configured secret replaced by <redacted>."""
        data = event.to_dict()
        encoded = json.dumps(data, ensure_ascii=False)
        for secret in self._secrets:
            encoded = encoded.replace(secret, "<redacted>")
        return RunEvent(**json.loads(encoded))
