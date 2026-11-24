"""Shared public-wire client for ninfer-serve benchmark requests."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from tools.streaming_http.client import (
    HttpExchangeResult,
    HttpResponseHead,
    PreparedExchange,
    StreamingHttpClient,
)
from tools.streaming_http.sse import SseDecodeError, SseDecoder, SseEvent


class ServeProtocolError(RuntimeError):
    """Raised for unsupported protocols and malformed or failed serve HTTP exchanges."""
    pass


@dataclass(frozen=True)
class ProtocolRequest:
    """A frozen benchmark request: protocol, endpoint path, JSON payload, and streaming mode."""
    protocol: str
    path: str
    payload: dict[str, Any]
    stream: bool = True


@dataclass(frozen=True)
class ProtocolEvent:
    """One normalized wire event with kind, event type, receive timestamp, and payload details."""
    kind: str
    event_type: str
    received_ns: int
    payload: dict[str, Any] | None = None
    response_id: str | None = None
    output: str = ""
    error_code: str | None = None
    error_message: str | None = None


class StreamAdapter(Protocol):
    """The protocol-specific SSE-to-ProtocolEvent conversion interface used by prepared exchanges."""
    def consume(self, event: SseEvent) -> list[ProtocolEvent]:
        """Normalize one SSE frame into zero or more protocol events."""
        ...


@dataclass
class ServeExchangeResult:
    """A completed exchange: the HTTP transport result plus decoded protocol events and errors."""
    protocol: str
    body_bytes: int
    http: HttpExchangeResult
    events: list[ProtocolEvent] = field(default_factory=list)
    protocol_error: str | None = None
    error_code: str | None = None
    error_message: str | None = None


def _adapter(protocol: str) -> StreamAdapter:
    """Return the lazy-imported stream adapter for a named protocol, raising ServeProtocolError when unsupported."""
    if protocol == "openai_chat":
        from tools.ninfer_serve.openai_chat import ChatStreamAdapter

        return ChatStreamAdapter()
    if protocol == "openai_responses":
        from tools.ninfer_serve.openai_responses import ResponsesStreamAdapter

        return ResponsesStreamAdapter()
    if protocol == "anthropic_messages":
        from tools.ninfer_serve.anthropic import AnthropicStreamAdapter

        return AnthropicStreamAdapter()
    raise ServeProtocolError(f"unsupported Serve protocol: {protocol}")


def parse_http_error(protocol: str, status: int | None, body: bytes) -> tuple[str | None, str | None]:
    """Extract an error code and message from a non-2xx JSON response body."""
    if status is None or 200 <= status < 300:
        return None, None
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, body.decode("utf-8", errors="replace") or None
    if not isinstance(payload, dict):
        return None, repr(payload)
    error = payload.get("error")
    if isinstance(error, dict):
        code = error.get("code")
        message = error.get("message")
        if protocol == "anthropic_messages" and not isinstance(code, str):
            code = error.get("type")
        return (
            code if isinstance(code, str) and code else None,
            message if isinstance(message, str) and message else None,
        )
    return None, payload.get("message") if isinstance(payload.get("message"), str) else None


class PreparedServeExchange:
    """A prepared request exchange ready to execute against the streaming HTTP client."""
    def __init__(
        self,
        request: ProtocolRequest,
        body: bytes,
        exchange: PreparedExchange,
    ) -> None:
        """Bind the serialized request body to its prepared low-level exchange."""
        self.request = request
        self.body = body
        self._exchange = exchange

    @property
    def body_bytes(self) -> int:
        """The UTF-8 size of the serialized request body."""
        return len(self.body)

    @property
    def cancel_ns(self) -> int | None:
        """The nanosecond timestamp at which cancellation was requested, if issued."""
        return self._exchange.cancel_ns

    def cancel(self) -> int:
        """Request cancellation of the exchange and return its timestamp."""
        return self._exchange.cancel()

    def execute(
        self,
        *,
        on_sent: Callable[[int], None] | None = None,
        on_body_sent: Callable[[int], None] | None = None,
        on_headers: Callable[[HttpResponseHead], None] | None = None,
        on_event: Callable[[ProtocolEvent], None] | None = None,
    ) -> ServeExchangeResult:
        """Execute the exchange, decode SSE frames into protocol events, and return the aggregate result."""
        adapter = _adapter(self.request.protocol)
        decoder = SseDecoder()
        events: list[ProtocolEvent] = []
        head: HttpResponseHead | None = None
        protocol_error: str | None = None

        def emit(event: ProtocolEvent) -> None:
            """Record an event and forward it to the caller's on_event callback."""
            events.append(event)
            if on_event is not None:
                on_event(event)

        def headers(value: HttpResponseHead) -> None:
            """Capture response headers, enforce the SSE content type, and forward to on_headers."""
            nonlocal head, protocol_error
            head = value
            if on_headers is not None:
                on_headers(value)
            if self.request.stream and 200 <= value.status < 300:
                media_type = value.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                if media_type != "text/event-stream":
                    protocol_error = (
                        "streaming response Content-Type is not text/event-stream: "
                        f"{media_type or '<missing>'}"
                    )

        def consume_frame(frame: SseEvent) -> None:
            """Adapt one SSE frame into protocol events unless a protocol error already occurred."""
            nonlocal protocol_error
            if protocol_error is not None:
                return
            try:
                for normalized in adapter.consume(frame):
                    emit(normalized)
            except (ServeProtocolError, ValueError, KeyError, TypeError) as error:
                protocol_error = f"{type(error).__name__}: {error}"

        def chunk(data: bytes, received_ns: int) -> None:
            """Feed a received chunk to the SSE decoder and adapt each completed frame."""
            nonlocal protocol_error
            if head is None or not (200 <= head.status < 300) or not self.request.stream:
                return
            if protocol_error is not None:
                return
            try:
                for frame in decoder.feed(data, received_ns):
                    consume_frame(frame)
            except SseDecodeError as error:
                protocol_error = str(error)

        http = self._exchange.execute(
            on_sent=on_sent,
            on_body_sent=on_body_sent,
            on_headers=headers,
            on_chunk=chunk,
        )
        if self.request.stream and http.cancel_requested:
            # A cancellation intent that races after a complete protocol terminal is not a
            # cancelled exchange.  Conversely, shutdown may surface as a clean socket EOF rather
            # than an exception, so absence of a terminal event is the protocol-level authority.
            http.cancelled = (
                http.error is None
                and protocol_error is None
                and (http.status is None or 200 <= http.status < 300)
                and http.cancel_ns is not None
                and http.ended_ns is not None
                and http.cancel_ns <= http.ended_ns
                and not any(event.kind in {"terminal", "error"} for event in events)
            )
        if (
            self.request.stream
            and http.status is not None
            and 200 <= http.status < 300
            and http.error is None
            and not http.cancelled
            and protocol_error is None
        ):
            try:
                for frame in decoder.finish(http.ended_ns or 0):
                    consume_frame(frame)
            except SseDecodeError as error:
                protocol_error = str(error)

        error_code, error_message = parse_http_error(
            self.request.protocol, http.status, http.body
        )
        for event in events:
            if event.kind == "error":
                error_code = event.error_code or error_code
                error_message = event.error_message or error_message
                break
        return ServeExchangeResult(
            protocol=self.request.protocol,
            body_bytes=len(self.body),
            http=http,
            events=events,
            protocol_error=protocol_error,
            error_code=error_code,
            error_message=error_message,
        )


class NInferServeClient:
    """A client for ninfer-serve's public wire protocols over one base URL."""
    def __init__(self, base_url: str, timeout_seconds: float, api_key: str | None = None) -> None:
        """Initialize the client with a base URL, timeout, and optional bearer API key."""
        self._http = StreamingHttpClient(base_url, timeout_seconds)
        self._api_key = api_key

    def prepare(self, request: ProtocolRequest) -> PreparedServeExchange:
        """Serialize a request to JSON and prepare its POST exchange with auth and accept headers."""
        body = json.dumps(
            request.payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if request.stream else "application/json",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        exchange = self._http.prepare("POST", request.path, body, headers)
        return PreparedServeExchange(request, body, exchange)

    def get_json(self, path: str) -> dict[str, Any]:
        """GET a JSON object from a path, raising ServeProtocolError on failure."""
        headers = {"Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        exchange = self._http.prepare("GET", path, b"", headers)
        result = exchange.execute()
        if result.status != 200:
            raise ServeProtocolError(
                f"GET {path} returned HTTP {result.status}: "
                + result.body.decode("utf-8", errors="replace")
            )
        try:
            value = json.loads(result.body)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ServeProtocolError(f"GET {path} returned invalid JSON") from error
        if not isinstance(value, dict):
            raise ServeProtocolError(f"GET {path} did not return a JSON object")
        return value

    def discover_model(self) -> str:
        """Return the single resident model id advertised by /v1/models."""
        value = self.get_json("/v1/models")
        entries = value.get("data")
        if not isinstance(entries, list) or len(entries) != 1:
            raise ServeProtocolError("ninfer-serve must expose exactly one resident model")
        model = entries[0].get("id") if isinstance(entries[0], dict) else None
        if not isinstance(model, str) or not model:
            raise ServeProtocolError("ninfer-serve model listing has no valid model id")
        return model
