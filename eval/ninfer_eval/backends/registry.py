"""Registry of named evaluation backends with lazy built-in loading."""

from __future__ import annotations

from .base import EvaluationBackend

_BACKENDS: dict[str, EvaluationBackend] = {}
_BUILTINS_LOADED = False


def register_backend(backend: EvaluationBackend) -> None:
    """Register a backend instance, refusing a duplicate name."""
    if backend.name in _BACKENDS:
        raise RuntimeError(f"backend already registered: {backend.name}")
    _BACKENDS[backend.name] = backend


def get_backend(name: str) -> EvaluationBackend:
    """Return the registered backend for name, loading builtins first."""
    _ensure_builtins()
    try:
        return _BACKENDS[name]
    except KeyError as exc:
        raise ValueError(f"unknown evaluation backend: {name}") from exc


def backend_names() -> list[str]:
    """Return the sorted names of every registered backend."""
    _ensure_builtins()
    return sorted(_BACKENDS)


def _ensure_builtins() -> None:
    """Register the built-in mock and EvalScope backends once."""
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    from .evalscope import EvalScopeBackend
    from .mock import MockBackend

    for backend in (MockBackend(), EvalScopeBackend()):
        if backend.name not in _BACKENDS:
            register_backend(backend)
    _BUILTINS_LOADED = True
