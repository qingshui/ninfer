"""Lazy logical values, encoded rows, and transforms independent of source files."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import prod
from typing import Callable, Mapping

import torch


@dataclass(frozen=True, slots=True)
class EncodedRows:
    """One encoded row group: format, code and scale tensors, and optional weight scale."""
    format: str
    codes: torch.Tensor
    scales: torch.Tensor
    weight_scale: bytes | None = None


@dataclass(frozen=True, slots=True)
class EncodedPlanes:
    """Native matrix planes; ranges and lengths are bytes, without numerical decoding."""

    layout: str
    sizes: Mapping[str, int]
    read: Callable[[str, int, int], torch.Tensor]
    exception_count: int


@dataclass(frozen=True, slots=True)
class LogicalSource:
    """Values use flat C-order elements; an optional encoded reader uses full rows."""

    shape: tuple[int, ...]
    label: str
    read_values: Callable[[int, int], torch.Tensor]
    read_encoded: Callable[[int, int], EncodedRows] | None = None
    weight_scale: Callable[[], bytes] | None = None
    activation_scale: Callable[[], bytes] | None = None
    encoded_format: str | Callable[[], str | None] | None = None
    native_planes: EncodedPlanes | None = None
    matrix_sources: tuple[LogicalSource, ...] = ()

    def encoding(self) -> str | None:
        """Describe encoded input without reading its weight rows."""
        return (
            self.encoded_format()
            if callable(self.encoded_format)
            else self.encoded_format
        )

    def values(self, begin: int = 0, end: int | None = None) -> torch.Tensor:
        """Read flat C-order elements in [begin, end), validating the range and count."""
        end = prod(self.shape) if end is None else end
        if not 0 <= begin <= end <= prod(self.shape):
            raise ValueError(
                f"{self.label}: logical range [{begin},{end}) exceeds {self.shape}"
            )
        values = self.read_values(begin, end)
        if values.numel() != end - begin:
            raise ValueError(
                f"{self.label}: source returned {values.numel()} values, expected {end-begin}"
            )
        return values.reshape(-1)

    def rows(self, begin: int, end: int) -> torch.Tensor:
        """Read whole rows [begin, end) of a two-dimensional source as a matrix."""
        if len(self.shape) != 2:
            raise ValueError(f"{self.label}: row access requires a matrix")
        k = self.shape[1]
        return self.values(begin * k, end * k).reshape(end - begin, k)

    def reshape(self, shape: tuple[int, ...]) -> LogicalSource:
        """Return a source with a new shape of equal element count."""
        if prod(shape) != prod(self.shape):
            raise ValueError(f"{self.label}: reshape changes the logical element count")
        if self.encoding() is not None and shape[-1:] != self.shape[-1:]:
            raise ValueError(
                f"{self.label}: encoded reshape must preserve matrix columns"
            )
        return replace(self, shape=shape)


def array_source(values: torch.Tensor, label: str) -> LogicalSource:
    """Return a LogicalSource reading from a detached contiguous CPU tensor."""
    data = values.detach().cpu().contiguous()
    return LogicalSource(
        tuple(data.shape), label, lambda begin, end: data.reshape(-1)[begin:end]
    )


def transpose_source(
    source: LogicalSource, axes: tuple[int, ...], shape: tuple[int, ...]
) -> LogicalSource:
    """Return a source that permutes axes of another source into the given flat shape."""
    # The architecture uses this for small convolution and norm-related tensors.
    def read(begin: int, end: int) -> torch.Tensor:
        """Read permuted flat elements in [begin, end)."""
        return (
            source.values()
            .reshape(source.shape)
            .permute(axes)
            .contiguous()
            .reshape(-1)[begin:end]
        )

    return LogicalSource(shape, f"transpose({source.label},{axes})", read)


def select_rows(
    source: LogicalSource, ranges: tuple[tuple[int, int], ...]
) -> LogicalSource:
    """Return a matrix source restricted to the given row ranges."""
    if len(source.shape) != 2 or not ranges:
        raise ValueError("row selection requires a matrix and nonempty ranges")
    n, k = source.shape
    for begin, end in ranges:
        if not 0 <= begin < end <= n:
            raise ValueError(f"{source.label}: invalid selected rows [{begin},{end})")
    rows = sum(end - begin for begin, end in ranges)
    if source.native_planes is not None:
        if ranges == ((0, n),):
            return source
        raise ValueError(
            f"{source.label}: native plane selection requires complete matrices"
        )

    def spans(begin: int, end: int):
        """Yield absolute value ranges covering [begin, end) across the selected rows."""
        cursor = 0
        for first, last in ranges:
            length = (last - first) * k
            low, high = max(begin, cursor), min(end, cursor + length)
            if low < high:
                yield first * k + low - cursor, first * k + high - cursor
            cursor += length

    def read(begin: int, end: int) -> torch.Tensor:
        """Read flat values in [begin, end) by concatenating the touched row ranges."""
        if begin == end:
            return source.values(0, 0)
        parts = [source.values(a, b) for a, b in spans(begin, end)]
        return torch.cat(parts) if len(parts) != 1 else parts[0]

    def encoded(begin: int, end: int) -> EncodedRows:
        """Read encoded rows in [begin, end) by concatenating encoded row groups."""
        if source.read_encoded is None:
            raise ValueError(f"{source.label}: encoded source rows are unavailable")
        pieces = [
            source.read_encoded(a // k, b // k) for a, b in spans(begin * k, end * k)
        ]
        first = pieces[0]
        if any(
            (part.format, part.weight_scale) != (first.format, first.weight_scale)
            for part in pieces
        ):
            raise ValueError(f"{source.label}: incompatible encoded row groups")
        return EncodedRows(
            first.format,
            torch.cat([p.codes for p in pieces]),
            torch.cat([p.scales for p in pieces]),
            first.weight_scale,
        )

    return LogicalSource(
        (rows, k),
        f"rows({source.label}, {len(ranges)} spans, {rows} rows)",
        read,
        encoded if source.read_encoded is not None else None,
        source.weight_scale,
        source.activation_scale,
        encoded_format=source.encoded_format,
    )


def gather_source(source: LogicalSource, indices: torch.Tensor) -> LogicalSource:
    """Return a matrix source reordered by the given row indices."""
    if len(source.shape) != 2:
        raise ValueError("gather source must be a matrix")
    ids = indices.to(torch.int64).cpu().tolist()
    if not ids or min(ids) < 0 or max(ids) >= source.shape[0]:
        raise ValueError("gather indices are outside source rows")
    return select_rows(source, tuple((row, row + 1) for row in ids))
