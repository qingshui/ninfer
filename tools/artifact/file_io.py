"""Keep large offline transfers from retaining whole artifacts in Linux's page cache."""

from __future__ import annotations

import os

IO_CHUNK_BYTES = 8 * 1024 * 1024
WRITEBACK_BYTES = 64 * 1024 * 1024
_PAGE_BYTES = os.sysconf("SC_PAGE_SIZE")


def discard_cached_pages(fd: int, offset: int = 0, count: int | None = None) -> None:
    """Ask the kernel to drop cached pages for a file range using POSIX_FADV_DONTNEED."""
    if count is None:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    elif count > 0:
        begin = offset // _PAGE_BYTES * _PAGE_BYTES
        end = (offset + count + _PAGE_BYTES - 1) // _PAGE_BYTES * _PAGE_BYTES
        os.posix_fadvise(fd, begin, end - begin, os.POSIX_FADV_DONTNEED)


class Writeback:
    """Bound dirty output across all open shards; release clean pages after writeback."""

    def __init__(self) -> None:
        """Start with zero pending bytes and no registered descriptors."""
        self._bytes = 0
        self._fds: set[int] = set()

    def written(self, fd: int, count: int) -> None:
        """Account for count bytes written to fd, flushing when the dirty-byte budget is exceeded."""
        self._fds.add(fd)
        self._bytes += count
        if self._bytes >= WRITEBACK_BYTES:
            self.flush()

    def flush(self) -> None:
        """Fdatasync all tracked descriptors, drop their clean pages, and reset the byte accounting."""
        for fd in self._fds:
            os.fdatasync(fd)
            discard_cached_pages(fd)
        self._fds.clear()
        self._bytes = 0
