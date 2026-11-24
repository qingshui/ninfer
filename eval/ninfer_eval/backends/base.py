"""Shared protocol and data types that every ninfer evaluation backend implements."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..config import JobConfig
from ..events import EventSink
from ..result import DatasetResult
from ..secrets import ResolvedTarget


class BackendDependencyError(RuntimeError):
    """Raised when a configured backend dependency is absent or incompatible."""


@dataclass(frozen=True)
class WorkPlan:
    """Frozen description of the work a backend will perform for a job."""

    total: int | None
    unit: str = "samples"
    subsets: tuple[str, ...] = ()
    supports_resume: bool = False
    requirements: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass
class RunContext:
    """Runtime inputs for one backend run: the job, its directories, cancellation, and limits."""

    run_id: str
    job: JobConfig
    job_dir: Path
    target: ResolvedTarget | None
    granted_concurrency: int
    cancel_event: threading.Event
    resume: bool
    plan: WorkPlan
    sample_retention: str = "all"


@dataclass
class BackendRun:
    """Timings and raw output of one completed backend run, plus its artifact paths."""

    started_monotonic: float
    finished_monotonic: float
    raw_result: Any
    artifacts: list[Path] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        """Elapsed wall-clock seconds of the run, clamped to be nonnegative."""
        return max(0.0, self.finished_monotonic - self.started_monotonic)


class EvaluationBackend(Protocol):
    """Protocol that every ninfer evaluation backend must implement."""

    name: str

    def validate(
        self, job: JobConfig, target: ResolvedTarget | None, for_run: bool = False
    ) -> None:
        """Check that this backend can run the job against the resolved target."""
        ...
    def plan(self, job: JobConfig, target: ResolvedTarget | None) -> WorkPlan:
        """Declare the work this backend would perform for the job."""
        ...
    def run(self, context: RunContext, events: EventSink) -> BackendRun:
        """Execute the evaluation job and return the raw backend run."""
        ...
    def normalize(self, context: RunContext, run: BackendRun) -> DatasetResult:
        """Convert one raw backend run into the common dataset result."""
        ...
