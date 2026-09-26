"""Console progress helpers for the reporting pipeline.

The reporting run is long (many edges times a 200k budget), so every stage reports
where it is. tqdm is the preferred bar, but it is not a hard dependency of the
package: the fast test suite imports this module without the ``reports`` or
``interop`` extras installed. Each helper therefore imports tqdm lazily and falls
back to a no-op bar or the plain iterable, so importing never fails.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, TypeVar

T = TypeVar("T")

# A process-wide switch a pool worker flips so no bar is drawn: several workers
# writing bars to one terminal interleave unreadably.
_DISABLED = False


class _NullBar:
    """A stand-in progress bar used when tqdm is not installed or bars are off."""

    def update(self, n: int = 1) -> None:
        pass

    def close(self) -> None:
        pass


def disable() -> None:
    """Silence progress bars in this process.

    How:
        Sets a module flag the bar builders consult. A worker process calls this
        before running a job so its bars do not interleave with other workers'.

    Returns:
        None: the flag is set for the rest of this process.
    """
    global _DISABLED
    _DISABLED = True


def enable() -> None:
    """Enable progress bars again after :func:`disable`.

    How:
        Clears the module flag so tests can restore the default in-process
        behaviour after exercising the disabled path.

    Returns:
        None: the flag is cleared for the rest of this process.
    """
    global _DISABLED
    _DISABLED = False


def progress_bar(total: int, desc: str = "") -> Any:
    """Create a step progress bar, or a no-op bar when tqdm is absent.

    Args:
        total: Number of steps the bar counts.
        desc: Leading label for the bar.

    How:
        Imports tqdm lazily; when the import fails, returns the no-op bar so
        callers can update and close unconditionally.

    Returns:
        A tqdm bar when tqdm is installed, otherwise a no-op bar.
    """
    if _DISABLED:
        return _NullBar()
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return _NullBar()
    return tqdm(total=total, desc=desc, unit="step", leave=True)


def track(iterable: Iterable[T], desc: str = "", leave: bool = True) -> Iterable[T]:
    """Wrap an iterable in a tqdm bar, or return it unchanged when tqdm is absent.

    Args:
        iterable: The iterable to iterate over.
        desc: Leading label for the bar.
        leave: Whether the finished bar stays on screen.

    How:
        Imports tqdm lazily; when the import fails, returns the iterable so a
        caller can always write ``for item in track(items)``.

    Returns:
        The iterable, wrapped in a tqdm bar when tqdm is installed.
    """
    if _DISABLED:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, desc=desc, leave=leave)
