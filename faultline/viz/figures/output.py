"""Saving a figure together with its caption file."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from faultline.viz.style import DEFAULT_FORMATS, save_figure


def save_with_caption(
    figure: Any, caption: str, stem: Path, formats: Sequence[str] = DEFAULT_FORMATS
) -> list[Path]:
    """Save a figure in each format and write its caption beside it.

    Args:
        figure: The finished figure; it is closed after saving.
        caption: The caption text.
        stem: Output path without a suffix.
        formats: Image formats to write.

    How:
        Writes ``<stem>.<format>`` for each format, then ``<stem>.caption.txt``, so the
        caption can go into a paper's own caption.

    Returns:
        The image paths, then the caption path.
    """
    paths = save_figure(figure, stem, formats)
    caption_path = stem.with_name(f"{stem.name}.caption.txt")
    caption_path.write_text(caption + "\n", encoding="utf-8")
    return [*paths, caption_path]
