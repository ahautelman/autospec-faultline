"""The footer band every figure shares: one legend, optionally a caption beneath it.

Constrained layout cannot place a legend and a caption together without overlap, so the
footer is sized explicitly in inches, the figure is grown by that much, and each item is
placed at a fixed offset inside it.
"""

from __future__ import annotations

import textwrap
from typing import Any

from faultline.viz.legend import add_legend, legend_handles, legend_rows
from faultline.viz.style import Role

# Footer geometry in inches: one legend row, one caption line, and the gap around them.
LEGEND_ROW_INCHES = 0.2
CAPTION_LINE_INCHES = 0.125
FOOTER_PADDING = 0.12
# Average width of a 7 pt serif character, so the caption wraps to the figure's width.
CAPTION_CHAR_INCHES = 0.06
CAPTION_SIZE = 7


def add_footer(
    figure: Any, roles: set[Role], caption: str | None, width: float, body_height: float
) -> None:
    """Reserve a footer band and put the legend, and optionally the caption, in it.

    Args:
        figure: The figure whose panels are already drawn.
        roles: The roles drawn on it.
        caption: Text to place under the legend, or ``None`` to omit it.
        width: Figure width in inches.
        body_height: Height of the panel area in inches.

    How:
        Sizes the footer from the legend's row count and the caption's wrapped line
        count, grows the figure by that much, confines the panels to the area above
        it, and places each footer item at a fixed offset inside it. Fixed offsets
        keep the legend and caption from overlapping whatever the layout does.
    """
    wrapped = textwrap.fill(caption, int(width / CAPTION_CHAR_INCHES)) if caption else ""
    legend_height = legend_rows(len(legend_handles(roles)[1])) * LEGEND_ROW_INCHES
    caption_height = (wrapped.count("\n") + 1) * CAPTION_LINE_INCHES if wrapped else 0.0
    footer = legend_height + caption_height + FOOTER_PADDING * (bool(legend_height) + bool(wrapped))
    total = body_height + footer
    figure.set_size_inches(width, total)
    figure.get_layout_engine().set(rect=(0.0, footer / total, 1.0, body_height / total))
    add_legend(figure, roles, bottom=(caption_height + FOOTER_PADDING * bool(wrapped)) / total)
    if wrapped:
        figure.text(0.5, (caption_height + FOOTER_PADDING / 2) / total, wrapped, ha="center",
                    va="top", fontsize=CAPTION_SIZE)
