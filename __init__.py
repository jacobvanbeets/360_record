# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""360 Record Plugin for LichtFeld Studio.

Record camera path videos with linear and orbital segments.
"""

import lichtfeld as lf

from .panels.linear_path_panel import LinearPathPanel
from .operators.point_picker import LINEARPATH_OT_pick_point
from .operators.path_preview import PATHPREVIEW_OT_preview

_classes = [LinearPathPanel, LINEARPATH_OT_pick_point, PATHPREVIEW_OT_preview]


def on_load():
    """Called when plugin loads."""
    for cls in _classes:
        lf.register_class(cls)
    lf.log.info("360 Record plugin loaded")


def on_unload():
    """Called when plugin unloads."""
    for cls in reversed(_classes):
        lf.unregister_class(cls)
    lf.log.info("360 Record plugin unloaded")


__all__ = [
    "LinearPathPanel",
]
