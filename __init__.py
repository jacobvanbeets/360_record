# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""360 Record Plugin for LichtFeld Studio.

Record circular camera track videos around a point of interest.
"""

import lichtfeld as lf

from .panels.camera_track_panel import CameraTrackPanel
from .panels.linear_path_panel import LinearPathPanel
from .operators.poi_picker import RECORD360_OT_pick_poi
from .operators.point_picker import LINEARPATH_OT_pick_point

_classes = [CameraTrackPanel, LinearPathPanel, RECORD360_OT_pick_poi, LINEARPATH_OT_pick_point]


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
    "CameraTrackPanel",
    "LinearPathPanel",
]
