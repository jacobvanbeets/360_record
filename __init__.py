# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""360 Record Plugin for LichtFeld Studio.

Record circular camera track videos around a point of interest.
"""

import lichtfeld as lf

from .panels.camera_track_panel import CameraTrackPanel
from .operators.poi_picker import RECORD360_OT_pick_poi

_classes = [CameraTrackPanel, RECORD360_OT_pick_poi]


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
]
