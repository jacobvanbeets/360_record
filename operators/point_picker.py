# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Generic point picker operator for selecting points in 3D space."""

from typing import Optional, Tuple, Callable

import lichtfeld as lf
import lichtfeld.selection as sel
from lfs_plugins.types import Operator, Event


# Module-level state for point picking
_point_callback: Optional[Callable[[Tuple[float, float, float]], None]] = None
_point_cancelled = False
_pick_label = "point"


def set_point_callback(
    callback: Callable[[Tuple[float, float, float]], None],
    label: str = "point"
):
    """Set the callback to invoke when a point is picked.
    
    Args:
        callback: Function that receives the picked world position (x, y, z)
        label: Label for the point being picked (for UI feedback)
    """
    global _point_callback, _point_cancelled, _pick_label
    _point_callback = callback
    _point_cancelled = False
    _pick_label = label


def clear_point_callback():
    """Clear the point callback."""
    global _point_callback, _point_cancelled
    _point_callback = None
    _point_cancelled = True


def was_point_cancelled() -> bool:
    """Check if point picking was cancelled and clear the flag.
    
    Returns:
        True if picking was cancelled
    """
    global _point_cancelled
    if _point_cancelled:
        _point_cancelled = False
        return True
    return False


def get_pick_label() -> str:
    """Get the current pick label."""
    return _pick_label


class LINEARPATH_OT_pick_point(Operator):
    """Modal operator for picking a point on the model."""
    
    label = "Pick Point"
    description = "Click on the model to pick a point"
    options = {'BLOCKING'}
    
    def invoke(self, context, event: Event) -> set:
        """Start modal mode."""
        return {'RUNNING_MODAL'}
    
    def modal(self, context, event: Event) -> set:
        """Handle mouse events for picking."""
        global _point_callback
        
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            # Try to pick at mouse position
            result = sel.pick_at_screen(event.mouse_region_x, event.mouse_region_y)
            
            if result is not None and _point_callback is not None:
                # Call callback with the picked position
                _point_callback(tuple(result.world_position))
                clear_point_callback()
                return {'FINISHED'}
            
            # If no hit, continue waiting
            return {'RUNNING_MODAL'}
        
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            clear_point_callback()
            return {'CANCELLED'}
        
        # Pass through other events
        return {'RUNNING_MODAL'}
    
    def cancel(self, context):
        """Clean up on cancel."""
        clear_point_callback()
