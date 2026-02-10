# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Point of Interest picker operator - modal operator for selecting the center point."""

from typing import Optional, Tuple, Callable

import lichtfeld as lf
import lichtfeld.selection as sel
from lfs_plugins.types import Operator, Event


# Module-level callback for when POI is picked
_poi_callback: Optional[Callable[[Tuple[float, float, float]], None]] = None
_poi_cancelled = False


def set_poi_callback(callback: Callable[[Tuple[float, float, float]], None]):
    """Set the callback to invoke when a POI is picked.
    
    Args:
        callback: Function that receives the picked world position (x, y, z)
    """
    global _poi_callback, _poi_cancelled
    _poi_callback = callback
    _poi_cancelled = False


def clear_poi_callback():
    """Clear the POI callback."""
    global _poi_callback, _poi_cancelled
    _poi_callback = None
    _poi_cancelled = True


def was_poi_cancelled() -> bool:
    """Check if POI picking was cancelled and clear the flag.
    
    Returns:
        True if picking was cancelled
    """
    global _poi_cancelled
    if _poi_cancelled:
        _poi_cancelled = False
        return True
    return False


class RECORD360_OT_pick_poi(Operator):
    """Modal operator for picking a point of interest on the model."""
    
    label = "Pick Point of Interest"
    description = "Click on the model to set the point of interest for the camera track"
    options = {'BLOCKING'}
    
    def invoke(self, context, event: Event) -> set:
        """Start modal mode."""
        return {'RUNNING_MODAL'}
    
    def modal(self, context, event: Event) -> set:
        """Handle mouse events for picking."""
        global _poi_callback
        
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            # Try to pick at mouse position
            result = sel.pick_at_screen(event.mouse_region_x, event.mouse_region_y)
            
            if result is not None and _poi_callback is not None:
                # Call callback with the picked position
                _poi_callback(tuple(result.world_position))
                clear_poi_callback()
                return {'FINISHED'}
            
            # If no hit, continue waiting
            return {'RUNNING_MODAL'}
        
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            clear_poi_callback()
            return {'CANCELLED'}
        
        # Pass through other events
        return {'RUNNING_MODAL'}
    
    def cancel(self, context):
        """Clean up on cancel."""
        clear_poi_callback()
