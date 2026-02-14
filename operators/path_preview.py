# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Live path preview operator for animating camera along the path."""

import time
from typing import Optional, Tuple

import lichtfeld as lf
from lfs_plugins.types import Operator, Event


# Module-level state for preview
_preview_active = False
_preview_path = None
_preview_fov = 60.0
_preview_start_time = 0.0
_preview_speed_multiplier = 1.0

# Original camera state to restore
_original_camera = None  # (eye, target, up, fov)


def is_preview_active() -> bool:
    """Check if preview is currently running."""
    return _preview_active


def get_preview_progress() -> float:
    """Get current preview progress (0-1)."""
    global _preview_path, _preview_start_time, _preview_speed_multiplier
    if not _preview_active or _preview_path is None:
        return 0.0
    
    elapsed = (time.time() - _preview_start_time) * _preview_speed_multiplier
    total_dur = _preview_path.get_total_duration()
    if total_dur <= 0:
        return 1.0
    return min(1.0, elapsed / total_dur)


def start_preview(path, fov: float = 60.0, speed_multiplier: float = 1.0):
    """Start the preview with given path and settings.
    
    Args:
        path: LinearPath object to preview
        fov: Field of view for preview
        speed_multiplier: Speed multiplier (1.0 = normal, 2.0 = 2x speed)
    """
    global _preview_path, _preview_fov, _preview_speed_multiplier, _original_camera
    
    _preview_path = path
    _preview_fov = fov
    _preview_speed_multiplier = speed_multiplier
    
    # Try to save original camera state
    try:
        cam = lf.get_camera()
        if cam:
            _original_camera = (
                tuple(cam.eye) if hasattr(cam, 'eye') else None,
                tuple(cam.target) if hasattr(cam, 'target') else None,
                tuple(cam.up) if hasattr(cam, 'up') else (0, 0, 1),
                cam.fov if hasattr(cam, 'fov') else 60.0
            )
    except:
        _original_camera = None
    
    # Invoke the preview operator
    op_id = "lfs_plugins.360_record.operators.path_preview.PATHPREVIEW_OT_preview"
    lf.ui.ops.invoke(op_id)


def stop_preview():
    """Stop the preview and restore original camera."""
    global _preview_active
    _preview_active = False
    lf.ui.ops.cancel_modal()


_api_warning_shown = False

def _set_viewport_camera(eye, target, up):
    """Attempt to set the viewport camera.
    
    Note: LFS currently doesn't have an API to move the viewport camera.
    lf.look_at() only computes view matrices, it doesn't set the camera.
    """
    global _api_warning_shown
    
    if not _api_warning_shown:
        lf.log.warning(
            "360_record: Live preview requires a camera control API that LFS doesn't currently provide. "
            "The lf.look_at() function only computes view matrices but doesn't move the viewport camera. "
            "Please request 'lf.set_camera(eye, target, up)' API from LFS developers."
        )
        _api_warning_shown = True
    
    return False


def restore_camera():
    """Restore the original camera state."""
    global _original_camera
    if _original_camera is not None:
        eye, target, up, fov = _original_camera
        if eye and target:
            _set_viewport_camera(eye, target, up or (0, 0, 1))


class PATHPREVIEW_OT_preview(Operator):
    """Modal operator for live camera path preview."""
    
    label = "Preview Path"
    description = "Animate camera along the path"
    options = {'BLOCKING'}
    
    def invoke(self, context, event: Event) -> set:
        """Start the preview."""
        global _preview_active, _preview_start_time
        _preview_active = True
        _preview_start_time = time.time()
        return {'RUNNING_MODAL'}
    
    def modal(self, context, event: Event) -> set:
        """Update camera position each frame."""
        global _preview_active, _preview_path, _preview_fov, _preview_start_time, _preview_speed_multiplier
        
        if not _preview_active:
            restore_camera()
            return {'CANCELLED'}
        
        # Check for cancel
        if event.type in {'RIGHTMOUSE', 'ESC'}:
            _preview_active = False
            restore_camera()
            return {'CANCELLED'}
        
        # Update camera position
        if _preview_path is not None:
            elapsed = (time.time() - _preview_start_time) * _preview_speed_multiplier
            total_dur = _preview_path.get_total_duration()
            
            if elapsed >= total_dur:
                # Preview complete
                _preview_active = False
                restore_camera()
                return {'FINISHED'}
            
            # Get camera state at current time
            eye = _preview_path.get_camera_position(elapsed)
            target = _preview_path.get_camera_target(elapsed)
            up = _preview_path.get_up_vector(elapsed)
            
            # Set the viewport camera - try multiple possible APIs
            _set_viewport_camera(eye, target, up)
            
            lf.ui.request_redraw()
        
        return {'RUNNING_MODAL'}
    
    def cancel(self, context):
        """Clean up on cancel."""
        global _preview_active
        _preview_active = False
        restore_camera()
