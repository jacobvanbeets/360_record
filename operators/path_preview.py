# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Live path preview using draw callback for smooth animation."""

import time

import lichtfeld as lf
from lfs_plugins.types import Operator, Event


# Module-level state for preview
_preview_active = False
_preview_path = None
_preview_fov = 60.0
_preview_start_time = 0.0
_preview_speed_multiplier = 1.0
_draw_handler = None

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


def _set_viewport_camera(eye, target, up):
    """Set the viewport camera position and orientation."""
    try:
        lf.set_camera(tuple(eye), tuple(target), tuple(up))
        return True
    except Exception as e:
        lf.log.error(f"360_record: Failed to set camera: {e}")
        return False


def _restore_camera():
    """Restore the original camera state."""
    global _original_camera
    if _original_camera is not None:
        eye, target, up, fov = _original_camera
        if eye and target:
            _set_viewport_camera(eye, target, up or (0, 0, 1))
            try:
                lf.set_camera_fov(fov)
            except:
                pass


def _preview_draw_callback(ctx):
    """Draw callback that updates camera each frame."""
    global _preview_active, _preview_path, _preview_start_time, _preview_speed_multiplier
    
    if not _preview_active or _preview_path is None:
        return
    
    elapsed = (time.time() - _preview_start_time) * _preview_speed_multiplier
    total_dur = _preview_path.get_total_duration()
    
    if elapsed >= total_dur:
        # Preview complete
        _stop_preview_internal()
        return
    
    # Get camera state at current time
    eye = _preview_path.get_camera_position(elapsed)
    target = _preview_path.get_camera_target(elapsed)
    up = _preview_path.get_up_vector(elapsed)
    
    # Set the viewport camera
    _set_viewport_camera(eye, target, up)
    
    # Request next frame
    lf.ui.request_redraw()


def _stop_preview_internal():
    """Internal stop - restore camera and clean up."""
    global _preview_active, _draw_handler
    
    _preview_active = False
    
    # Unregister draw handler
    if _draw_handler:
        try:
            lf.remove_draw_handler(_draw_handler)
        except:
            pass
        _draw_handler = None
    
    # Restore original camera
    _restore_camera()
    
    # Cancel modal if running
    try:
        lf.ui.ops.cancel_modal()
    except:
        pass


def start_preview(path, fov: float = 60.0, speed_multiplier: float = 1.0):
    """Start the preview with given path and settings.
    
    Args:
        path: LinearPath object to preview
        fov: Field of view for preview
        speed_multiplier: Speed multiplier (1.0 = normal, 2.0 = 2x speed)
    """
    global _preview_path, _preview_fov, _preview_speed_multiplier, _original_camera
    global _preview_active, _preview_start_time, _draw_handler
    
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
    
    # Set FOV
    try:
        lf.set_camera_fov(fov)
    except:
        pass
    
    # Start the preview
    _preview_active = True
    _preview_start_time = time.time()
    
    # Register draw handler for continuous updates
    try:
        _draw_handler = "preview_camera_update"
        lf.add_draw_handler(_draw_handler, _preview_draw_callback, "POST_VIEW")
    except:
        _draw_handler = None
    
    # Start modal operator for ESC handling
    op_id = "lfs_plugins.360_record.operators.path_preview.PATHPREVIEW_OT_preview"
    lf.ui.ops.invoke(op_id)
    
    # Trigger first update
    lf.ui.request_redraw()


def stop_preview():
    """Stop the preview and restore original camera."""
    _stop_preview_internal()


class PATHPREVIEW_OT_preview(Operator):
    """Modal operator for handling ESC to cancel preview."""
    
    label = "Preview Path"
    description = "Animate camera along the path"
    options = {'BLOCKING'}
    
    def invoke(self, context, event: Event) -> set:
        """Start the modal."""
        return {'RUNNING_MODAL'}
    
    def modal(self, context, event: Event) -> set:
        """Handle cancel events."""
        global _preview_active
        
        if not _preview_active:
            return {'FINISHED'}
        
        # Check for cancel
        if event.type in {'RIGHTMOUSE', 'ESC'}:
            _stop_preview_internal()
            return {'CANCELLED'}
        
        return {'RUNNING_MODAL'}
    
    def cancel(self, context):
        """Clean up on cancel."""
        _stop_preview_internal()
