# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Camera Track Panel for 360 video recording."""

import os
from typing import Optional

import numpy as np
import lichtfeld as lf
from lfs_plugins.types import Panel

from ..core.camera_track import CameraTrack, compute_camera_position
from ..core.recorder import RecordingSettings, get_default_output_path, record_circular_video, preview_camera_position
from ..operators.poi_picker import set_poi_callback, clear_poi_callback, was_poi_cancelled


# Module-level state for draw handler
_draw_handler_registered = False
_track_state = {
    'poi': None,  # (x, y, z) or None
    'elevation': 1.5,
    'radius': 5.0,
    'starting_angle': 0.0,
    'show_preview': True,
}

# Pending POI pick result
_pending_poi = None


def _on_poi_picked(world_pos):
    """Callback when POI is picked."""
    global _pending_poi
    _pending_poi = world_pos
    lf.ui.request_redraw()


def _camera_track_draw_handler(ctx):
    """Draw handler for visualizing the camera track in the viewport."""
    poi = _track_state['poi']
    if poi is None or not _track_state['show_preview']:
        return
    
    elevation = _track_state['elevation']
    radius = _track_state['radius']
    starting_angle = _track_state['starting_angle']
    
    # Draw the POI marker (red sphere)
    ctx.draw_point_3d(poi, (1.0, 0.2, 0.2, 1.0), 25.0)
    
    # Draw POI label
    screen_poi = ctx.world_to_screen(poi)
    if screen_poi:
        ctx.draw_text_2d(
            (screen_poi[0] + 15, screen_poi[1] - 10),
            f"POI ({poi[0]:.1f}, {poi[1]:.1f}, {poi[2]:.1f})",
            (1.0, 0.8, 0.2, 1.0)
        )
    
    # Draw the circular track
    track_center = (poi[0], poi[1], poi[2] + elevation)
    num_segments = 64
    track_color = (0.2, 0.8, 1.0, 0.8)
    
    prev_point = None
    for i in range(num_segments + 1):
        angle = (i / num_segments) * 360.0
        pos = compute_camera_position(poi, elevation, radius, angle)
        
        if prev_point is not None:
            ctx.draw_line_3d(prev_point, pos, track_color, 2.0)
        prev_point = pos
    
    # Draw vertical line from POI to track center
    ctx.draw_line_3d(poi, track_center, (0.5, 0.5, 0.5, 0.5), 1.0)
    
    # Draw starting point marker (green)
    start_pos = compute_camera_position(poi, elevation, radius, starting_angle)
    ctx.draw_point_3d(start_pos, (0.2, 1.0, 0.2, 1.0), 20.0)
    
    screen_start = ctx.world_to_screen(start_pos)
    if screen_start:
        ctx.draw_text_2d(
            (screen_start[0] + 12, screen_start[1] - 8),
            "Start",
            (0.2, 1.0, 0.2, 1.0)
        )
    
    # Draw line from starting point to POI (camera look direction)
    ctx.draw_line_3d(start_pos, poi, (0.2, 1.0, 0.2, 0.5), 1.5)
    
    # Draw radius indicator
    radius_end = compute_camera_position(poi, elevation, radius, starting_angle + 90)
    ctx.draw_line_3d(track_center, radius_end, (1.0, 1.0, 0.2, 0.5), 1.0)


def _ensure_draw_handler():
    """Ensure the draw handler is registered."""
    global _draw_handler_registered
    if not _draw_handler_registered:
        try:
            lf.remove_draw_handler("camera_track_preview")
        except:
            pass
        lf.add_draw_handler("camera_track_preview", _camera_track_draw_handler, "POST_VIEW")
        _draw_handler_registered = True


class CameraTrackPanel(Panel):
    """Panel for configuring and recording circular camera track videos."""
    
    label = "Camera Track"
    space = "MAIN_PANEL_TAB"
    order = 30
    
    RESOLUTION_ITEMS = [
        ((1920, 1080), "1080p (1920x1080)"),
        ((2560, 1440), "1440p (2560x1440)"),
        ((3840, 2160), "4K (3840x2160)"),
        ((1280, 720), "720p (1280x720)"),
        ((1080, 1080), "Square (1080x1080)"),
    ]
    
    def __init__(self):
        # Point of Interest
        self._poi: Optional[tuple] = None
        self._picking_poi = False
        
        # Track settings
        self._elevation = 1.5
        self._elevation_units = "units"  # or "relative"
        self._radius = 5.0
        self._speed = 30.0  # seconds per circle
        self._starting_angle = 0.0
        
        # Recording settings
        self._resolution_idx = 0
        self._fps = 30.0
        self._quality = 85
        self._output_path = ""
        
        # Preview
        self._show_preview = True
        self._preview_angle = 0.0
        
        # State
        self._status_msg = ""
        self._status_is_error = False
        self._recording = False
        self._record_progress = 0.0
    
    @classmethod
    def poll(cls, context) -> bool:
        return lf.has_scene()
    
    def _process_pending_poi(self) -> bool:
        """Process any pending POI pick."""
        global _pending_poi
        if _pending_poi is None:
            return False
        
        self._poi = _pending_poi
        _pending_poi = None
        self._picking_poi = False
        
        self._status_msg = f"POI set: ({self._poi[0]:.2f}, {self._poi[1]:.2f}, {self._poi[2]:.2f})"
        self._status_is_error = False
        
        # Update track state for draw handler
        _track_state['poi'] = self._poi
        
        lf.ui.request_redraw()
        return True
    
    def _start_poi_picking(self):
        """Start picking mode for POI."""
        global _pending_poi
        _pending_poi = None
        
        self._picking_poi = True
        self._status_msg = "Click on model to set Point of Interest..."
        self._status_is_error = False
        
        set_poi_callback(_on_poi_picked)
        op_id = "lfs_plugins.360_record.operators.poi_picker.RECORD360_OT_pick_poi"
        lf.ui.ops.invoke(op_id)
        lf.ui.request_redraw()
    
    def _cancel_poi_picking(self):
        """Cancel POI picking mode."""
        self._picking_poi = False
        clear_poi_callback()
        lf.ui.ops.cancel_modal()
        self._status_msg = "Picking cancelled"
        self._status_is_error = False
        lf.ui.request_redraw()
    
    def _get_track(self) -> Optional[CameraTrack]:
        """Get the current camera track configuration."""
        if self._poi is None:
            return None
        
        return CameraTrack(
            poi=self._poi,
            elevation=self._elevation,
            radius=self._radius,
            speed=self._speed,
            starting_angle=self._starting_angle
        )
    
    def _get_recording_settings(self) -> RecordingSettings:
        """Get the current recording settings."""
        resolution = self.RESOLUTION_ITEMS[self._resolution_idx][0]
        
        # Set default output path if empty
        if not self._output_path:
            self._output_path = get_default_output_path()
        
        return RecordingSettings(
            output_path=self._output_path,
            resolution=resolution,
            fps=self._fps,
            quality=self._quality
        )
    
    def _on_record_progress(self, progress: float, message: str):
        """Progress callback for recording."""
        self._record_progress = progress
        self._status_msg = message
        self._status_is_error = False
        lf.ui.request_redraw()
    
    def _start_recording(self):
        """Start video recording."""
        track = self._get_track()
        if track is None:
            self._status_msg = "Set a Point of Interest first"
            self._status_is_error = True
            return
        
        settings = self._get_recording_settings()
        self._recording = True
        self._record_progress = 0.0
        
        # Run recording (this will block the UI - in a real implementation,
        # this should be done in a background thread with proper synchronization)
        success, msg = record_circular_video(track, settings, self._on_record_progress)
        
        self._recording = False
        self._status_msg = msg
        self._status_is_error = not success
    
    def _preview_at_angle(self, angle: float):
        """Move camera to preview position at given angle."""
        track = self._get_track()
        if track is None:
            return
        
        # Create a track with the preview angle as starting angle
        preview_track = CameraTrack(
            poi=track.poi,
            elevation=track.elevation,
            radius=track.radius,
            speed=track.speed,
            starting_angle=angle
        )
        
        preview_camera_position(preview_track, angle)
    
    def draw(self, layout):
        theme = lf.ui.theme()
        scale = layout.get_dpi_scale()
        
        # Process pending POI pick
        self._process_pending_poi()
        
        # Check if picking was cancelled
        if was_poi_cancelled() and self._picking_poi:
            self._picking_poi = False
            self._status_msg = "POI picking cancelled"
            self._status_is_error = False
        
        # Ensure draw handler for track visualization
        _ensure_draw_handler()
        
        # Update track state for draw handler
        _track_state['poi'] = self._poi
        _track_state['elevation'] = self._elevation
        _track_state['radius'] = self._radius
        _track_state['starting_angle'] = self._starting_angle
        _track_state['show_preview'] = self._show_preview
        
        # === Point of Interest Section ===
        if layout.collapsing_header("Point of Interest", default_open=True):
            if self._poi:
                layout.text_colored(
                    f"POI: ({self._poi[0]:.2f}, {self._poi[1]:.2f}, {self._poi[2]:.2f})",
                    (1.0, 0.8, 0.2, 1.0)
                )
            else:
                layout.text_colored("No POI set", theme.palette.text_dim)
            
            # Pick button
            if self._picking_poi:
                if layout.button_styled("[x] Cancel Picking##cancelpoi", "error", (-1, 32 * scale)):
                    self._cancel_poi_picking()
            else:
                if layout.button("Pick Point of Interest##pickpoi", (-1, 32 * scale)):
                    self._start_poi_picking()
            
            if layout.is_item_hovered():
                layout.set_tooltip("Click on the model to set the center point for the camera orbit")
        
        layout.separator()
        
        # === Track Settings Section ===
        settings_changed = False
        
        if layout.collapsing_header("Track Settings", default_open=True):
            # Elevation
            layout.label("Elevation (height above POI):")
            layout.push_item_width(-1)
            changed, self._elevation = layout.slider_float(
                "##elevation", self._elevation, 0.0, 20.0, "%.2f units"
            )
            settings_changed |= changed
            layout.pop_item_width()
            
            # Fine adjustment buttons for elevation
            btn_w = 45 * scale
            if layout.button("-1##elevsub1", (btn_w, 0)):
                self._elevation = max(0.0, self._elevation - 1.0)
                settings_changed = True
            layout.same_line()
            if layout.button("+1##elevadd1", (btn_w, 0)):
                self._elevation += 1.0
                settings_changed = True
            layout.same_line()
            if layout.button("-0.1##elevsub01", (btn_w, 0)):
                self._elevation = max(0.0, self._elevation - 0.1)
                settings_changed = True
            layout.same_line()
            if layout.button("+0.1##elevadd01", (btn_w, 0)):
                self._elevation += 0.1
                settings_changed = True
            
            layout.spacing()
            
            # Radius
            layout.label("Radius (distance from POI):")
            layout.push_item_width(-1)
            changed, self._radius = layout.slider_float(
                "##radius", self._radius, 0.5, 50.0, "%.2f units"
            )
            settings_changed |= changed
            layout.pop_item_width()
            
            # Fine adjustment buttons for radius
            if layout.button("-1##radsub1", (btn_w, 0)):
                self._radius = max(0.5, self._radius - 1.0)
                settings_changed = True
            layout.same_line()
            if layout.button("+1##radadd1", (btn_w, 0)):
                self._radius += 1.0
                settings_changed = True
            layout.same_line()
            if layout.button("-0.1##radsub01", (btn_w, 0)):
                self._radius = max(0.5, self._radius - 0.1)
                settings_changed = True
            layout.same_line()
            if layout.button("+0.1##radadd01", (btn_w, 0)):
                self._radius += 0.1
                settings_changed = True
            
            layout.spacing()
            
            # Speed
            layout.label("Speed (seconds per full circle):")
            layout.push_item_width(-1)
            changed, self._speed = layout.slider_float(
                "##speed", self._speed, 5.0, 120.0, "%.1f sec"
            )
            settings_changed |= changed
            layout.pop_item_width()
            
            layout.spacing()
            
            # Starting Angle
            layout.label("Starting Angle:")
            layout.push_item_width(-1)
            changed, self._starting_angle = layout.slider_float(
                "##startangle", self._starting_angle, 0.0, 360.0, "%.1f°"
            )
            settings_changed |= changed
            layout.pop_item_width()
            
            # Quick angle buttons
            if layout.button("0°##ang0", (btn_w, 0)):
                self._starting_angle = 0.0
                settings_changed = True
            layout.same_line()
            if layout.button("90°##ang90", (btn_w, 0)):
                self._starting_angle = 90.0
                settings_changed = True
            layout.same_line()
            if layout.button("180°##ang180", (btn_w, 0)):
                self._starting_angle = 180.0
                settings_changed = True
            layout.same_line()
            if layout.button("270°##ang270", (btn_w, 0)):
                self._starting_angle = 270.0
                settings_changed = True
        
        layout.separator()
        
        # === Preview Section ===
        if layout.collapsing_header("Preview", default_open=True):
            changed, self._show_preview = layout.checkbox("Show Track Preview##showpreview", self._show_preview)
            if changed:
                lf.ui.request_redraw()
            
            if self._poi is not None:
                layout.spacing()
                layout.label("Preview camera position:")
                layout.push_item_width(-1)
                changed, self._preview_angle = layout.slider_float(
                    "##previewangle", self._preview_angle, 0.0, 360.0, "%.1f°"
                )
                layout.pop_item_width()
                
                if layout.button("Go to Preview Position##gotoprev", (-1, 28 * scale)):
                    self._preview_at_angle(self._preview_angle)
                
                if layout.button("Go to Start Position##gotostart", (-1, 0)):
                    self._preview_at_angle(self._starting_angle)
        
        layout.separator()
        
        # === Recording Settings Section ===
        if layout.collapsing_header("Recording Settings", default_open=True):
            # Resolution
            resolution_labels = [item[1] for item in self.RESOLUTION_ITEMS]
            changed, self._resolution_idx = layout.combo("Resolution##res", self._resolution_idx, resolution_labels)
            
            # FPS
            layout.push_item_width(100 * scale)
            changed, self._fps = layout.input_float("FPS##fps", self._fps, 0.0, 0.0, "%.0f")
            self._fps = max(1.0, min(120.0, self._fps))
            layout.pop_item_width()
            
            # Quality
            layout.push_item_width(100 * scale)
            changed, self._quality = layout.slider_int("Quality##qual", self._quality, 10, 100)
            layout.pop_item_width()
            
            layout.spacing()
            
            # Output path
            layout.label("Output Path:")
            if not self._output_path:
                self._output_path = get_default_output_path()
            
            layout.push_item_width(-60 * scale)
            changed, self._output_path = layout.input_text("##outpath", self._output_path)
            layout.pop_item_width()
            layout.same_line()
            if layout.button("...##browse", (50 * scale, 0)):
                # TODO: Open file browser dialog
                pass
            
            # Display estimated file info
            track = self._get_track()
            if track:
                total_frames = track.get_total_frames(self._fps)
                duration = self._speed
                layout.text_colored(
                    f"Duration: {duration:.1f}s | Frames: {total_frames}",
                    theme.palette.text_dim
                )
        
        layout.separator()
        
        # === Record Button ===
        if self._recording:
            # Show progress
            layout.progress_bar(self._record_progress, (-1, 28 * scale), f"{self._record_progress * 100:.0f}%")
        else:
            can_record = self._poi is not None and not self._picking_poi
            
            if not can_record:
                layout.text_colored("Set a Point of Interest to enable recording", theme.palette.text_dim)
            
            # Record button
            if can_record:
                if layout.button_styled("RECORD CIRCULAR VIDEO##record", "primary", (-1, 40 * scale)):
                    self._start_recording()
            else:
                layout.button("RECORD CIRCULAR VIDEO##record_disabled", (-1, 40 * scale))
        
        # Status message
        if self._status_msg:
            layout.spacing()
            color = (1.0, 0.4, 0.4, 1.0) if self._status_is_error else (0.4, 1.0, 0.4, 1.0)
            layout.text_colored(self._status_msg, color)
