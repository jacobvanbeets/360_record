# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Linear Path Panel for multi-segment camera path recording."""

import os
import threading
from typing import Optional, List, Dict, Any

import numpy as np
import lichtfeld as lf
from lfs_plugins.types import Panel

from ..core.linear_path import LinearPath, LineSegment, compute_linear_camera_position
from ..core.recorder import RecordingSettings, get_default_output_path, record_linear_video, record_linear_frames_to_folder
from ..operators.point_picker import set_point_callback, clear_point_callback, was_point_cancelled


# Module-level state for draw handler
_linear_draw_handler_registered = False
_linear_path_state = {
    'segments': [],  # List of segment dicts
    'show_preview': True,
    'smooth_factor': 0.5,
    'elevation': 1.5,
    'up_axis': 'z',
    'invert_elevation': False,
}

# Pending point pick result
_pending_point = None
_pending_point_target = None  # ("segment", idx, "start"|"end"|"poi")


def _on_point_picked(world_pos):
    """Callback when a point is picked."""
    global _pending_point, _pending_point_target
    _pending_point = world_pos
    lf.ui.request_redraw()


def _linear_path_draw_handler(ctx):
    """Draw handler for visualizing the linear path in the viewport."""
    segments = _linear_path_state['segments']
    if not segments or not _linear_path_state['show_preview']:
        return
    
    smooth_factor = _linear_path_state['smooth_factor']
    
    # Colors
    line_color = (0.2, 0.8, 1.0, 0.8)
    smooth_color = (1.0, 0.8, 0.2, 0.6)
    start_color = (0.2, 1.0, 0.2, 1.0)
    end_color = (1.0, 0.4, 0.2, 1.0)
    poi_color = (1.0, 0.2, 0.8, 1.0)
    transition_color = (0.5, 0.5, 1.0, 0.5)
    
    # Get elevation settings
    elevation = _linear_path_state.get('elevation', 0.0)
    up_axis = _linear_path_state.get('up_axis', 'z')
    invert_elevation = _linear_path_state.get('invert_elevation', False)
    
    # Build LinearPath for smooth curve visualization
    path = LinearPath(
        smooth_factor=smooth_factor,
        elevation=elevation,
        up_axis=up_axis,
        invert_elevation=invert_elevation
    )
    for seg_data in segments:
        if seg_data.get('start') and seg_data.get('end'):
            segment = LineSegment(
                start_point=seg_data['start'],
                end_point=seg_data['end'],
                look_mode=seg_data.get('look_mode', 'forward'),
                poi=seg_data.get('poi')
            )
            path.segments.append(segment)
    
    # Draw each segment
    for i, seg_data in enumerate(segments):
        start = seg_data.get('start')
        end = seg_data.get('end')
        poi = seg_data.get('poi')
        look_mode = seg_data.get('look_mode', 'forward')
        
        # Draw start point (even if end is not set yet)
        if start:
            ctx.draw_point_3d(start, start_color, 18.0)
            screen_start = ctx.world_to_screen(start)
            if screen_start:
                ctx.draw_text_2d(
                    (screen_start[0] + 10, screen_start[1] - 8),
                    f"S{i+1}",
                    start_color
                )
        
        # Draw end point (even if start is not set yet)
        if end:
            ctx.draw_point_3d(end, end_color, 18.0)
            screen_end = ctx.world_to_screen(end)
            if screen_end:
                ctx.draw_text_2d(
                    (screen_end[0] + 10, screen_end[1] - 8),
                    f"E{i+1}",
                    end_color
                )
        
        # Draw segment line if both points are set
        if start and end:
            ctx.draw_line_3d(start, end, line_color, 2.0)
            
            # Draw direction arrow at midpoint
            mid = (
                (start[0] + end[0]) / 2,
                (start[1] + end[1]) / 2,
                (start[2] + end[2]) / 2
            )
            ctx.draw_point_3d(mid, line_color, 8.0)
        
        # Draw POI if set
        if look_mode == "poi" and poi:
            ctx.draw_point_3d(poi, poi_color, 20.0)
            screen_poi = ctx.world_to_screen(poi)
            if screen_poi:
                ctx.draw_text_2d(
                    (screen_poi[0] + 10, screen_poi[1] - 8),
                    f"POI{i+1}",
                    poi_color
                )
            # Draw line from segment midpoint to POI
            if start and end:
                mid = (
                    (start[0] + end[0]) / 2,
                    (start[1] + end[1]) / 2,
                    (start[2] + end[2]) / 2
                )
                ctx.draw_line_3d(mid, poi, (poi_color[0], poi_color[1], poi_color[2], 0.4), 1.0)
        
        # Draw transition to next segment
        if i < len(segments) - 1 and end:
            next_start = segments[i + 1].get('start')
            if next_start:
                ctx.draw_line_3d(end, next_start, transition_color, 1.5)
    
    # Draw smoothed path overlay if smooth_factor > 0
    if smooth_factor > 0 and len(path.segments) >= 1:
        total_dist = path.get_total_distance()
        if total_dist > 0:
            num_samples = max(50, int(total_dist * 10))
            prev_pos = None
            for i in range(num_samples + 1):
                t = (i / num_samples) * path.get_total_duration()
                pos = path.get_camera_position(t)
                if prev_pos is not None:
                    ctx.draw_line_3d(prev_pos, pos, smooth_color, 1.5)
                prev_pos = pos


def _ensure_linear_draw_handler():
    """Ensure the draw handler is registered."""
    global _linear_draw_handler_registered
    if not _linear_draw_handler_registered:
        try:
            lf.remove_draw_handler("linear_path_preview")
        except:
            pass
        lf.add_draw_handler("linear_path_preview", _linear_path_draw_handler, "POST_VIEW")
        _linear_draw_handler_registered = True


class LinearPathPanel(Panel):
    """Panel for configuring and recording linear camera path videos."""
    
    label = "Linear Path"
    space = "MAIN_PANEL_TAB"
    order = 31  # After Camera Track panel
    
    RESOLUTION_ITEMS = [
        ((1920, 1080), "1080p (1920x1080)"),
        ((2560, 1440), "1440p (2560x1440)"),
        ((3840, 2160), "4K (3840x2160)"),
        ((1280, 720), "720p (1280x720)"),
        ((1080, 1080), "Square (1080x1080)"),
    ]
    
    LOOK_MODE_ITEMS = [
        ("forward", "Look Forward"),
        ("poi", "Look at POI"),
    ]
    
    UP_AXIS_ITEMS = [
        ("z", "Z-Axis (Up)"),
        ("y", "Y-Axis (Up)"),
        ("x", "X-Axis (Up)"),
    ]
    
    # Walking speed in meters/second (average ~1.4 m/s)
    WALKING_SPEED = 1.4
    
    def __init__(self):
        # Segments list - each segment is a dict with start, end, look_mode, poi
        self._segments: List[Dict[str, Any]] = []
        self._expanded_segments: Dict[int, bool] = {}  # Track which segments are expanded
        
        # Picking state
        self._picking = False
        self._pick_target = None  # (segment_idx, "start"|"end"|"poi")
        
        # Path settings
        self._speed = 1.0  # units per second
        self._smooth_factor = 0.5  # 0-1
        self._elevation = 1.5  # offset above path
        self._up_axis_idx = 0  # 0=Z, 1=Y, 2=X
        self._invert_elevation = False
        
        # Recording settings
        self._resolution_idx = 0
        self._fps = 30.0
        self._fov = 60.0
        self._output_path = ""
        self._export_frames = False
        
        # Preview
        self._show_preview = True
        
        # State
        self._status_msg = ""
        self._status_is_error = False
        self._recording = False
        self._record_progress = 0.0
        self._record_thread: Optional[threading.Thread] = None
        self._record_result: Optional[tuple] = None  # (success, message)
    
    @classmethod
    def poll(cls, context) -> bool:
        return lf.has_scene()
    
    def _process_pending_point(self) -> bool:
        """Process any pending point pick."""
        global _pending_point, _pending_point_target
        if _pending_point is None:
            return False
        
        if self._pick_target:
            seg_idx, point_type = self._pick_target
            
            if seg_idx < len(self._segments):
                if point_type == "start":
                    self._segments[seg_idx]['start'] = _pending_point
                elif point_type == "end":
                    self._segments[seg_idx]['end'] = _pending_point
                elif point_type == "poi":
                    self._segments[seg_idx]['poi'] = _pending_point
                
                self._status_msg = f"Point set: ({_pending_point[0]:.2f}, {_pending_point[1]:.2f}, {_pending_point[2]:.2f})"
                self._status_is_error = False
        
        _pending_point = None
        self._picking = False
        self._pick_target = None
        
        self._update_draw_state()
        lf.ui.request_redraw()
        return True
    
    def _start_picking(self, segment_idx: int, point_type: str):
        """Start picking mode for a point."""
        global _pending_point
        _pending_point = None
        
        self._picking = True
        self._pick_target = (segment_idx, point_type)
        
        label_map = {"start": "start point", "end": "end point", "poi": "POI"}
        self._status_msg = f"Click to set {label_map.get(point_type, point_type)}..."
        self._status_is_error = False
        
        set_point_callback(_on_point_picked, point_type)
        op_id = "lfs_plugins.360_record.operators.point_picker.LINEARPATH_OT_pick_point"
        lf.ui.ops.invoke(op_id)
        lf.ui.request_redraw()
    
    def _cancel_picking(self):
        """Cancel picking mode."""
        self._picking = False
        self._pick_target = None
        clear_point_callback()
        lf.ui.ops.cancel_modal()
        self._status_msg = "Picking cancelled"
        self._status_is_error = False
        lf.ui.request_redraw()
    
    def _add_segment(self):
        """Add a new segment."""
        new_segment = {
            'start': None,
            'end': None,
            'look_mode': 'forward',
            'poi': None,
        }
        
        # If we have segments, default start to previous end
        if self._segments and self._segments[-1].get('end'):
            new_segment['start'] = self._segments[-1]['end']
        
        self._segments.append(new_segment)
        self._expanded_segments[len(self._segments) - 1] = True  # Expand new segment
        self._update_draw_state()
        lf.ui.request_redraw()
    
    def _remove_segment(self, idx: int):
        """Remove a segment."""
        if 0 <= idx < len(self._segments):
            del self._segments[idx]
            # Update expanded state indices
            new_expanded = {}
            for old_idx, expanded in self._expanded_segments.items():
                if old_idx < idx:
                    new_expanded[old_idx] = expanded
                elif old_idx > idx:
                    new_expanded[old_idx - 1] = expanded
            self._expanded_segments = new_expanded
            self._update_draw_state()
            lf.ui.request_redraw()
    
    def _move_segment_up(self, idx: int):
        """Move a segment up in the list."""
        if idx > 0:
            self._segments[idx], self._segments[idx - 1] = self._segments[idx - 1], self._segments[idx]
            # Swap expanded states
            exp_curr = self._expanded_segments.get(idx, False)
            exp_prev = self._expanded_segments.get(idx - 1, False)
            self._expanded_segments[idx] = exp_prev
            self._expanded_segments[idx - 1] = exp_curr
            self._update_draw_state()
            lf.ui.request_redraw()
    
    def _move_segment_down(self, idx: int):
        """Move a segment down in the list."""
        if idx < len(self._segments) - 1:
            self._segments[idx], self._segments[idx + 1] = self._segments[idx + 1], self._segments[idx]
            # Swap expanded states
            exp_curr = self._expanded_segments.get(idx, False)
            exp_next = self._expanded_segments.get(idx + 1, False)
            self._expanded_segments[idx] = exp_next
            self._expanded_segments[idx + 1] = exp_curr
            self._update_draw_state()
            lf.ui.request_redraw()
    
    def _update_draw_state(self):
        """Update the draw handler state."""
        _linear_path_state['segments'] = self._segments
        _linear_path_state['show_preview'] = self._show_preview
        _linear_path_state['smooth_factor'] = self._smooth_factor
        _linear_path_state['elevation'] = self._elevation
        _linear_path_state['up_axis'] = self.UP_AXIS_ITEMS[self._up_axis_idx][0]
        _linear_path_state['invert_elevation'] = self._invert_elevation
    
    def _get_path(self) -> Optional[LinearPath]:
        """Get the current linear path configuration."""
        if not self._segments:
            return None
        
        up_axis = self.UP_AXIS_ITEMS[self._up_axis_idx][0]
        
        path = LinearPath(
            speed=self._speed,
            smooth_factor=self._smooth_factor,
            elevation=self._elevation,
            up_axis=up_axis,
            invert_elevation=self._invert_elevation
        )
        
        for seg_data in self._segments:
            if seg_data.get('start') and seg_data.get('end'):
                segment = LineSegment(
                    start_point=seg_data['start'],
                    end_point=seg_data['end'],
                    look_mode=seg_data.get('look_mode', 'forward'),
                    poi=seg_data.get('poi')
                )
                path.segments.append(segment)
        
        return path if path.segments else None
    
    def _get_recording_settings(self) -> RecordingSettings:
        """Get the current recording settings."""
        resolution = self.RESOLUTION_ITEMS[self._resolution_idx][0]
        
        if not self._output_path:
            base_path = get_default_output_path()
            self._output_path = base_path.replace("_360.mp4", "_linear.mp4")
        
        return RecordingSettings(
            output_path=self._output_path,
            resolution=resolution,
            fps=self._fps,
            fov=self._fov
        )
    
    def _on_record_progress(self, progress: float, message: str):
        """Progress callback for recording."""
        self._record_progress = progress
        self._status_msg = message
        self._status_is_error = False
        lf.ui.request_redraw()
    
    def _record_thread_func(self, path: LinearPath, settings: RecordingSettings, export_frames: bool):
        """Recording function that runs in a background thread."""
        try:
            if export_frames:
                folder = os.path.splitext(settings.output_path)[0] + "_frames"
                success, msg = record_linear_frames_to_folder(path, settings, folder, self._on_record_progress)
            else:
                success, msg = record_linear_video(path, settings, self._on_record_progress)
            self._record_result = (success, msg)
        except Exception as e:
            self._record_result = (False, f"Recording failed: {str(e)}")
        lf.ui.request_redraw()
    
    def _start_recording(self):
        """Start video recording in a background thread."""
        path = self._get_path()
        if path is None:
            self._status_msg = "Add at least one complete segment first"
            self._status_is_error = True
            return
        
        settings = self._get_recording_settings()
        self._recording = True
        self._record_progress = 0.0
        self._record_result = None
        self._status_msg = "Starting recording..."
        self._status_is_error = False
        
        # Start recording in a background thread
        self._record_thread = threading.Thread(
            target=self._record_thread_func,
            args=(path, settings, self._export_frames),
            daemon=True
        )
        self._record_thread.start()
    
    def _check_recording_complete(self):
        """Check if recording thread has completed."""
        if self._recording and self._record_thread is not None:
            if not self._record_thread.is_alive():
                # Thread completed
                self._recording = False
                self._record_thread = None
                if self._record_result:
                    success, msg = self._record_result
                    self._status_msg = msg
                    self._status_is_error = not success
                    self._record_result = None
    
    def _format_point(self, point: Optional[tuple]) -> str:
        """Format a point for display."""
        if point is None:
            return "Not set"
        return f"({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})"
    
    def draw(self, layout):
        theme = lf.ui.theme()
        scale = layout.get_dpi_scale()
        
        # Check if recording thread completed
        self._check_recording_complete()
        
        # Process pending point pick
        self._process_pending_point()
        
        # Check if picking was cancelled
        if was_point_cancelled() and self._picking:
            self._picking = False
            self._pick_target = None
            self._status_msg = "Picking cancelled"
            self._status_is_error = False
        
        # Ensure draw handler
        _ensure_linear_draw_handler()
        
        # Update draw state
        self._update_draw_state()
        
        # === Segments Section ===
        if layout.collapsing_header("Path Segments", default_open=True):
            if not self._segments:
                layout.text_colored("No segments. Click 'Add Segment' to start.", theme.palette.text_dim)
            
            # Draw each segment
            for i, seg_data in enumerate(self._segments):
                is_expanded = self._expanded_segments.get(i, True)
                
                # Segment header
                header_text = f"Segment {i + 1}"
                if seg_data.get('start') and seg_data.get('end'):
                    header_text += " ✓"
                else:
                    header_text += " (incomplete)"
                
                # Expand/collapse button
                if layout.button(f"{'▼' if is_expanded else '▶'}##{i}_expand", (20 * scale, 0)):
                    self._expanded_segments[i] = not is_expanded
                layout.same_line()
                
                # Header label
                layout.label(header_text)
                layout.same_line()
                
                # Move buttons
                btn_w = 25 * scale
                if i > 0:
                    if layout.button(f"↑##{i}_up", (btn_w, 0)):
                        self._move_segment_up(i)
                    layout.same_line()
                
                if i < len(self._segments) - 1:
                    if layout.button(f"↓##{i}_down", (btn_w, 0)):
                        self._move_segment_down(i)
                    layout.same_line()
                
                # Delete button
                if layout.button(f"✕##{i}_del", (btn_w, 0)):
                    self._remove_segment(i)
                    continue  # Skip rest of this segment
                
                # Segment details (if expanded)
                if is_expanded:
                    layout.indent(15 * scale)
                    
                    # Check if this segment's start is connected to previous segment's end
                    is_connected = i > 0 and seg_data.get('start') is not None
                    
                    # Start point
                    layout.label("Start:")
                    layout.same_line()
                    
                    if is_connected:
                        # Show as connected (linked to previous segment)
                        layout.text_colored(self._format_point(seg_data.get('start')), (0.5, 0.8, 1.0, 1.0))
                        layout.same_line()
                        layout.text_colored("(linked)", theme.palette.text_dim)
                    else:
                        # First segment or not connected - show picker
                        layout.text_colored(self._format_point(seg_data.get('start')), 
                                           (0.2, 1.0, 0.2, 1.0) if seg_data.get('start') else theme.palette.text_dim)
                        layout.same_line()
                        
                        picking_start = self._picking and self._pick_target == (i, "start")
                        if picking_start:
                            if layout.button_styled(f"[Cancel]##{i}_start_cancel", "error", (70 * scale, 0)):
                                self._cancel_picking()
                        else:
                            if layout.button(f"Pick##{i}_start", (50 * scale, 0)):
                                self._start_picking(i, "start")
                    
                    # End point
                    layout.label("End:")
                    layout.same_line()
                    layout.text_colored(self._format_point(seg_data.get('end')),
                                       (1.0, 0.4, 0.2, 1.0) if seg_data.get('end') else theme.palette.text_dim)
                    layout.same_line()
                    
                    picking_end = self._picking and self._pick_target == (i, "end")
                    if picking_end:
                        if layout.button_styled(f"[Cancel]##{i}_end_cancel", "error", (70 * scale, 0)):
                            self._cancel_picking()
                    else:
                        if layout.button(f"Pick##{i}_end", (50 * scale, 0)):
                            self._start_picking(i, "end")
                    
                    # Look mode
                    layout.label("Look Mode:")
                    look_mode_labels = [item[1] for item in self.LOOK_MODE_ITEMS]
                    current_look_idx = 0
                    for idx, item in enumerate(self.LOOK_MODE_ITEMS):
                        if item[0] == seg_data.get('look_mode', 'forward'):
                            current_look_idx = idx
                            break
                    
                    layout.push_item_width(150 * scale)
                    changed, new_look_idx = layout.combo(f"##look_mode_{i}", current_look_idx, look_mode_labels)
                    if changed:
                        seg_data['look_mode'] = self.LOOK_MODE_ITEMS[new_look_idx][0]
                        self._update_draw_state()
                    layout.pop_item_width()
                    
                    # POI (only if look_mode is "poi")
                    if seg_data.get('look_mode') == 'poi':
                        layout.label("POI:")
                        layout.same_line()
                        layout.text_colored(self._format_point(seg_data.get('poi')),
                                           (1.0, 0.2, 0.8, 1.0) if seg_data.get('poi') else theme.palette.text_dim)
                        layout.same_line()
                        
                        picking_poi = self._picking and self._pick_target == (i, "poi")
                        if picking_poi:
                            if layout.button_styled(f"[Cancel]##{i}_poi_cancel", "error", (70 * scale, 0)):
                                self._cancel_picking()
                        else:
                            if layout.button(f"Pick##{i}_poi", (50 * scale, 0)):
                                self._start_picking(i, "poi")
                    
                    layout.unindent(15 * scale)
                    layout.spacing()
            
            # Add segment button
            layout.spacing()
            if layout.button("+ Add Segment##add_seg", (-1, 30 * scale)):
                self._add_segment()
        
        layout.separator()
        
        # === Path Settings Section ===
        if layout.collapsing_header("Path Settings", default_open=True):
            btn_w = 45 * scale
            
            # Up Axis (camera level)
            layout.label("Camera Up Axis:")
            up_axis_labels = [item[1] for item in self.UP_AXIS_ITEMS]
            changed, self._up_axis_idx = layout.combo("##upaxis", self._up_axis_idx, up_axis_labels)
            if changed:
                self._update_draw_state()
            if layout.is_item_hovered():
                layout.set_tooltip("Sets which axis points up for the camera")
            
            layout.spacing()
            
            # Elevation offset
            axis_name = self.UP_AXIS_ITEMS[self._up_axis_idx][0].upper()
            layout.label(f"Elevation (offset along {axis_name}):")
            
            layout.push_item_width(-80 * scale)
            changed, self._elevation = layout.slider_float("##elev_slider", self._elevation, 0.0, 50.0)
            if changed:
                self._update_draw_state()
            layout.pop_item_width()
            layout.same_line()
            layout.push_item_width(70 * scale)
            changed, self._elevation = layout.input_float("##elev_input", self._elevation, 0.0, 0.0)
            if changed:
                self._elevation = max(0.0, self._elevation)
                self._update_draw_state()
            layout.pop_item_width()
            
            # Elevation adjustment buttons
            if layout.button("-0.5##elevsub05", (btn_w, 0)):
                self._elevation = max(0.0, self._elevation - 0.5)
                self._update_draw_state()
            layout.same_line()
            if layout.button("+0.5##elevadd05", (btn_w, 0)):
                self._elevation += 0.5
                self._update_draw_state()
            layout.same_line()
            if layout.button("-1##elevsub1", (btn_w, 0)):
                self._elevation = max(0.0, self._elevation - 1.0)
                self._update_draw_state()
            layout.same_line()
            if layout.button("+1##elevadd1", (btn_w, 0)):
                self._elevation += 1.0
                self._update_draw_state()
            
            # Invert elevation checkbox
            changed, self._invert_elevation = layout.checkbox("Invert Elevation##invert_elev", self._invert_elevation)
            if changed:
                self._update_draw_state()
            if layout.is_item_hovered():
                layout.set_tooltip("Flip offset direction (e.g., below instead of above)")
            
            layout.spacing()
            
            # Speed (units per second)
            layout.label("Speed (units/second):")
            
            layout.push_item_width(-80 * scale)
            changed, self._speed = layout.slider_float("##speed_slider", self._speed, 0.1, 10.0)
            layout.pop_item_width()
            layout.same_line()
            layout.push_item_width(70 * scale)
            changed, self._speed = layout.input_float("##speed_input", self._speed, 0.0, 0.0)
            if changed:
                self._speed = max(0.01, self._speed)
            layout.pop_item_width()
            
            # Speed adjustment buttons + Walking speed
            if layout.button("-0.1##spdsub01", (btn_w, 0)):
                self._speed = max(0.01, self._speed - 0.1)
            layout.same_line()
            if layout.button("+0.1##spdadd01", (btn_w, 0)):
                self._speed += 0.1
            layout.same_line()
            if layout.button("-1##spdsub1", (btn_w, 0)):
                self._speed = max(0.01, self._speed - 1.0)
            layout.same_line()
            if layout.button("+1##spdadd1", (btn_w, 0)):
                self._speed += 1.0
            
            # Walking speed button
            if layout.button("Walking Speed (1.4 m/s)##walkspeed", (-1, 0)):
                self._speed = self.WALKING_SPEED
            if layout.is_item_hovered():
                layout.set_tooltip("Set to average human walking speed (assumes units are meters)")
            
            # Show estimated duration based on current speed
            path = self._get_path()
            if path:
                total_dur = path.get_total_duration()
                mins = int(total_dur // 60)
                secs = total_dur % 60
                if mins > 0:
                    dur_str = f"{mins}m {secs:.1f}s"
                else:
                    dur_str = f"{secs:.1f}s"
                layout.text_colored(
                    f"Estimated video duration: {dur_str}",
                    (0.4, 1.0, 0.4, 1.0)
                )
            
            layout.spacing()
            
            # Smooth factor
            layout.label("Smoothing:")
            layout.push_item_width(-1)
            changed, self._smooth_factor = layout.slider_float("##smooth", self._smooth_factor, 0.0, 1.0)
            if changed:
                self._update_draw_state()
            layout.pop_item_width()
            if layout.is_item_hovered():
                layout.set_tooltip("0 = Sharp corners, 1 = Maximum smoothing at transitions")
        
        layout.separator()
        
        # === Preview Section ===
        if layout.collapsing_header("Preview", default_open=True):
            changed, self._show_preview = layout.checkbox("Show Path Preview##showpreview", self._show_preview)
            if changed:
                self._update_draw_state()
                lf.ui.request_redraw()
            
            layout.text_colored(
                "Path visualization shown in viewport",
                theme.palette.text_dim
            )
            
            # Show path info
            path = self._get_path()
            if path:
                total_dist = path.get_total_distance()
                total_dur = path.get_total_duration()
                layout.text_colored(
                    f"Distance: {total_dist:.2f} units | Duration: {total_dur:.1f}s",
                    theme.palette.text_dim
                )
        
        layout.separator()
        
        # === Recording Settings Section ===
        if layout.collapsing_header("Recording Settings", default_open=True):
            # Resolution
            resolution_labels = [item[1] for item in self.RESOLUTION_ITEMS]
            changed, self._resolution_idx = layout.combo("Resolution##res", self._resolution_idx, resolution_labels)
            
            # FPS
            layout.push_item_width(100 * scale)
            changed, self._fps = layout.input_float("FPS##fps", self._fps, 0.0, 0.0)
            self._fps = max(1.0, min(120.0, self._fps))
            layout.pop_item_width()
            
            # FOV
            layout.push_item_width(100 * scale)
            changed, self._fov = layout.slider_float("FOV##fov", self._fov, 30.0, 120.0)
            layout.pop_item_width()
            if layout.is_item_hovered():
                layout.set_tooltip("Field of view in degrees")
            
            layout.spacing()
            
            # Output path
            layout.label("Output Path:")
            if not self._output_path:
                base_path = get_default_output_path()
                self._output_path = base_path.replace("_360.mp4", "_linear.mp4")
            
            layout.push_item_width(-60 * scale)
            changed, self._output_path = layout.input_text("##outpath", self._output_path)
            layout.pop_item_width()
            layout.same_line()
            if layout.button("...##browse", (50 * scale, 0)):
                pass  # TODO: File browser
            
            # Export as frames
            changed, self._export_frames = layout.checkbox("Export as PNG frames##exportframes", self._export_frames)
            if layout.is_item_hovered():
                layout.set_tooltip("Export individual PNG frames instead of MP4 video")
            
            # Display estimated info
            path = self._get_path()
            if path:
                total_frames = path.get_total_frames(self._fps)
                duration = path.get_total_duration()
                layout.text_colored(
                    f"Duration: {duration:.1f}s | Frames: {total_frames}",
                    theme.palette.text_dim
                )
        
        layout.separator()
        
        # === Record Button ===
        if self._recording:
            # Show progress bar with status
            # progress_bar(fraction, overlay, width, height)
            layout.progress_bar(self._record_progress, f"{self._record_progress * 100:.0f}%", -1, 28 * scale)
            layout.text_colored(self._status_msg, theme.palette.text_dim)
        else:
            path = self._get_path()
            can_record = path is not None and len(path.segments) > 0 and not self._picking
            
            if not can_record:
                layout.text_colored("Add complete segments to enable recording", theme.palette.text_dim)
            
            button_label = "EXPORT FRAMES" if self._export_frames else "RECORD VIDEO"
            if can_record:
                if layout.button_styled(f"{button_label}##record", "primary", (-1, 40 * scale)):
                    self._start_recording()
            else:
                layout.button(f"{button_label}##record_disabled", (-1, 40 * scale))
        
        # Status message
        if self._status_msg:
            layout.spacing()
            color = (1.0, 0.4, 0.4, 1.0) if self._status_is_error else (0.4, 1.0, 0.4, 1.0)
            layout.text_colored(self._status_msg, color)
